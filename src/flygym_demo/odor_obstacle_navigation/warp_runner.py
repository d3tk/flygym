from __future__ import annotations

from copy import deepcopy
import os
import time
from typing import Any

import mujoco as mj
import mujoco_warp as mjw
import numpy as np
import warp as wp

from flygym.anatomy import BodySegment, ContactBodiesPreset, LEGS
from flygym.compose import ActuatorType
from flygym.utils.math import Rotation3D
from flygym.vision.retina import Retina
from flygym.warp import GPUSimulation
from flygym_demo.complex_terrain import make_locomotion_fly
from flygym_demo.complex_terrain.common import dof_spec_to_jointdof
from flygym_demo.complex_terrain.cpg_controller import make_tripod_cpg_network
from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps
from flygym_demo.odor_obstacle_navigation.task import (
    ACTION_PRESET_NAMES,
    ACTION_PRESET_VALUES,
    NavigationTaskConfig,
    _compute_ommatidia_coms,
    _empty_vision_features,
    _finite_float32,
    _heading_from_quat_wxyz,
    _odor_intensity_from_sources,
    _odor_by_side,
    _project_action_norm,
    _sample_navigation_layout,
    _yaw_to_quat_wxyz,
    compute_motion_diagnostics,
    navigation_policy_action_from_raw,
)
from flygym_demo.odor_obstacle_navigation.world import ObstacleOdorWorld

_N_LEGS = len(LEGS)
_N_STEP_DOFS = len(PreprogrammedSteps.dofs_per_leg)
_TWO_PI = 2 * np.pi


@wp.kernel
def _copy_row_kernel(
    dst: wp.array2d(dtype=wp.float32),
    rows: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.float32),
):
    row_idx, col_idx = wp.tid()
    dst[rows[row_idx], col_idx] = values[col_idx]


@wp.kernel
def _set_root_state_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    rows: wp.array(dtype=wp.int32),
    positions: wp.array2d(dtype=wp.float32),
    quats: wp.array2d(dtype=wp.float32),
    root_qposadr: int,
    root_qveladr: int,
):
    row_idx = wp.tid()
    world_idx = rows[row_idx]
    qpos[world_idx, root_qposadr + 0] = positions[row_idx, 0]
    qpos[world_idx, root_qposadr + 1] = positions[row_idx, 1]
    qpos[world_idx, root_qposadr + 2] = positions[row_idx, 2]
    qpos[world_idx, root_qposadr + 3] = quats[row_idx, 0]
    qpos[world_idx, root_qposadr + 4] = quats[row_idx, 1]
    qpos[world_idx, root_qposadr + 5] = quats[row_idx, 2]
    qpos[world_idx, root_qposadr + 6] = quats[row_idx, 3]
    qvel[world_idx, root_qveladr + 0] = 0.0
    qvel[world_idx, root_qveladr + 1] = 0.0
    qvel[world_idx, root_qveladr + 2] = 0.0
    qvel[world_idx, root_qveladr + 3] = 0.0
    qvel[world_idx, root_qveladr + 4] = 0.0
    qvel[world_idx, root_qveladr + 5] = 0.0


@wp.kernel
def _set_mocap_positions_kernel(
    mocap_pos: wp.array2d(dtype=wp.vec3),
    rows: wp.array(dtype=wp.int32),
    mocap_ids: wp.array(dtype=wp.int32),
    positions: wp.array3d(dtype=wp.float32),
):
    row_idx, item_idx = wp.tid()
    world_idx = rows[row_idx]
    mocap_id = mocap_ids[item_idx]
    mocap_pos[world_idx, mocap_id] = wp.vec3(
        positions[row_idx, item_idx, 0],
        positions[row_idx, item_idx, 1],
        positions[row_idx, item_idx, 2],
    )


@wp.kernel
def _accumulate_ommatidia_readouts_kernel(
    rgb: wp.array4d(dtype=wp.vec3),
    src_rows: wp.array(dtype=wp.int32),
    src_cols: wp.array(dtype=wp.int32),
    ommatidia_ids: wp.array(dtype=wp.int32),
    pale_type_mask: wp.array(dtype=wp.int32),
    num_pixels_per_ommatidium: wp.array(dtype=wp.float32),
    num_ommatidia_per_eye: int,
    readouts: wp.array2d(dtype=wp.float32),
):
    world_idx, eye_idx, pixel_idx = wp.tid()
    ommatidium_idx = ommatidia_ids[pixel_idx]
    channel_idx = pale_type_mask[ommatidium_idx]
    color = rgb[world_idx, eye_idx, src_rows[pixel_idx], src_cols[pixel_idx]]
    value = wp.float32(0.0)
    if channel_idx == 0:
        value = color[1]
    else:
        value = color[2]
    value = value / num_pixels_per_ommatidium[ommatidium_idx]
    readout_col = eye_idx * num_ommatidia_per_eye * 2 + ommatidium_idx * 2 + channel_idx
    wp.atomic_add(readouts, world_idx, readout_col, value)


@wp.kernel
def _summarize_vision_kernel(
    readouts: wp.array2d(dtype=wp.float32),
    ommatidia_coms: wp.array2d(dtype=wp.float32),
    object_threshold: float,
    num_ommatidia_per_eye: int,
    summary: wp.array2d(dtype=wp.float32),
):
    world_idx, eye_idx, ommatidium_idx = wp.tid()
    readout_col = eye_idx * num_ommatidia_per_eye * 2 + ommatidium_idx * 2
    channel_0 = readouts[world_idx, readout_col]
    channel_1 = readouts[world_idx, readout_col + 1]
    if wp.max(channel_0, channel_1) < object_threshold:
        summary_col = eye_idx * 3
        wp.atomic_add(
            summary, world_idx, summary_col + 0, ommatidia_coms[ommatidium_idx, 0]
        )
        wp.atomic_add(
            summary, world_idx, summary_col + 1, ommatidia_coms[ommatidium_idx, 1]
        )
        wp.atomic_add(summary, world_idx, summary_col + 2, 1.0)


@wp.kernel
def _cpg_update_kernel(
    actions: wp.array2d(dtype=wp.float32),
    phases: wp.array2d(dtype=wp.float32),
    magnitudes: wp.array2d(dtype=wp.float32),
    base_intrinsic_freqs: wp.array(dtype=wp.float32),
    coupling_weights: wp.array2d(dtype=wp.float32),
    phase_biases: wp.array2d(dtype=wp.float32),
    convergence_coefs: wp.array(dtype=wp.float32),
    next_phases: wp.array2d(dtype=wp.float32),
    next_magnitudes: wp.array2d(dtype=wp.float32),
    timestep: float,
    action_low: float,
    action_high: float,
):
    env_idx, leg_idx = wp.tid()
    side_idx = wp.int32(0)
    if leg_idx >= 3:
        side_idx = 1
    descending_signal = actions[env_idx, side_idx]
    if descending_signal < action_low:
        descending_signal = action_low
    if descending_signal > action_high:
        descending_signal = action_high

    intrinsic_freq = base_intrinsic_freqs[leg_idx]
    if descending_signal < 0.0:
        intrinsic_freq = -intrinsic_freq

    phase = phases[env_idx, leg_idx]
    coupling = wp.float32(0.0)
    for other_leg_idx in range(_N_LEGS):
        phase_diff = phases[env_idx, other_leg_idx] - phase
        coupling += (
            magnitudes[env_idx, other_leg_idx]
            * coupling_weights[leg_idx, other_leg_idx]
            * wp.sin(phase_diff - phase_biases[leg_idx, other_leg_idx])
        )

    dtheta_dt = wp.float32(_TWO_PI) * intrinsic_freq + coupling
    intrinsic_amp = wp.abs(descending_signal)
    dr_dt = convergence_coefs[leg_idx] * (
        intrinsic_amp - magnitudes[env_idx, leg_idx]
    )

    next_phase = phase + dtheta_dt * timestep
    next_phase = next_phase - wp.floor(next_phase / wp.float32(_TWO_PI)) * wp.float32(
        _TWO_PI
    )
    next_phases[env_idx, leg_idx] = next_phase
    next_magnitudes[env_idx, leg_idx] = magnitudes[env_idx, leg_idx] + dr_dt * timestep


@wp.kernel
def _cpg_output_kernel(
    phases: wp.array2d(dtype=wp.float32),
    magnitudes: wp.array2d(dtype=wp.float32),
    next_phases: wp.array2d(dtype=wp.float32),
    next_magnitudes: wp.array2d(dtype=wp.float32),
    spline_coeffs: wp.array4d(dtype=wp.float32),
    neutral_angles: wp.array2d(dtype=wp.float32),
    dof_to_output_idx: wp.array2d(dtype=wp.int32),
    swing_periods: wp.array2d(dtype=wp.float32),
    joint_angles: wp.array2d(dtype=wp.float32),
    adhesion: wp.array2d(dtype=wp.float32),
    phase_grid_step: float,
    n_intervals: int,
):
    env_idx, leg_idx = wp.tid()
    phase = next_phases[env_idx, leg_idx]
    magnitude = next_magnitudes[env_idx, leg_idx]
    phases[env_idx, leg_idx] = phase
    magnitudes[env_idx, leg_idx] = magnitude

    adhesion_value = wp.float32(1.0)
    if phase > swing_periods[leg_idx, 0] and phase < swing_periods[leg_idx, 1]:
        adhesion_value = wp.float32(0.0)
    adhesion[env_idx, leg_idx] = adhesion_value

    interval_idx = wp.int32(wp.floor(phase / phase_grid_step))
    if interval_idx < 0:
        interval_idx = 0
    if interval_idx >= n_intervals:
        interval_idx = n_intervals - 1
    phase_offset = phase - wp.float32(interval_idx) * phase_grid_step

    for dof_idx in range(_N_STEP_DOFS):
        output_idx = dof_to_output_idx[leg_idx, dof_idx]
        if output_idx >= 0:
            value = (
                (
                    spline_coeffs[leg_idx, dof_idx, 0, interval_idx] * phase_offset
                    + spline_coeffs[leg_idx, dof_idx, 1, interval_idx]
                )
                * phase_offset
                + spline_coeffs[leg_idx, dof_idx, 2, interval_idx]
            ) * phase_offset + spline_coeffs[leg_idx, dof_idx, 3, interval_idx]
            neutral = neutral_angles[leg_idx, dof_idx]
            joint_angles[env_idx, output_idx] = neutral + magnitude * (value - neutral)


class WarpNavigationTaskRunner:
    """Batched MuJoCo-Warp runner for the odor obstacle navigation PPO loop.

    The runner exposes the same numpy-facing API as the CPU sync/subproc runners while
    keeping physics in one `GPUSimulation` with `num_envs` parallel worlds.
    """

    def __init__(self, config: NavigationTaskConfig, num_envs: int) -> None:
        self._require_cuda_device()
        self.config = deepcopy(config)
        if self.config.render:
            raise NotImplementedError("WarpNavigationTaskRunner does not render.")
        self.num_envs = int(num_envs)
        self.action_space = self.config.action_space
        if self.action_space == "continuous":
            self.action_shape = (2,)
            self.action_n = None
            self.action_low = np.full(2, self.config.action_bounds[0], dtype=np.float32)
            self.action_high = np.full(
                2, self.config.action_bounds[1], dtype=np.float32
            )
            self.action_max_norm = self.config.action_max_norm
        else:
            self.action_shape = ()
            self.action_n = len(ACTION_PRESET_NAMES)
            self.action_low = np.array([0], dtype=np.int64)
            self.action_high = np.array([self.action_n - 1], dtype=np.int64)
            self.action_max_norm = None
        self.observation_shape = (8,)

        self.fly = make_locomotion_fly()
        if self.config.enable_vision:
            self.fly.add_vision()
        if self.config.enable_olfaction:
            self.fly.add_olfaction()
        self.world = self._make_world()
        self.world.mjcf_root.option.timestep = self.config.timestep
        self.world.add_fly(
            self.fly,
            self.config.spawn_position,
            Rotation3D("quat", _yaw_to_quat_wxyz(self.config.spawn_yaw)),
            bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
            add_ground_contact_sensors=False,
        )
        self.world.mjcf_root.option.ccd_iterations = self.config.ccd_iterations
        self.sim = GPUSimulation(self.world, n_worlds=self.num_envs)
        self.sim.mj_model.opt.timestep = self.config.timestep
        self.sim.mjw_model, self.sim.mjw_data = self.sim._mj_structs_to_mjw_structs()
        self._vision_renderer = None
        self._retina = None
        self._setup_vision()
        self._wp_device = self.sim.mjw_data.qpos.device
        if not self._wp_device.is_cuda:
            raise RuntimeError(f"Expected a CUDA Warp device, got {self._wp_device}.")

        self._root_qposadr, self._root_qveladr = self._find_root_addresses()
        self._odor_mocap_ids = self._cache_mocap_ids(
            self.world.odor_marker_body_full_identifiers()
        )
        self._obstacle_mocap_ids = self._cache_mocap_ids(
            self.world.obstacle_body_full_identifiers()
        )
        self._odor_mocap_ids_wp = wp.array(
            self._odor_mocap_ids, dtype=wp.int32, device=self._wp_device
        )
        self._obstacle_mocap_ids_wp = wp.array(
            self._obstacle_mocap_ids, dtype=wp.int32, device=self._wp_device
        )
        self._thorax_body_idx = self.fly.get_bodysegs_order().index(
            BodySegment("c_thorax")
        )
        self._olfaction_site_ids = self.sim._internal_olfaction_siteids_by_fly.get(
            self.fly.name, np.array([], dtype=np.int32)
        )
        self._actions_wp = wp.zeros(
            (self.num_envs, 2), dtype=wp.float32, device=self._wp_device
        )
        self.controller = _GPUOpenLoopTurningController(
            num_envs=self.num_envs,
            timestep=self.config.timestep,
            output_dof_order=self.fly.get_actuated_jointdofs_order(
                ActuatorType.POSITION
            ),
            action_bounds=self.config.action_bounds,
            device=self._wp_device,
        )
        self._decision_graph = self._capture_decision_graph()
        self._neutral_qpos = np.asarray(self.sim.mj_data.qpos, dtype=np.float32).copy()
        self._neutral_qvel = np.asarray(self.sim.mj_data.qvel, dtype=np.float32).copy()
        self._neutral_ctrl = np.asarray(self.sim.mj_data.ctrl, dtype=np.float32).copy()
        self._rngs = [np.random.default_rng() for _ in range(self.num_envs)]
        self._decision_counts = np.zeros(self.num_envs, dtype=np.int32)
        self._prev_distance_to_target = np.zeros(self.num_envs, dtype=np.float32)
        self._prev_object_size = np.zeros(self.num_envs, dtype=np.float32)
        self._prev_fly_xy = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._prev_heading = np.zeros(self.num_envs, dtype=np.float32)
        self._prev_policy_action_idx = np.full(self.num_envs, -1, dtype=np.int32)
        self._has_motion_history = np.zeros(self.num_envs, dtype=bool)
        self.current_odor_source = np.repeat(
            self.config.odor_source[np.newaxis, :, :], self.num_envs, axis=0
        )
        self.current_obstacle_positions = np.repeat(
            self.config.obstacle_positions[np.newaxis, :, :], self.num_envs, axis=0
        )
        self.current_active_obstacle_counts = np.full(
            self.num_envs,
            self.config.obstacle_positions.shape[0],
            dtype=np.int32,
        )
        self._diagnostic_step_count = 0
        self._diagnostic_wall_seconds = {
            "action_assign": 0.0,
            "graph_launch": 0.0,
            "state_reward": 0.0,
            "odor": 0.0,
            "vision": 0.0,
            "observation_assemble": 0.0,
            "info": 0.0,
            "total_step": 0.0,
        }
        self._last_action_source = ""

    @property
    def decision_steps(self) -> float:
        return float(self.config.decision_steps)

    @property
    def decision_interval(self) -> float:
        return float(self.config.decision_interval)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
        seeds = [
            None if seed is None else int(seed) + idx for idx in range(self.num_envs)
        ]
        return self.reset_at(range(self.num_envs), seeds)

    def reset_at(
        self,
        indices,
        seeds: list[int | None] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        indices = np.asarray(list(indices), dtype=np.int32)
        if indices.size == 0:
            return np.empty((0, *self.observation_shape), dtype=np.float32), []
        seeds = [None] * len(indices) if seeds is None else seeds
        spawn_positions = np.empty((len(indices), 3), dtype=np.float32)
        spawn_quats = np.empty((len(indices), 4), dtype=np.float32)
        for row, (env_idx, seed) in enumerate(zip(indices, seeds, strict=True)):
            env_idx = int(env_idx)
            if seed is not None:
                self._rngs[env_idx] = np.random.default_rng(seed)
            layout = _sample_navigation_layout(self.config, self._rngs[env_idx])
            spawn_positions[row] = layout.spawn_position
            spawn_quats[row] = _yaw_to_quat_wxyz(layout.spawn_yaw)
            self.current_odor_source[env_idx] = layout.odor_source
            self.current_obstacle_positions[env_idx] = layout.obstacle_positions
            self.current_active_obstacle_counts[env_idx] = layout.active_obstacle_count

        rows_wp = wp.array(indices, dtype=wp.int32)
        self._copy_rows(self.sim.mjw_data.qpos, rows_wp, self._neutral_qpos)
        self._copy_rows(self.sim.mjw_data.qvel, rows_wp, self._neutral_qvel)
        self._copy_rows(self.sim.mjw_data.ctrl, rows_wp, self._neutral_ctrl)
        wp.launch(
            _set_root_state_kernel,
            dim=len(indices),
            inputs=[
                self.sim.mjw_data.qpos,
                self.sim.mjw_data.qvel,
                rows_wp,
                wp.array(spawn_positions, dtype=wp.float32),
                wp.array(spawn_quats, dtype=wp.float32),
                self._root_qposadr,
                self._root_qveladr,
            ],
        )
        self._set_layout_mocaps(indices, rows_wp)
        self._forward()

        self.controller.reset_at(indices, seeds)
        self._decision_counts[indices] = 0
        self._prev_object_size[indices] = 0.0
        fly_positions = self._fly_positions()[indices]
        self._prev_distance_to_target[indices] = self._distance_to_target(
            fly_positions, indices
        )
        self._prev_fly_xy[indices] = fly_positions[:, :2]
        self._prev_heading[indices] = self._fly_headings()[indices]
        self._prev_policy_action_idx[indices] = -1
        self._has_motion_history[indices] = False
        observations, features = self._make_observations(indices=indices)
        infos = [
            self._make_info(
                int(env_idx),
                features[row],
                0.0,
                0.0,
                0.0,
                False,
                False,
                fly_position=fly_positions[row],
                distance=float(self._prev_distance_to_target[int(env_idx)]),
                forward_speed=0.0,
                yaw_rate=0.0,
                policy_action_idx=None,
                policy_action_label="",
                switched_action=False,
            )
            for row, env_idx in enumerate(indices)
        ]
        return observations, infos

    def step(
        self,
        actions: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        step_started_at = time.perf_counter()
        raw_actions = actions
        actions = self._resolve_actions(actions)
        policy_action_meta = self._policy_action_metadata(raw_actions)

        phase_started_at = time.perf_counter()
        self._last_action_source = self._assign_actions(actions)
        self._record_diagnostic_time("action_assign", phase_started_at)

        phase_started_at = time.perf_counter()
        wp.capture_launch(self._decision_graph)
        self._record_diagnostic_time("graph_launch", phase_started_at)

        phase_started_at = time.perf_counter()
        indices = np.arange(self.num_envs, dtype=np.int32)
        self._decision_counts += 1
        fly_positions = self._fly_positions()
        headings = self._fly_headings()
        distances = self._distance_to_target(fly_positions, indices)
        successes = distances <= self.config.target_radius
        progress = self._prev_distance_to_target - distances
        rewards = (
            self.config.progress_reward_weight * progress - self.config.time_penalty
        )
        rewards = rewards.astype(np.float32)
        rewards[successes] += self.config.success_bonus
        self._prev_distance_to_target = distances.astype(np.float32)
        terminated = successes.astype(bool)
        truncated = np.logical_and(
            ~terminated, self._decision_counts >= self.config.max_decision_steps
        )
        self._record_diagnostic_time("state_reward", phase_started_at)

        observations, features = self._make_observations(
            indices=indices,
            record_diagnostics=True,
        )

        phase_started_at = time.perf_counter()
        current_fly_xy = fly_positions[:, :2]
        infos = []
        for row, env_idx in enumerate(indices):
            env_idx = int(env_idx)
            if self._has_motion_history[env_idx]:
                forward_speed, yaw_rate = compute_motion_diagnostics(
                    self._prev_fly_xy[env_idx],
                    float(self._prev_heading[env_idx]),
                    current_fly_xy[env_idx],
                    float(headings[env_idx]),
                    self.config.decision_interval,
                )
            else:
                forward_speed, yaw_rate = 0.0, 0.0
            policy_action_idx, policy_action_label = policy_action_meta[env_idx]
            switched_action = (
                policy_action_idx is not None
                and self._prev_policy_action_idx[env_idx] >= 0
                and policy_action_idx != self._prev_policy_action_idx[env_idx]
            )
            low_level_action = None
            if self.action_space == "continuous":
                low_level_action = {
                    "left_force": float(actions[env_idx, 0]),
                    "right_force": float(actions[env_idx, 1]),
                    "groom": False,
                }
            infos.append(
                self._make_info(
                    env_idx,
                    features[row],
                    float(self.config.progress_reward_weight * progress[env_idx]),
                    float(-self.config.time_penalty),
                    float(self.config.success_bonus if terminated[env_idx] else 0.0),
                    bool(terminated[env_idx]),
                    bool(truncated[env_idx]),
                    fly_position=fly_positions[env_idx],
                    distance=float(distances[env_idx]),
                    forward_speed=forward_speed,
                    yaw_rate=yaw_rate,
                    policy_action_idx=policy_action_idx,
                    policy_action_label=policy_action_label,
                    switched_action=switched_action,
                    low_level_action=low_level_action,
                )
            )
            if policy_action_idx is not None:
                self._prev_policy_action_idx[env_idx] = policy_action_idx
        self._prev_fly_xy = current_fly_xy.astype(np.float32)
        self._prev_heading = headings.astype(np.float32)
        self._has_motion_history[:] = True
        self._record_diagnostic_time("info", phase_started_at)
        self._diagnostic_step_count += 1
        self._record_diagnostic_time("total_step", step_started_at)
        return observations, rewards, terminated, truncated, infos

    def close(self) -> None:
        self.sim.close()

    def diagnostics(self) -> list[dict[str, Any]]:
        observation_wall_seconds = (
            self._diagnostic_wall_seconds["odor"]
            + self._diagnostic_wall_seconds["vision"]
            + self._diagnostic_wall_seconds["observation_assemble"]
        )
        return [
            {
                "rank": 0,
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "cpu_affinity": _cpu_affinity(),
                "cpu_affinity_count": _cpu_affinity_count(),
                "cpu_seconds": _process_cpu_seconds(),
                "num_envs": self.num_envs,
                "warp_devices": _warp_device_names(),
                "warp_cuda_devices": _warp_cuda_device_names(),
                "selected_warp_device": str(self._wp_device),
                "warp_cuda_graph_captured": True,
                "warp_last_action_source": self._last_action_source,
                "warp_step_count": self._diagnostic_step_count,
                "warp_step_total_wall_seconds": self._diagnostic_wall_seconds[
                    "total_step"
                ],
                "warp_step_action_assign_wall_seconds": self._diagnostic_wall_seconds[
                    "action_assign"
                ],
                "warp_step_graph_launch_wall_seconds": self._diagnostic_wall_seconds[
                    "graph_launch"
                ],
                "warp_step_state_reward_wall_seconds": self._diagnostic_wall_seconds[
                    "state_reward"
                ],
                "warp_step_observation_wall_seconds": observation_wall_seconds,
                "warp_step_odor_wall_seconds": self._diagnostic_wall_seconds["odor"],
                "warp_step_vision_wall_seconds": self._diagnostic_wall_seconds[
                    "vision"
                ],
                "warp_step_observation_assemble_wall_seconds": (
                    self._diagnostic_wall_seconds["observation_assemble"]
                ),
                "warp_step_info_wall_seconds": self._diagnostic_wall_seconds["info"],
            }
        ]

    @staticmethod
    def _require_cuda_device() -> None:
        cuda_devices = _warp_cuda_device_names()
        if not cuda_devices:
            devices = ", ".join(_warp_device_names()) or "none"
            raise RuntimeError(
                "WarpNavigationTaskRunner requires a CUDA Warp device; "
                f"available Warp devices: {devices}."
            )

    def _assign_actions(self, actions: Any) -> str:
        if isinstance(actions, wp.array):
            wp.copy(self._actions_wp, actions)
            return "warp"
        if actions.__class__.__module__.startswith("torch"):
            source = wp.from_torch(actions.detach().contiguous(), dtype=wp.float32)
            wp.copy(self._actions_wp, source)
            return f"torch:{actions.device}"
        self._actions_wp.assign(np.asarray(actions, dtype=np.float32))
        return "numpy"

    def _capture_decision_graph(self):
        with wp.ScopedCapture() as capture:
            for _ in range(self.config.decision_steps):
                joint_angles, adhesion = self.controller.step(self._actions_wp)
                self.sim.set_actuator_inputs(
                    self.fly.name, ActuatorType.POSITION, joint_angles
                )
                self.sim.set_leg_adhesion_states(self.fly.name, adhesion)
                self.sim.step()
        return capture.graph

    def _record_diagnostic_time(self, name: str, started_at: float) -> None:
        self._diagnostic_wall_seconds[name] += time.perf_counter() - started_at

    def _make_world(self) -> ObstacleOdorWorld:
        return ObstacleOdorWorld(
            odor_source=self.config.odor_source,
            peak_odor_intensity=self.config.peak_odor_intensity,
            obstacle_positions=self.config.obstacle_positions,
            obstacle_radius=self.config.obstacle_radius,
            obstacle_height=self.config.obstacle_height,
            obstacle_colors=self.config.obstacle_colors,
        )

    def _resolve_actions(self, actions: Any) -> Any:
        if self.action_space == "continuous":
            if actions.__class__.__module__.startswith("torch"):
                if tuple(actions.shape) != (self.num_envs, 2):
                    raise ValueError(f"actions must have shape ({self.num_envs}, 2).")
                return actions
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape != (self.num_envs, 2):
                raise ValueError(f"actions must have shape ({self.num_envs}, 2).")
            actions = np.clip(actions, self.action_low, self.action_high)
            return np.stack(
                [
                    _project_action_norm(action, self.action_max_norm)
                    for action in actions
                ]
            ).astype(np.float32)

        if actions.__class__.__module__.startswith("torch"):
            action_indices = actions.detach().cpu().numpy()
        else:
            action_indices = np.asarray(actions)
        if action_indices.shape == (self.num_envs, 1):
            action_indices = action_indices[:, 0]
        if action_indices.shape != (self.num_envs,):
            raise ValueError(
                f"discrete actions must have shape ({self.num_envs},) "
                f"or ({self.num_envs}, 1)."
            )
        if np.issubdtype(action_indices.dtype, np.floating):
            if not np.all(np.equal(action_indices, np.floor(action_indices))):
                raise ValueError("discrete actions must be integer indices.")
        action_indices = action_indices.astype(np.int64)
        if np.any((action_indices < 0) | (action_indices >= len(ACTION_PRESET_VALUES))):
            raise ValueError(
                f"discrete action indices must be in [0, {len(ACTION_PRESET_VALUES) - 1}]."
            )
        return ACTION_PRESET_VALUES[action_indices].astype(np.float32, copy=True)

    def _copy_rows(self, dst, rows_wp, values: np.ndarray) -> None:
        wp.launch(
            _copy_row_kernel,
            dim=(rows_wp.shape[0], len(values)),
            inputs=[dst, rows_wp, wp.array(values, dtype=wp.float32)],
        )

    def _set_layout_mocaps(self, indices: np.ndarray, rows_wp) -> None:
        self._set_mocap_positions(
            rows_wp,
            self._odor_mocap_ids_wp,
            self.current_odor_source[indices].astype(np.float32, copy=False),
        )
        obstacle_positions = np.empty(
            (len(indices), self.current_obstacle_positions.shape[1], 3),
            dtype=np.float32,
        )
        obstacle_positions[:, :, :2] = self.current_obstacle_positions[indices]
        obstacle_positions[:, :, 2] = self.config.obstacle_height / 2
        self._set_mocap_positions(
            rows_wp,
            self._obstacle_mocap_ids_wp,
            obstacle_positions,
        )

    def _set_mocap_positions(self, rows_wp, mocap_ids_wp, positions: np.ndarray) -> None:
        if mocap_ids_wp.shape[0] == 0:
            return
        wp.launch(
            _set_mocap_positions_kernel,
            dim=(rows_wp.shape[0], mocap_ids_wp.shape[0]),
            inputs=[
                self.sim.mjw_data.mocap_pos,
                rows_wp,
                mocap_ids_wp,
                wp.array(positions, dtype=wp.float32, device=self._wp_device),
            ],
        )

    def _forward(self) -> None:
        mjw.forward(self.sim.mjw_model, self.sim.mjw_data)

    def _find_root_addresses(self) -> tuple[int, int]:
        free_joint_ids = [
            joint_id
            for joint_id in range(self.sim.mj_model.njnt)
            if self.sim.mj_model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_FREE
        ]
        if len(free_joint_ids) != 1:
            raise RuntimeError(
                "Expected exactly one free joint for the navigation fly."
            )
        joint_id = free_joint_ids[0]
        return (
            int(self.sim.mj_model.jnt_qposadr[joint_id]),
            int(self.sim.mj_model.jnt_dofadr[joint_id]),
        )

    def _cache_mocap_ids(self, body_names: list[str]) -> np.ndarray:
        mocap_ids = []
        for body_name in body_names:
            body_id = mj.mj_name2id(
                self.sim.mj_model,
                mj.mjtObj.mjOBJ_BODY,
                body_name,
            )
            mocap_id = int(self.sim.mj_model.body_mocapid[body_id])
            if mocap_id < 0:
                raise RuntimeError(f"Expected mocap body for {body_name!r}.")
            mocap_ids.append(mocap_id)
        return np.asarray(mocap_ids, dtype=np.int32)

    def _fly_positions(self) -> np.ndarray:
        body_positions = self.sim.get_body_positions(self.fly.name).numpy()
        return body_positions[:, self._thorax_body_idx, :].astype(np.float32)

    def _fly_headings(self) -> np.ndarray:
        qpos = self.sim.mjw_data.qpos.numpy()
        quats = qpos[:, self._root_qposadr + 3 : self._root_qposadr + 7]
        return np.asarray(
            [_heading_from_quat_wxyz(quat) for quat in quats],
            dtype=np.float32,
        )

    def _policy_action_metadata(
        self, raw_actions: Any
    ) -> list[tuple[int | None, str]]:
        if self.action_space == "continuous":
            return [(None, "low_level") for _ in range(self.num_envs)]
        if raw_actions.__class__.__module__.startswith("torch"):
            action_indices = raw_actions.detach().cpu().numpy()
        else:
            action_indices = np.asarray(raw_actions)
        if action_indices.shape == (self.num_envs, 1):
            action_indices = action_indices[:, 0]
        metadata: list[tuple[int | None, str]] = []
        for action in action_indices:
            metadata.append(
                navigation_policy_action_from_raw(
                    action,
                    action_space=self.action_space,
                )
            )
        return metadata

    def _distance_to_target(
        self, fly_positions: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        targets = self.current_odor_source[indices, 0, :]
        return np.linalg.norm(fly_positions[:, :2] - targets[:, :2], axis=1)

    def _make_observations(
        self, *, indices: np.ndarray, record_diagnostics: bool = False
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        observations = np.zeros(
            (len(indices), *self.observation_shape), dtype=np.float32
        )
        phase_started_at = time.perf_counter()
        odor = self._odor_features(indices)
        odor = _finite_float32(odor)
        if record_diagnostics:
            self._record_diagnostic_time("odor", phase_started_at)
        phase_started_at = time.perf_counter()
        vision = self._vision_features(indices)
        if record_diagnostics:
            self._record_diagnostic_time("vision", phase_started_at)
        phase_started_at = time.perf_counter()
        features = []
        for row in range(len(indices)):
            vision_features = vision[row]
            com_per_eye = _finite_float32(vision_features["com_per_eye"])
            size_per_eye = _finite_float32(vision_features["size_per_eye"])
            observations[row] = np.array(
                [
                    odor[row, 0],
                    odor[row, 1],
                    com_per_eye[0, 0],
                    com_per_eye[0, 1],
                    size_per_eye[0],
                    com_per_eye[1, 0],
                    com_per_eye[1, 1],
                    size_per_eye[1],
                ],
                dtype=np.float32,
            )
            observations[row] = _finite_float32(observations[row])
            features.append(
                {
                    "odor_left": float(odor[row, 0]),
                    "odor_right": float(odor[row, 1]),
                    **vision_features,
                }
            )
        if record_diagnostics:
            self._record_diagnostic_time("observation_assemble", phase_started_at)
        return observations, features

    def _setup_vision(self) -> None:
        if not self.config.enable_vision:
            self._vision_rgb = None
            self._vision_readouts = None
            self._vision_summary = None
            return

        self._retina = Retina()
        self.sim.retina = self._retina
        eye_cameras = list(self.fly.eyecameraname_to_mjcfcamera.values())
        if len(eye_cameras) != 2:
            raise RuntimeError(
                f"Expected two FlyGym eye cameras, got {len(eye_cameras)}."
            )
        self._vision_renderer = self.sim.set_renderer(
            eye_cameras,
            camera_res=(self._retina.nrows, self._retina.ncols),
            buffer_frames=False,
            worlds=list(range(self.num_envs)),
            use_gpu_batch_rendering=True,
            enabled_geom_groups=[0],
        )
        self.sim.mj_model.opt.timestep = self.config.timestep
        self._vision_rgb = wp.zeros(
            (self.num_envs, 2, self._retina.nrows, self._retina.ncols),
            dtype=wp.vec3f,
        )
        self._num_ommatidia_per_eye = int(self._retina.num_ommatidia_per_eye)
        self._vision_readouts = wp.zeros(
            (self.num_envs, 2 * self._num_ommatidia_per_eye * 2),
            dtype=wp.float32,
        )
        self._vision_summary = wp.zeros((self.num_envs, 2 * 3), dtype=wp.float32)

        src_rows, src_cols, ommatidia_ids = _fisheye_ommatidia_pixel_map(self._retina)
        self._vision_src_rows = wp.array(src_rows, dtype=wp.int32)
        self._vision_src_cols = wp.array(src_cols, dtype=wp.int32)
        self._vision_ommatidia_ids = wp.array(ommatidia_ids, dtype=wp.int32)
        self._vision_pale_type_mask = wp.array(
            np.asarray(self._retina.pale_type_mask, dtype=np.int32),
            dtype=wp.int32,
        )
        self._vision_pixels_per_ommatidium = wp.array(
            np.asarray(self._retina.num_pixels_per_ommatidia, dtype=np.float32),
            dtype=wp.float32,
        )
        self._vision_ommatidia_coms = wp.array(
            _compute_ommatidia_coms(self._retina).astype(np.float32),
            dtype=wp.float32,
        )

    def _vision_features(self, indices: np.ndarray) -> list[dict[str, Any]]:
        if not self.config.enable_vision:
            return [_empty_vision_features() for _ in indices]

        assert self._retina is not None
        assert self._vision_renderer is not None
        assert self._vision_rgb is not None
        assert self._vision_readouts is not None
        assert self._vision_summary is not None

        self._vision_readouts.fill_(0.0)
        self._vision_summary.fill_(0.0)
        self._vision_renderer.render_rgb(self.sim.mjw_data, out=self._vision_rgb)
        wp.launch(
            _accumulate_ommatidia_readouts_kernel,
            dim=(self.num_envs, 2, self._vision_src_rows.shape[0]),
            inputs=[
                self._vision_rgb,
                self._vision_src_rows,
                self._vision_src_cols,
                self._vision_ommatidia_ids,
                self._vision_pale_type_mask,
                self._vision_pixels_per_ommatidium,
                self._num_ommatidia_per_eye,
            ],
            outputs=[self._vision_readouts],
        )
        wp.launch(
            _summarize_vision_kernel,
            dim=(self.num_envs, 2, self._num_ommatidia_per_eye),
            inputs=[
                self._vision_readouts,
                self._vision_ommatidia_coms,
                float(self.config.object_threshold),
                self._num_ommatidia_per_eye,
            ],
            outputs=[self._vision_summary],
        )

        summary = self._vision_summary.numpy().reshape(self.num_envs, 2, 3)
        return [
            self._features_from_vision_summary(int(env_idx), summary[int(env_idx)])
            for env_idx in indices
        ]

    def _features_from_vision_summary(
        self, env_idx: int, summary: np.ndarray
    ) -> dict[str, Any]:
        assert self._retina is not None
        summary = _finite_float32(summary)
        counts = summary[:, 2]
        if counts.sum() == 0:
            self._prev_object_size[env_idx] = 0.0
            return _empty_vision_features()

        vision_features = np.zeros((2, 3), dtype=np.float32)
        visible = counts > 0
        vision_features[visible, 0] = summary[visible, 0] / counts[visible]
        vision_features[visible, 1] = summary[visible, 1] / counts[visible]
        vision_features[:, 2] = counts
        vision_features[:, 0] /= self._retina.nrows
        vision_features[:, 1] /= self._retina.ncols
        vision_features[:, 2] /= self._retina.num_ommatidia_per_eye

        size_per_eye = vision_features[:, 2].copy()
        com_per_eye = np.full((2, 2), np.nan, dtype=np.float32)
        visible = size_per_eye > 0
        com_per_eye[visible] = vision_features[visible, :2]
        center_deviation = np.zeros(2, dtype=np.float32)
        if size_per_eye[0] > 0:
            center_deviation[0] = 1.0 - com_per_eye[0, 1]
        if size_per_eye[1] > 0:
            center_deviation[1] = com_per_eye[1, 1]

        object_azimuth = float(
            (
                -size_per_eye[0] * center_deviation[0]
                + size_per_eye[1] * center_deviation[1]
            )
            / size_per_eye.sum()
        )
        object_size = float(np.clip(size_per_eye.sum(), 0.0, 1.0))
        looming_signal = max(0.0, object_size - self._prev_object_size[env_idx])
        self._prev_object_size[env_idx] = object_size
        proximity = float(np.clip(object_size + looming_signal, 0.0, 1.0))
        frontality = float(np.clip(1.0 - abs(object_azimuth), 0.0, 1.0))
        collision_risk = proximity * frontality
        if object_azimuth < 0:
            avoidance_direction = 1.0
        elif object_azimuth > 0:
            avoidance_direction = -1.0
        else:
            avoidance_direction = 1.0 if size_per_eye[0] >= size_per_eye[1] else -1.0

        return {
            "com_per_eye": com_per_eye,
            "size_per_eye": size_per_eye,
            "center_deviation": center_deviation,
            "object_mask_counts": np.rint(
                size_per_eye * self._retina.num_ommatidia_per_eye
            ).astype(np.int32),
            "object_azimuth": object_azimuth,
            "object_size": object_size,
            "object_proximity_proxy": proximity,
            "avoidance_direction": avoidance_direction,
            "collision_risk": collision_risk,
            "looming_signal": looming_signal,
        }

    def _odor_features(self, indices: np.ndarray) -> np.ndarray:
        if not self.config.enable_olfaction:
            return np.zeros((len(indices), 2), dtype=np.float32)
        if len(self._olfaction_site_ids) == 0:
            raise RuntimeError("Fly was constructed without olfaction sensor sites.")
        site_positions = self.sim.mjw_data.site_xpos.numpy()
        odor = np.zeros((len(indices), 2), dtype=np.float32)
        for row, env_idx in enumerate(indices):
            env_idx = int(env_idx)
            sensor_positions = site_positions[env_idx, self._olfaction_site_ids, :]
            odor[row] = _odor_by_side(
                _odor_intensity_from_sources(
                    sensor_positions,
                    self.current_odor_source[env_idx],
                    self.config.peak_odor_intensity,
                ),
                self.config.odor_palps_antennae_weights,
            )
        return odor

    def _make_info(
        self,
        env_idx: int,
        features: dict[str, Any],
        progress_reward: float,
        time_reward: float,
        success_reward: float,
        terminated: bool,
        truncated: bool,
        fly_position: np.ndarray | None = None,
        distance: float | None = None,
        forward_speed: float = 0.0,
        yaw_rate: float = 0.0,
        policy_action_idx: int | None = None,
        policy_action_label: str = "",
        switched_action: bool = False,
        low_level_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if fly_position is None:
            fly_position = self._fly_positions()[env_idx]
        if distance is None:
            distance = float(
                np.linalg.norm(
                    fly_position[:2] - self.current_odor_source[env_idx, 0, :2]
                )
            )
        reward_terms = {
            "progress": float(progress_reward),
            "time": float(time_reward),
            "success": float(success_reward),
            "total": float(progress_reward + time_reward + success_reward),
        }
        info = {
            "sim_time": float(
                self._decision_counts[env_idx] * self.config.decision_interval
            ),
            "decision_count": int(self._decision_counts[env_idx]),
            "fly_position": fly_position.copy(),
            "target_position": self.current_odor_source[env_idx, 0].copy(),
            "obstacle_positions": self.current_obstacle_positions[env_idx].copy(),
            "active_obstacle_count": int(self.current_active_obstacle_counts[env_idx]),
            "active_obstacle_positions": self.current_obstacle_positions[
                env_idx, : self.current_active_obstacle_counts[env_idx]
            ].copy(),
            "distance_to_target": distance,
            "processed_features": features,
            "reward_terms": reward_terms,
            "success": bool(terminated),
            "is_success": bool(terminated),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "forward_speed": float(forward_speed),
            "yaw_rate": float(yaw_rate),
            "policy_action": policy_action_idx,
            "policy_action_label": policy_action_label,
            "switched_action": bool(switched_action),
            "failure_reason": None,
        }
        if low_level_action is not None:
            info["low_level_action"] = low_level_action
        return info


class _GPUOpenLoopTurningController:
    def __init__(
        self,
        num_envs: int,
        timestep: float,
        output_dof_order,
        action_bounds: tuple[float, float],
        device,
    ) -> None:
        self.num_envs = int(num_envs)
        self.timestep = float(timestep)
        self.output_dof_order = list(output_dof_order)
        self.preprogrammed_steps = PreprogrammedSteps()
        network = make_tripod_cpg_network(timestep)
        self._dof_to_output_idx = {
            dof: idx for idx, dof in enumerate(self.output_dof_order)
        }
        self._phases_host = np.zeros((self.num_envs, len(LEGS)), dtype=np.float32)
        self._magnitudes_host = np.zeros((self.num_envs, len(LEGS)), dtype=np.float32)
        self.phases = wp.zeros(
            (self.num_envs, len(LEGS)), dtype=wp.float32, device=device
        )
        self.magnitudes = wp.zeros(
            (self.num_envs, len(LEGS)), dtype=wp.float32, device=device
        )
        self._next_phases = wp.zeros(
            (self.num_envs, len(LEGS)), dtype=wp.float32, device=device
        )
        self._next_magnitudes = wp.zeros(
            (self.num_envs, len(LEGS)), dtype=wp.float32, device=device
        )
        self.joint_angles = wp.zeros(
            (self.num_envs, len(self.output_dof_order)),
            dtype=wp.float32,
            device=device,
        )
        self.adhesion = wp.zeros(
            (self.num_envs, len(LEGS)), dtype=wp.float32, device=device
        )
        self._base_intrinsic_freqs = wp.array(
            network.intrinsic_freqs.astype(np.float32),
            dtype=wp.float32,
            device=device,
        )
        self._coupling_weights = wp.array(
            network.coupling_weights.astype(np.float32),
            dtype=wp.float32,
            device=device,
        )
        self._phase_biases = wp.array(
            network.phase_biases.astype(np.float32), dtype=wp.float32, device=device
        )
        self._convergence_coefs = wp.array(
            network.convergence_coefs.astype(np.float32),
            dtype=wp.float32,
            device=device,
        )
        self._spline_coeffs, self._neutral_angles, self._phase_grid_step = (
            self._make_step_interpolation_tables()
        )
        self._n_intervals = self._spline_coeffs.shape[3]
        self._spline_coeffs = wp.array(
            self._spline_coeffs, dtype=wp.float32, device=device
        )
        self._neutral_angles = wp.array(
            self._neutral_angles, dtype=wp.float32, device=device
        )
        self._dof_to_output_idx_wp = wp.array(
            self._make_dof_to_output_idx(), dtype=wp.int32, device=device
        )
        self._swing_periods = wp.array(
            self._make_swing_periods(), dtype=wp.float32, device=device
        )
        self._action_low = float(action_bounds[0])
        self._action_high = float(action_bounds[1])
        self.reset_at(np.arange(self.num_envs, dtype=np.int32), [None] * self.num_envs)

    def reset_at(self, indices: np.ndarray, seeds: list[int | None]) -> None:
        for env_idx, seed in zip(indices, seeds, strict=True):
            rng = np.random.default_rng(seed)
            self._phases_host[int(env_idx)] = (
                rng.random(len(LEGS), dtype=np.float32) * np.float32(_TWO_PI)
            )
            self._magnitudes_host[int(env_idx)] = 0.0
        self.phases.assign(self._phases_host)
        self.magnitudes.assign(self._magnitudes_host)

    def step(self, descending_signals) -> tuple[wp.array, wp.array]:
        wp.launch(
            _cpg_update_kernel,
            dim=(self.num_envs, len(LEGS)),
            inputs=[
                descending_signals,
                self.phases,
                self.magnitudes,
                self._base_intrinsic_freqs,
                self._coupling_weights,
                self._phase_biases,
                self._convergence_coefs,
                self._next_phases,
                self._next_magnitudes,
                self.timestep,
                self._action_low,
                self._action_high,
            ],
        )
        wp.launch(
            _cpg_output_kernel,
            dim=(self.num_envs, len(LEGS)),
            inputs=[
                self.phases,
                self.magnitudes,
                self._next_phases,
                self._next_magnitudes,
                self._spline_coeffs,
                self._neutral_angles,
                self._dof_to_output_idx_wp,
                self._swing_periods,
                self.joint_angles,
                self.adhesion,
                self._phase_grid_step,
                self._n_intervals,
            ],
        )
        return self.joint_angles, self.adhesion

    def _make_step_interpolation_tables(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        n_intervals = self.preprogrammed_steps._length - 1
        coeffs = np.empty(
            (len(LEGS), _N_STEP_DOFS, 4, n_intervals), dtype=np.float32
        )
        neutral_angles = np.empty((len(LEGS), _N_STEP_DOFS), dtype=np.float32)
        for leg_idx, leg in enumerate(LEGS):
            coeffs[leg_idx] = np.transpose(
                self.preprogrammed_steps._psi_funcs[leg].c, (2, 0, 1)
            ).astype(np.float32)
            neutral_angles[leg_idx] = self.preprogrammed_steps.neutral_pos[leg][
                :, 0
            ].astype(np.float32)
        return coeffs, neutral_angles, float(_TWO_PI / n_intervals)

    def _make_dof_to_output_idx(self) -> np.ndarray:
        dof_to_output_idx = np.full((len(LEGS), _N_STEP_DOFS), -1, dtype=np.int32)
        for leg_idx, leg in enumerate(LEGS):
            for dof_idx, dof_spec in enumerate(self.preprogrammed_steps.dofs_per_leg):
                output_idx = self._dof_to_output_idx.get(
                    dof_spec_to_jointdof(leg, dof_spec)
                )
                if output_idx is not None:
                    dof_to_output_idx[leg_idx, dof_idx] = output_idx
        return dof_to_output_idx

    def _make_swing_periods(self) -> np.ndarray:
        return np.asarray(
            [self.preprogrammed_steps.swing_period[leg] for leg in LEGS],
            dtype=np.float32,
        )


def _warp_device_names() -> list[str]:
    return [str(device) for device in wp.get_devices()]


def _warp_cuda_device_names() -> list[str]:
    return [str(device) for device in wp.get_devices() if device.is_cuda]


def _cpu_affinity() -> list[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    return sorted(os.sched_getaffinity(0))


def _cpu_affinity_count() -> int | None:
    affinity = _cpu_affinity()
    return None if affinity is None else len(affinity)


def _process_cpu_seconds() -> float:
    process_times = os.times()
    return float(process_times.user + process_times.system)


def _fisheye_ommatidia_pixel_map(
    retina: Retina,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src_rows = []
    src_cols = []
    ommatidia_ids = []
    nrows = int(retina.nrows)
    ncols = int(retina.ncols)
    ommatidia_id_map = np.asarray(retina.ommatidia_id_map)
    for dst_row in range(nrows):
        for dst_col in range(ncols):
            ommatidium_idx = int(ommatidia_id_map[dst_row, dst_col]) - 1
            if ommatidium_idx < 0:
                continue
            dst_row_norm = ((2 * dst_row - nrows) / nrows) / retina.zoom
            dst_col_norm = ((2 * dst_col - ncols) / ncols) / retina.zoom
            dst_radius_norm_sq = dst_col_norm**2 + dst_row_norm**2
            denom = 1 - (retina.distortion_coefficient * dst_radius_norm_sq) + 1e-6
            src_row_norm = dst_row_norm / denom
            src_col_norm = dst_col_norm / denom
            src_row = int(((src_row_norm + 1) * nrows) / 2)
            src_col = int(((src_col_norm + 1) * ncols) / 2)
            if 0 <= src_row < nrows and 0 <= src_col < ncols:
                src_rows.append(src_row)
                src_cols.append(src_col)
                ommatidia_ids.append(ommatidium_idx)
    return (
        np.asarray(src_rows, dtype=np.int32),
        np.asarray(src_cols, dtype=np.int32),
        np.asarray(ommatidia_ids, dtype=np.int32),
    )
