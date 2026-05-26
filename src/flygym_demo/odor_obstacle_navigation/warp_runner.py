from __future__ import annotations

from copy import deepcopy
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
    _odor_by_side,
    _project_action_norm,
    _yaw_to_quat_wxyz,
)
from flygym_demo.odor_obstacle_navigation.world import ObstacleOdorWorld


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


class WarpNavigationTaskRunner:
    """Batched MuJoCo-Warp runner for the odor obstacle navigation PPO loop.

    The runner exposes the same numpy-facing API as the CPU sync/subproc runners while
    keeping physics in one `GPUSimulation` with `num_envs` parallel worlds.
    """

    def __init__(self, config: NavigationTaskConfig, num_envs: int) -> None:
        self.config = deepcopy(config)
        if self.config.randomize_layout:
            raise NotImplementedError(
                "WarpNavigationTaskRunner currently requires a fixed layout."
            )
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
        self.sim = GPUSimulation(self.world, n_worlds=self.num_envs)
        self.sim.mj_model.opt.timestep = self.config.timestep
        self.sim.mjw_model, self.sim.mjw_data = self.sim._mj_structs_to_mjw_structs()
        self._vision_renderer = None
        self._retina = None
        self._setup_vision()

        self._root_qposadr, self._root_qveladr = self._find_root_addresses()
        self._thorax_body_idx = self.fly.get_bodysegs_order().index(
            BodySegment("c_thorax")
        )
        self._olfaction_site_ids = self.sim._internal_olfaction_siteids_by_fly.get(
            self.fly.name, np.array([], dtype=np.int32)
        )
        self.controller = _BatchedOpenLoopTurningController(
            num_envs=self.num_envs,
            timestep=self.config.timestep,
            output_dof_order=self.fly.get_actuated_jointdofs_order(
                ActuatorType.POSITION
            ),
        )
        self._neutral_qpos = np.asarray(self.sim.mj_data.qpos, dtype=np.float32).copy()
        self._neutral_qvel = np.asarray(self.sim.mj_data.qvel, dtype=np.float32).copy()
        self._neutral_ctrl = np.asarray(self.sim.mj_data.ctrl, dtype=np.float32).copy()
        self._rngs = [np.random.default_rng() for _ in range(self.num_envs)]
        self._decision_counts = np.zeros(self.num_envs, dtype=np.int32)
        self._prev_distance_to_target = np.zeros(self.num_envs, dtype=np.float32)
        self._prev_object_size = np.zeros(self.num_envs, dtype=np.float32)
        self.current_odor_source = np.repeat(
            self.config.odor_source[np.newaxis, :, :], self.num_envs, axis=0
        )
        self.current_obstacle_positions = np.repeat(
            self.config.obstacle_positions[np.newaxis, :, :], self.num_envs, axis=0
        )

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
            if seed is not None:
                self._rngs[int(env_idx)] = np.random.default_rng(seed)
            spawn_positions[row] = self.config.spawn_position
            spawn_quats[row] = _yaw_to_quat_wxyz(self.config.spawn_yaw)

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
        self._forward()

        self.controller.reset_at(indices, seeds)
        self._decision_counts[indices] = 0
        self._prev_object_size[indices] = 0.0
        fly_positions = self._fly_positions()[indices]
        self._prev_distance_to_target[indices] = self._distance_to_target(
            fly_positions, indices
        )
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
            )
            for row, env_idx in enumerate(indices)
        ]
        return observations, infos

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        actions = self._resolve_actions(actions)

        for _ in range(self.config.decision_steps):
            joint_angles, adhesion = self.controller.step(actions)
            self.sim.set_actuator_inputs(
                self.fly.name, ActuatorType.POSITION, joint_angles
            )
            self.sim.set_leg_adhesion_states(self.fly.name, adhesion)
            self.sim.step()

        indices = np.arange(self.num_envs, dtype=np.int32)
        self._decision_counts += 1
        fly_positions = self._fly_positions()
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
        observations, features = self._make_observations(indices=indices)
        infos = [
            self._make_info(
                int(env_idx),
                features[row],
                float(self.config.progress_reward_weight * progress[env_idx]),
                float(-self.config.time_penalty),
                float(self.config.success_bonus if terminated[env_idx] else 0.0),
                bool(terminated[env_idx]),
                bool(truncated[env_idx]),
                fly_position=fly_positions[env_idx],
                distance=float(distances[env_idx]),
            )
            for row, env_idx in enumerate(indices)
        ]
        return observations, rewards, terminated, truncated, infos

    def close(self) -> None:
        self.sim.close()

    def diagnostics(self) -> list[dict[str, Any]]:
        return []

    def _make_world(self) -> ObstacleOdorWorld:
        return ObstacleOdorWorld(
            odor_source=self.config.odor_source,
            peak_odor_intensity=self.config.peak_odor_intensity,
            obstacle_positions=self.config.obstacle_positions,
            obstacle_radius=self.config.obstacle_radius,
            obstacle_height=self.config.obstacle_height,
            obstacle_colors=self.config.obstacle_colors,
        )

    def _resolve_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.action_space == "continuous":
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

    def _fly_positions(self) -> np.ndarray:
        body_positions = self.sim.get_body_positions(self.fly.name).numpy()
        return body_positions[:, self._thorax_body_idx, :].astype(np.float32)

    def _distance_to_target(
        self, fly_positions: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        targets = self.current_odor_source[indices, 0, :]
        return np.linalg.norm(fly_positions[:, :2] - targets[:, :2], axis=1)

    def _make_observations(
        self, *, indices: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        observations = np.zeros(
            (len(indices), *self.observation_shape), dtype=np.float32
        )
        odor = self._odor_features(indices)
        vision = self._vision_features(indices)
        features = []
        for row in range(len(indices)):
            vision_features = vision[row]
            com_per_eye = np.nan_to_num(vision_features["com_per_eye"], nan=0.0)
            size_per_eye = vision_features["size_per_eye"]
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
            features.append(
                {
                    "odor_left": float(odor[row, 0]),
                    "odor_right": float(odor[row, 1]),
                    **vision_features,
                }
            )
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
            sensor_positions = site_positions[int(env_idx), self._olfaction_site_ids, :]
            odor[row] = _odor_by_side(
                self.world.get_olfaction(sensor_positions),
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
        return {
            "sim_time": float(
                self._decision_counts[env_idx] * self.config.decision_interval
            ),
            "decision_count": int(self._decision_counts[env_idx]),
            "fly_position": fly_position.copy(),
            "target_position": self.current_odor_source[env_idx, 0].copy(),
            "obstacle_positions": self.current_obstacle_positions[env_idx].copy(),
            "distance_to_target": distance,
            "processed_features": features,
            "reward_terms": reward_terms,
            "success": bool(terminated),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }


class _BatchedOpenLoopTurningController:
    def __init__(self, num_envs: int, timestep: float, output_dof_order) -> None:
        self.num_envs = int(num_envs)
        self.timestep = float(timestep)
        self.output_dof_order = list(output_dof_order)
        self.preprogrammed_steps = PreprogrammedSteps()
        network = make_tripod_cpg_network(timestep)
        self._base_intrinsic_freqs = network.intrinsic_freqs.astype(np.float32)
        self._coupling_weights = network.coupling_weights.astype(np.float32)
        self._phase_biases = network.phase_biases.astype(np.float32)
        self._convergence_coefs = network.convergence_coefs.astype(np.float32)
        self.phases = np.zeros((self.num_envs, len(LEGS)), dtype=np.float32)
        self.magnitudes = np.zeros((self.num_envs, len(LEGS)), dtype=np.float32)
        self._dof_to_output_idx = {
            dof: idx for idx, dof in enumerate(self.output_dof_order)
        }
        self.reset_at(np.arange(self.num_envs, dtype=np.int32), [None] * self.num_envs)

    def reset_at(self, indices: np.ndarray, seeds: list[int | None]) -> None:
        for env_idx, seed in zip(indices, seeds, strict=True):
            rng = np.random.default_rng(seed)
            self.phases[int(env_idx)] = rng.random(len(LEGS)) * 2 * np.pi
            self.magnitudes[int(env_idx)] = 0.0

    def step(self, descending_signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        intrinsic_amps = np.repeat(
            np.abs(descending_signals[:, :, np.newaxis]), 3, axis=2
        ).reshape(self.num_envs, len(LEGS))
        intrinsic_freqs = np.repeat(
            self._base_intrinsic_freqs[np.newaxis, :], self.num_envs, axis=0
        )
        intrinsic_freqs[:, :3] *= np.where(descending_signals[:, 0:1] >= 0, 1.0, -1.0)
        intrinsic_freqs[:, 3:] *= np.where(descending_signals[:, 1:2] >= 0, 1.0, -1.0)

        phase_diff = self.phases[:, np.newaxis, :] - self.phases[:, :, np.newaxis]
        coupling = (
            self.magnitudes[:, np.newaxis, :]
            * self._coupling_weights[np.newaxis, :, :]
            * np.sin(phase_diff - self._phase_biases[np.newaxis, :, :])
        ).sum(axis=2)
        dtheta_dt = 2 * np.pi * intrinsic_freqs + coupling
        dr_dt = self._convergence_coefs * (intrinsic_amps - self.magnitudes)
        self.phases += (dtheta_dt * self.timestep).astype(np.float32)
        self.phases = np.mod(self.phases, 2 * np.pi).astype(np.float32)
        self.magnitudes += (dr_dt * self.timestep).astype(np.float32)
        return self._joint_angles(), self._adhesion()

    def _joint_angles(self) -> np.ndarray:
        joint_angles = np.zeros(
            (self.num_envs, len(self.output_dof_order)), dtype=np.float32
        )
        for leg_idx, leg in enumerate(LEGS):
            leg_angles = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.phases[:, leg_idx],
                self.magnitudes[:, leg_idx],
            )
            for dof_idx, dof_spec in enumerate(self.preprogrammed_steps.dofs_per_leg):
                output_idx = self._dof_to_output_idx.get(
                    dof_spec_to_jointdof(leg, dof_spec)
                )
                if output_idx is not None:
                    joint_angles[:, output_idx] = leg_angles[dof_idx]
        return joint_angles

    def _adhesion(self) -> np.ndarray:
        adhesion = np.ones((self.num_envs, len(LEGS)), dtype=np.float32)
        phase_mod = np.mod(self.phases, 2 * np.pi)
        for leg_idx, leg in enumerate(LEGS):
            swing_start, swing_end = self.preprogrammed_steps.swing_period[leg]
            adhesion[:, leg_idx] = ~(
                (phase_mod[:, leg_idx] > swing_start)
                & (phase_mod[:, leg_idx] < swing_end)
            )
        return adhesion


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
