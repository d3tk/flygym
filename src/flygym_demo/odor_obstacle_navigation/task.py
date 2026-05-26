from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np

from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import ActuatorType
from flygym.utils.math import Rotation3D
from flygym.vision.retina import Retina
from flygym_demo.complex_terrain import (
    HybridTurningController,
    apply_locomotion_action,
    make_locomotion_fly,
)
from flygym_demo.odor_obstacle_navigation.world import ObstacleOdorWorld


ACTION_SPACES = ("continuous", "discrete")
ACTION_PRESETS = {
    "forward": np.array([1.0, 1.0], dtype=np.float32),
    "left": np.array([0.8, 1.2], dtype=np.float32),
    "right": np.array([1.2, 0.8], dtype=np.float32),
}
ACTION_PRESET_NAMES = tuple(ACTION_PRESETS)
ACTION_PRESET_VALUES = np.stack(tuple(ACTION_PRESETS.values())).astype(np.float32)


@dataclass(slots=True)
class NavigationTaskConfig:
    timestep: float = 1e-4
    decision_steps: int = 500
    episode_time: float = 5.0
    target_radius: float = 2.0
    progress_reward_weight: float = 1.0
    time_penalty: float = 0.01
    success_bonus: float = 10.0
    action_space: str = "continuous"
    action_bounds: tuple[float, float] = (1e-6, 1.8)
    action_max_norm: float | None = 2.0
    spawn_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.8], dtype=float)
    )
    spawn_yaw: float = 0.0
    odor_source: np.ndarray = field(
        default_factory=lambda: np.array([[25.0, 0.0, 2.0]], dtype=float)
    )
    peak_odor_intensity: np.ndarray = field(
        default_factory=lambda: np.array([[1.0]], dtype=float)
    )
    obstacle_positions: np.ndarray = field(
        default_factory=lambda: np.array(
            [(7.5, 0.0), (12.5, 5.0), (17.5, -5.0)],
            dtype=float,
        )
    )
    obstacle_radius: float = 1.0
    obstacle_height: float = 4.0
    obstacle_colors: np.ndarray = field(
        default_factory=lambda: np.array((0.0, 0.0, 0.0, 1.0), dtype=float)
    )
    odor_palps_antennae_weights: tuple[float, float] = (1.0, 9.0)
    object_threshold: float = 0.15
    randomize_layout: bool = False
    spawn_xy_jitter: float = 1.0
    spawn_yaw_jitter: float = math.pi / 4
    odor_source_x_range: tuple[float, float] = (20.0, 30.0)
    odor_source_y_range: tuple[float, float] = (-6.0, 6.0)
    obstacle_x_range: tuple[float, float] = (6.0, 20.0)
    obstacle_y_range: tuple[float, float] = (-8.0, 8.0)
    enable_vision: bool = True
    enable_olfaction: bool = True
    render: bool = False
    camera_res: tuple[int, int] = (240, 320)
    playback_speed: float = 0.2
    output_fps: int = 25
    warmup_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.spawn_position = np.asarray(self.spawn_position, dtype=float)
        self.odor_source = np.asarray(self.odor_source, dtype=float)
        self.peak_odor_intensity = np.asarray(self.peak_odor_intensity, dtype=float)
        self.obstacle_positions = np.asarray(self.obstacle_positions, dtype=float)
        self.obstacle_colors = np.asarray(self.obstacle_colors, dtype=float)
        if self.timestep <= 0:
            raise ValueError("timestep must be positive.")
        if self.decision_steps <= 0:
            raise ValueError("decision_steps must be positive.")
        if self.episode_time <= 0:
            raise ValueError("episode_time must be positive.")
        if self.target_radius <= 0:
            raise ValueError("target_radius must be positive.")
        if self.time_penalty < 0:
            raise ValueError("time_penalty must be non-negative.")
        if self.action_space not in ACTION_SPACES:
            raise ValueError(
                f"action_space must be one of {ACTION_SPACES}; got {self.action_space!r}."
            )
        if self.action_bounds[0] >= self.action_bounds[1]:
            raise ValueError("action_bounds must be ordered as (low, high).")
        if self.action_max_norm is not None and self.action_max_norm <= 0:
            raise ValueError("action_max_norm must be positive or None.")
        if self.spawn_position.shape != (3,):
            raise ValueError("spawn_position must have shape (3,).")
        if self.odor_source.shape != (1, 3):
            raise ValueError("odor_source must have shape (1, 3).")
        if self.peak_odor_intensity.ndim != 2 or self.peak_odor_intensity.shape[0] != 1:
            raise ValueError("peak_odor_intensity must have shape (1, odor_dims).")
        if self.obstacle_positions.ndim != 2 or self.obstacle_positions.shape[1] != 2:
            raise ValueError("obstacle_positions must have shape (n_obstacles, 2).")
        if self.warmup_seconds < 0:
            raise ValueError("warmup_seconds must be non-negative.")

    @property
    def decision_interval(self) -> float:
        return self.decision_steps * self.timestep

    @property
    def max_decision_steps(self) -> int:
        return math.ceil(self.episode_time / self.decision_interval)


@dataclass(slots=True)
class NavigationTaskFactory:
    config: NavigationTaskConfig

    def __call__(self) -> "OdorObstacleNavigationTask":
        return OdorObstacleNavigationTask(deepcopy(self.config))


class OdorObstacleNavigationTask:
    observation_shape = (8,)

    def __init__(self, config: NavigationTaskConfig | None = None) -> None:
        self.config = config or NavigationTaskConfig()
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
        self._base_spawn_position = self.config.spawn_position.copy()
        self._base_spawn_yaw = float(self.config.spawn_yaw)
        self._base_odor_source = self.config.odor_source.copy()
        self._base_obstacle_positions = self.config.obstacle_positions.copy()
        self.current_odor_source = self._base_odor_source.copy()
        self.current_obstacle_positions = self._base_obstacle_positions.copy()
        self._rng = np.random.default_rng()
        self._decision_count = 0
        self._prev_distance_to_target: float | None = None
        self._prev_object_size = 0.0

        self.fly = self._make_fly()
        self.world = self._make_world()
        self.world.add_fly(
            self.fly,
            self.config.spawn_position,
            Rotation3D("quat", _yaw_to_quat_wxyz(self.config.spawn_yaw)),
            bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
            add_ground_contact_sensors=False,
        )
        self.camera = self._add_camera() if self.config.render else None
        self.sim = Simulation(self.world)
        self.sim.mj_model.opt.timestep = self.config.timestep
        if self.camera is not None:
            self.renderer = self.sim.set_renderer(
                [self.camera],
                camera_res=self.config.camera_res,
                playback_speed=self.config.playback_speed,
                output_fps=self.config.output_fps,
            )
        else:
            self.renderer = None
        self.controller = HybridTurningController(
            timestep=self.config.timestep,
            output_dof_order=self.fly.get_actuated_jointdofs_order(
                ActuatorType.POSITION
            ),
        )
        self._root_qposadr, self._root_qveladr = self._find_root_addresses()
        self._odor_marker_body_ids = self._cache_body_ids(
            self.world.odor_marker_body_full_identifiers()
        )
        self._obstacle_body_ids = self._cache_body_ids(
            self.world.obstacle_body_full_identifiers()
        )
        self._retina = Retina() if self.config.enable_vision else None
        self.sim.retina = self._retina
        self._ommatidia_coms = (
            _compute_ommatidia_coms(self._retina) if self._retina is not None else None
        )

    @property
    def target_position(self) -> np.ndarray:
        return self.current_odor_source[0].copy()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        spawn_position, spawn_yaw, odor_source, obstacle_positions = (
            self._sample_layout()
        )
        self.sim.reset()
        self.sim.mj_model.opt.timestep = self.config.timestep
        self.controller.reset(seed=seed)
        self.set_layout(
            odor_source=odor_source,
            obstacle_positions=obstacle_positions,
            forward=False,
        )
        self._set_root_state(spawn_position, spawn_yaw)
        if self.config.warmup_seconds > 0:
            self.sim.warmup(self.config.warmup_seconds)
        mj.mj_forward(self.sim.mj_model, self.sim.mj_data)

        self._decision_count = 0
        self._prev_object_size = 0.0
        fly_position = self._fly_position()
        self._prev_distance_to_target = self._distance_to_target(fly_position)
        observation, features = self._make_observation()
        info = self._make_info(
            features,
            reward_terms={"progress": 0.0, "success": 0.0, "total": 0.0},
            terminated=False,
            truncated=False,
        )
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        descending_signal = resolve_navigation_action(
            action,
            action_space=self.action_space,
            action_low=self.action_low,
            action_high=self.action_high,
            action_max_norm=self.action_max_norm,
        )

        for _ in range(self.config.decision_steps):
            locomotion_action = self.controller.step(
                descending_signal, self.sim, self.fly.name
            )
            apply_locomotion_action(self.sim, self.fly.name, locomotion_action)
            self.sim.step()

        self._decision_count += 1
        fly_position = self._fly_position()
        distance = self._distance_to_target(fly_position)
        success = distance <= self.config.target_radius
        reward, reward_terms = self._compute_reward(distance, success)
        self._prev_distance_to_target = distance
        terminated = bool(success)
        truncated = bool(
            not terminated and self._decision_count >= self.config.max_decision_steps
        )
        observation, features = self._make_observation()
        info = self._make_info(
            features,
            reward_terms=reward_terms,
            terminated=terminated,
            truncated=truncated,
        )
        return observation, reward, terminated, truncated, info

    def set_layout(
        self,
        *,
        odor_source: np.ndarray | None = None,
        obstacle_positions: np.ndarray | None = None,
        forward: bool = True,
    ) -> None:
        if odor_source is not None:
            odor_source = np.asarray(odor_source, dtype=float)
            if odor_source.shape != self._base_odor_source.shape:
                raise ValueError(
                    f"odor_source must have shape {self._base_odor_source.shape}."
                )
            self.current_odor_source = odor_source.copy()
            self.world.odor_source = odor_source.copy()
            for body_id, position in zip(self._odor_marker_body_ids, odor_source):
                self._set_mocap_body_position(body_id, position)

        if obstacle_positions is not None:
            obstacle_positions = np.asarray(obstacle_positions, dtype=float)
            if obstacle_positions.shape != self._base_obstacle_positions.shape:
                raise ValueError(
                    "obstacle_positions must have shape "
                    f"{self._base_obstacle_positions.shape}."
                )
            self.current_obstacle_positions = obstacle_positions.copy()
            z_pos = self.config.obstacle_height / 2
            for body_id, xy in zip(self._obstacle_body_ids, obstacle_positions):
                self._set_mocap_body_position(
                    body_id,
                    np.array([xy[0], xy[1], z_pos], dtype=float),
                )
        if forward:
            mj.mj_forward(self.sim.mj_model, self.sim.mj_data)

    def render(self) -> list[np.ndarray | None]:
        if self.renderer is None:
            return [None]
        if not self.sim.render_as_needed():
            return [None for _ in self.renderer.frames]
        return [frames[-1] for frames in self.renderer.frames.values()]

    def save_video(self, path: str | Path) -> None:
        if self.renderer is None:
            raise RuntimeError("render=True is required to save video.")
        self.renderer.save_video(path)

    def close(self) -> None:
        self.sim.close()

    def _make_fly(self):
        fly = make_locomotion_fly()
        if self.config.enable_vision:
            fly.add_vision()
        if self.config.enable_olfaction:
            fly.add_olfaction()
        return fly

    def _make_world(self) -> ObstacleOdorWorld:
        return ObstacleOdorWorld(
            odor_source=self.config.odor_source,
            peak_odor_intensity=self.config.peak_odor_intensity,
            obstacle_positions=self.config.obstacle_positions,
            obstacle_radius=self.config.obstacle_radius,
            obstacle_height=self.config.obstacle_height,
            obstacle_colors=self.config.obstacle_colors,
        )

    def _add_camera(self):
        x_extent = max(float(self.config.odor_source[:, 0].max()), 10.0)
        if self.config.obstacle_positions.size:
            x_extent = max(x_extent, float(self.config.obstacle_positions[:, 0].max()))
        camera_height = max(20.0, x_extent + 8.0)
        return self.world.mjcf_root.worldbody.add(
            "camera",
            name="navigation_birdeye",
            mode="fixed",
            pos=(max(8.0, x_extent / 2), 0.0, camera_height),
            euler=(0.0, 0.0, 0.0),
            fovy=45.0,
        )

    def _sample_layout(self) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        spawn_position = self._base_spawn_position.copy()
        spawn_yaw = self._base_spawn_yaw
        odor_source = self._base_odor_source.copy()
        obstacle_positions = self._base_obstacle_positions.copy()

        if not self.config.randomize_layout:
            return spawn_position, spawn_yaw, odor_source, obstacle_positions

        spawn_position[:2] += self._rng.uniform(
            -self.config.spawn_xy_jitter,
            self.config.spawn_xy_jitter,
            size=2,
        )
        spawn_yaw += float(
            self._rng.uniform(
                -self.config.spawn_yaw_jitter,
                self.config.spawn_yaw_jitter,
            )
        )
        odor_source[0, 0] = self._rng.uniform(*self.config.odor_source_x_range)
        odor_source[0, 1] = self._rng.uniform(*self.config.odor_source_y_range)
        if obstacle_positions.size:
            obstacle_positions[:, 0] = self._rng.uniform(
                *self.config.obstacle_x_range,
                size=obstacle_positions.shape[0],
            )
            obstacle_positions[:, 1] = self._rng.uniform(
                *self.config.obstacle_y_range,
                size=obstacle_positions.shape[0],
            )
        return spawn_position, spawn_yaw, odor_source, obstacle_positions

    def _make_observation(self) -> tuple[np.ndarray, dict[str, Any]]:
        odor_features = (
            _odor_by_side(
                self.sim.get_odor_intensity(self.fly.name),
                self.config.odor_palps_antennae_weights,
            )
            if self.config.enable_olfaction
            else np.zeros(2, dtype=np.float32)
        )
        vision_features = (
            self._process_vision(self.sim.get_ommatidia_readouts(self.fly.name))
            if self.config.enable_vision
            else _empty_vision_features()
        )
        com_per_eye = np.nan_to_num(vision_features["com_per_eye"], nan=0.0)
        size_per_eye = vision_features["size_per_eye"]
        observation = np.array(
            [
                odor_features[0],
                odor_features[1],
                com_per_eye[0, 0],
                com_per_eye[0, 1],
                size_per_eye[0],
                com_per_eye[1, 0],
                com_per_eye[1, 1],
                size_per_eye[1],
            ],
            dtype=np.float32,
        )
        features = {
            "odor_left": float(odor_features[0]),
            "odor_right": float(odor_features[1]),
            **vision_features,
        }
        return observation, features

    def _process_vision(self, vision_input: np.ndarray) -> dict[str, Any]:
        if self._retina is None or self._ommatidia_coms is None:
            return _empty_vision_features()

        vision_features = np.zeros((2, 3), dtype=np.float32)
        for eye_idx, ommatidia_readings in enumerate(vision_input):
            object_mask = ommatidia_readings.max(axis=1) < self.config.object_threshold
            object_coords = self._ommatidia_coms[object_mask]
            if object_coords.shape[0] > 0:
                vision_features[eye_idx, :2] = object_coords.mean(axis=0)
            vision_features[eye_idx, 2] = object_coords.shape[0]

        if vision_features[:, 2].sum() == 0:
            self._prev_object_size = 0.0
            return _empty_vision_features()

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
        looming_signal = max(0.0, object_size - self._prev_object_size)
        self._prev_object_size = object_size
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

    def _make_info(
        self,
        features: dict[str, Any],
        *,
        reward_terms: dict[str, float],
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        fly_position = self._fly_position()
        return {
            "sim_time": float(self.sim.time),
            "decision_count": self._decision_count,
            "fly_position": fly_position.copy(),
            "target_position": self.target_position,
            "obstacle_positions": self.current_obstacle_positions.copy(),
            "distance_to_target": self._distance_to_target(fly_position),
            "processed_features": features,
            "reward_terms": reward_terms,
            "success": terminated,
            "terminated": terminated,
            "truncated": truncated,
        }

    def _compute_reward(
        self, distance_to_target: float, success: bool
    ) -> tuple[float, dict[str, float]]:
        previous_distance = (
            distance_to_target
            if self._prev_distance_to_target is None
            else self._prev_distance_to_target
        )
        progress = previous_distance - distance_to_target
        terms = {
            "progress": self.config.progress_reward_weight * progress,
            "time": -self.config.time_penalty,
            "success": self.config.success_bonus if success else 0.0,
        }
        reward = float(sum(terms.values()))
        terms["total"] = reward
        return reward, terms

    def _distance_to_target(self, fly_position: np.ndarray) -> float:
        return float(np.linalg.norm(fly_position[:2] - self.target_position[:2]))

    def _fly_position(self) -> np.ndarray:
        body_order = self.fly.get_bodysegs_order()
        thorax_idx = body_order.index(BodySegment("c_thorax"))
        return self.sim.get_body_positions(self.fly.name)[thorax_idx].astype(np.float32)

    def _set_root_state(self, position: np.ndarray, yaw: float) -> None:
        self.sim.mj_data.qpos[self._root_qposadr : self._root_qposadr + 3] = position
        self.sim.mj_data.qpos[self._root_qposadr + 3 : self._root_qposadr + 7] = (
            _yaw_to_quat_wxyz(yaw)
        )
        self.sim.mj_data.qvel[self._root_qveladr : self._root_qveladr + 6] = 0.0
        mj.mj_forward(self.sim.mj_model, self.sim.mj_data)

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

    def _cache_body_ids(self, body_names: list[str]) -> list[int]:
        return [
            mj.mj_name2id(self.sim.mj_model, mj.mjtObj.mjOBJ_BODY, name)
            for name in body_names
        ]

    def _set_mocap_body_position(self, body_id: int, position: np.ndarray) -> None:
        mocap_id = int(self.sim.mj_model.body_mocapid[body_id])
        if mocap_id < 0:
            raise RuntimeError("Expected a mocap body.")
        self.sim.mj_data.mocap_pos[mocap_id] = np.asarray(position, dtype=float)


def _yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = yaw / 2
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def resolve_navigation_action(
    action: Any,
    *,
    action_space: str,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    action_max_norm: float | None = None,
) -> np.ndarray:
    if action_space == "continuous":
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (2,):
            raise ValueError("continuous action must have shape (2,).")
        if action_low is not None and action_high is not None:
            action_array = np.clip(action_array, action_low, action_high)
        return _project_action_norm(action_array, action_max_norm)

    if action_space != "discrete":
        raise ValueError(
            f"action_space must be one of {ACTION_SPACES}; got {action_space!r}."
        )

    action_array = np.asarray(action)
    if action_array.shape == (1,):
        action_value = action_array[0]
    elif action_array.shape == ():
        action_value = action_array.item()
    else:
        raise ValueError("discrete action must be a scalar integer index.")
    if (
        isinstance(action_value, (float, np.floating))
        and not float(action_value).is_integer()
    ):
        raise ValueError("discrete action must be an integer index.")
    action_idx = int(action_value)
    if action_idx < 0 or action_idx >= len(ACTION_PRESET_VALUES):
        raise ValueError(
            f"discrete action index must be in [0, {len(ACTION_PRESET_VALUES) - 1}]."
        )
    return ACTION_PRESET_VALUES[action_idx].copy()


def _project_action_norm(
    action: np.ndarray, action_max_norm: float | None
) -> np.ndarray:
    if action_max_norm is None:
        return action
    norm = float(np.linalg.norm(action))
    if norm <= action_max_norm or norm == 0.0:
        return action
    return (action * (action_max_norm / norm)).astype(np.float32)


def _compute_ommatidia_coms(retina: Retina) -> np.ndarray:
    coms = np.empty((retina.num_ommatidia_per_eye, 2), dtype=np.float32)
    for ommatidium_idx in range(retina.num_ommatidia_per_eye):
        mask = retina.ommatidia_id_map == ommatidium_idx + 1
        coms[ommatidium_idx] = np.argwhere(mask).mean(axis=0)
    return coms


def _odor_by_side(
    odor_intensity: np.ndarray,
    palps_antennae_weights: tuple[float, float],
) -> np.ndarray:
    odor_intensity = np.asarray(odor_intensity, dtype=np.float32)
    if odor_intensity.size == 0:
        return np.zeros(2, dtype=np.float32)
    odor_by_sensor = odor_intensity.reshape((-1, 2, 2))
    weighted = np.average(
        odor_by_sensor,
        axis=1,
        weights=np.asarray(palps_antennae_weights, dtype=np.float32),
    )
    return weighted.sum(axis=0).astype(np.float32)


def _empty_vision_features() -> dict[str, Any]:
    return {
        "com_per_eye": np.full((2, 2), np.nan, dtype=np.float32),
        "size_per_eye": np.zeros(2, dtype=np.float32),
        "center_deviation": np.zeros(2, dtype=np.float32),
        "object_mask_counts": np.zeros(2, dtype=np.int32),
        "object_azimuth": 0.0,
        "object_size": 0.0,
        "object_proximity_proxy": 0.0,
        "avoidance_direction": 0.0,
        "collision_risk": 0.0,
        "looming_signal": 0.0,
    }
