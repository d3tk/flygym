from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
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
FOLLOW_CAMERA_OFFSET = np.array((0.0, -7.5, 6.0), dtype=float)


def _follow_video_path(path: Path) -> Path:
    suffix = path.suffix or ".mp4"
    return path.with_name(f"{path.stem}_follow{suffix}")


@dataclass(slots=True)
class NavigationTaskConfig:
    timestep: float = 1e-4
    ccd_iterations: int = 300
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
    obstacle_count_range: tuple[int, int] | None = None
    inactive_obstacle_position: np.ndarray = field(
        default_factory=lambda: np.array([1000.0, 0.0], dtype=float)
    )
    layout_min_target_spawn_distance: float = 0.0
    layout_min_obstacle_spawn_distance: float = 0.0
    layout_min_obstacle_target_distance: float = 0.0
    layout_min_obstacle_obstacle_distance: float = 0.0
    layout_sampling_attempts: int = 1000
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
        self.inactive_obstacle_position = np.asarray(
            self.inactive_obstacle_position, dtype=float
        )
        if self.timestep <= 0:
            raise ValueError("timestep must be positive.")
        if self.ccd_iterations <= 0:
            raise ValueError("ccd_iterations must be positive.")
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
        if self.inactive_obstacle_position.shape != (2,):
            raise ValueError("inactive_obstacle_position must have shape (2,).")
        if self.obstacle_count_range is not None:
            min_count, max_count = self.obstacle_count_range
            if min_count < 0 or max_count < min_count:
                raise ValueError(
                    "obstacle_count_range must be ordered non-negative integers."
                )
            if max_count > self.obstacle_positions.shape[0]:
                raise ValueError(
                    "obstacle_count_range cannot exceed the number of obstacle slots."
                )
            self.obstacle_count_range = (int(min_count), int(max_count))
        clearance_values = (
            self.layout_min_target_spawn_distance,
            self.layout_min_obstacle_spawn_distance,
            self.layout_min_obstacle_target_distance,
            self.layout_min_obstacle_obstacle_distance,
        )
        if any(distance < 0 for distance in clearance_values):
            raise ValueError("layout clearance distances must be non-negative.")
        if self.layout_sampling_attempts <= 0:
            raise ValueError("layout_sampling_attempts must be positive.")
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


@dataclass(frozen=True, slots=True)
class NavigationLayout:
    spawn_position: np.ndarray
    spawn_yaw: float
    odor_source: np.ndarray
    obstacle_positions: np.ndarray
    active_obstacle_count: int

    @property
    def active_obstacle_positions(self) -> np.ndarray:
        return self.obstacle_positions[: self.active_obstacle_count]


class OdorObstacleNavigationTask(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    observation_shape = (8,)

    def __init__(self, config: NavigationTaskConfig | None = None) -> None:
        super().__init__()
        self.config = config or NavigationTaskConfig()
        self.action_space_name = self.config.action_space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.observation_shape,
            dtype=np.float32,
        )
        self.render_mode = "rgb_array" if self.config.render else None
        if self.action_space_name == "continuous":
            self.action_shape = (2,)
            self.action_n = None
            self.action_low = np.full(2, self.config.action_bounds[0], dtype=np.float32)
            self.action_high = np.full(
                2, self.config.action_bounds[1], dtype=np.float32
            )
            self.action_max_norm = self.config.action_max_norm
            self.action_space = spaces.Box(
                low=self.action_low,
                high=self.action_high,
                shape=self.action_shape,
                dtype=np.float32,
            )
        else:
            self.action_shape = ()
            self.action_n = len(ACTION_PRESET_NAMES)
            self.action_low = np.array([0], dtype=np.int64)
            self.action_high = np.array([self.action_n - 1], dtype=np.int64)
            self.action_max_norm = None
            self.action_space = spaces.Discrete(self.action_n)
        self._base_spawn_position = self.config.spawn_position.copy()
        self._base_spawn_yaw = float(self.config.spawn_yaw)
        self._base_odor_source = self.config.odor_source.copy()
        self._base_obstacle_positions = self.config.obstacle_positions.copy()
        self.current_odor_source = self._base_odor_source.copy()
        self.current_obstacle_positions = self._base_obstacle_positions.copy()
        self.current_active_obstacle_count = self._base_obstacle_positions.shape[0]
        self._rng = np.random.default_rng()
        self._decision_count = 0
        self._prev_distance_to_target: float | None = None
        self._prev_object_size = 0.0
        self._prev_fly_xy: np.ndarray | None = None
        self._prev_heading: float | None = None
        self._prev_policy_action_idx: int | None = None

        self.fly = self._make_fly()
        self.world = self._make_world()
        self.world.add_fly(
            self.fly,
            self.config.spawn_position,
            Rotation3D("quat", _yaw_to_quat_wxyz(self.config.spawn_yaw)),
            bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
            add_ground_contact_sensors=False,
        )
        self.camera = self._add_birdeye_camera() if self.config.render else None
        self.follow_camera = self._add_follow_camera() if self.config.render else None
        self.cameras = [
            camera
            for camera in (self.camera, self.follow_camera)
            if camera is not None
        ]
        self.sim = Simulation(self.world)
        self.sim.mj_model.opt.timestep = self.config.timestep
        self._follow_camera_id = self._camera_id(self.follow_camera)
        if self.cameras:
            self.renderer = self.sim.set_renderer(
                self.cameras,
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

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        layout = self._sample_layout()
        self.sim.reset()
        self.sim.mj_model.opt.timestep = self.config.timestep
        self.controller.reset(seed=seed)
        self.set_layout(
            odor_source=layout.odor_source,
            obstacle_positions=layout.obstacle_positions,
            active_obstacle_count=layout.active_obstacle_count,
            forward=False,
        )
        self._set_root_state(layout.spawn_position, layout.spawn_yaw)
        if self.config.warmup_seconds > 0:
            self.sim.warmup(self.config.warmup_seconds)
        mj.mj_forward(self.sim.mj_model, self.sim.mj_data)

        self._decision_count = 0
        self._prev_object_size = 0.0
        fly_position = self._fly_position()
        self._prev_distance_to_target = self._distance_to_target(fly_position)
        self._prev_fly_xy = np.asarray(fly_position[:2], dtype=np.float32).copy()
        self._prev_heading = self._fly_heading()
        self._prev_policy_action_idx = None
        observation, features = self._make_observation()
        info = self._make_info(
            features,
            reward_terms={"progress": 0.0, "success": 0.0, "total": 0.0},
            terminated=False,
            truncated=False,
            forward_speed=0.0,
            yaw_rate=0.0,
            policy_action_idx=None,
            policy_action_label="",
            switched_action=False,
        )
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        policy_action_idx, policy_action_label = navigation_policy_action_from_raw(
            action,
            action_space=self.action_space_name,
        )
        descending_signal = resolve_navigation_action(
            action,
            action_space=self.action_space_name,
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
            if self.renderer is not None:
                self._render_as_needed()

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
        current_fly_xy = np.asarray(fly_position[:2], dtype=np.float32)
        current_heading = self._fly_heading()
        forward_speed, yaw_rate = compute_motion_diagnostics(
            self._prev_fly_xy,
            self._prev_heading,
            current_fly_xy,
            current_heading,
            self.config.decision_interval,
        )
        switched_action = (
            policy_action_idx is not None
            and self._prev_policy_action_idx is not None
            and policy_action_idx != self._prev_policy_action_idx
        )
        self._prev_fly_xy = current_fly_xy.copy()
        self._prev_heading = current_heading
        self._prev_policy_action_idx = policy_action_idx
        observation, features = self._make_observation()
        low_level_action = None
        if self.action_space_name == "continuous":
            low_level_action = {
                "left_force": float(descending_signal[0]),
                "right_force": float(descending_signal[1]),
                "groom": False,
            }
        info = self._make_info(
            features,
            reward_terms=reward_terms,
            terminated=terminated,
            truncated=truncated,
            forward_speed=forward_speed,
            yaw_rate=yaw_rate,
            policy_action_idx=policy_action_idx,
            policy_action_label=policy_action_label,
            switched_action=switched_action,
            low_level_action=low_level_action,
        )
        return observation, reward, terminated, truncated, info

    def set_layout(
        self,
        *,
        odor_source: np.ndarray | None = None,
        obstacle_positions: np.ndarray | None = None,
        active_obstacle_count: int | None = None,
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
            if active_obstacle_count is None:
                active_obstacle_count = obstacle_positions.shape[0]
        if active_obstacle_count is not None:
            active_obstacle_count = int(active_obstacle_count)
            if not 0 <= active_obstacle_count <= self.current_obstacle_positions.shape[0]:
                raise ValueError(
                    "active_obstacle_count must be between 0 and the number of "
                    "configured obstacle slots."
                )
            self.current_active_obstacle_count = active_obstacle_count
        if forward:
            mj.mj_forward(self.sim.mj_model, self.sim.mj_data)

    def render(self) -> list[np.ndarray | None]:
        if self.renderer is None:
            return [None]
        if not self._render_as_needed():
            return [None for _ in self.renderer.frames]
        return [frames[-1] for frames in self.renderer.frames.values()]

    def save_video(self, path: str | Path) -> None:
        if self.renderer is None:
            raise RuntimeError("render=True is required to save video.")
        output_path = Path(path)
        if self.follow_camera is None:
            self.renderer.save_video(output_path)
            return
        self.renderer.save_video(
            {
                self.camera: output_path,
                self.follow_camera: _follow_video_path(output_path),
            }
        )

    def close(self) -> None:
        self.sim.close()

    def _make_fly(self):
        fly = make_locomotion_fly(colorize=self.config.render)
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

    def _add_birdeye_camera(self):
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

    def _add_follow_camera(self):
        return self.world.mjcf_root.worldbody.add(
            "camera",
            name="navigation_follow",
            mode="fixed",
            pos=tuple(FOLLOW_CAMERA_OFFSET),
            fovy=30.0,
            **Rotation3D("xyaxes", (1, 0, 0, 0, 0.6, 0.8)).as_kwargs(),
        )

    def _camera_id(self, camera) -> int | None:
        if camera is None:
            return None
        return mj.mj_name2id(
            self.sim.mj_model,
            mj.mjtObj.mjOBJ_CAMERA,
            camera.full_identifier,
        )

    def _render_as_needed(self) -> bool:
        self._update_follow_camera()
        return self.sim.render_as_needed()

    def _update_follow_camera(self) -> None:
        if self._follow_camera_id is None:
            return
        self.sim.mj_model.cam_pos[self._follow_camera_id] = (
            self._fly_position().astype(float) + FOLLOW_CAMERA_OFFSET
        )

    def _sample_layout(self) -> NavigationLayout:
        return _sample_navigation_layout(
            self.config,
            self._rng,
            base_spawn_position=self._base_spawn_position,
            base_spawn_yaw=self._base_spawn_yaw,
            base_odor_source=self._base_odor_source,
            base_obstacle_positions=self._base_obstacle_positions,
        )

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
        odor_features = _finite_float32(odor_features)
        com_per_eye = _finite_float32(vision_features["com_per_eye"])
        size_per_eye = _finite_float32(vision_features["size_per_eye"])
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
        observation = _finite_float32(observation)
        features = {
            "odor_left": float(odor_features[0]),
            "odor_right": float(odor_features[1]),
            **vision_features,
        }
        return observation, features

    def _process_vision(self, vision_input: np.ndarray) -> dict[str, Any]:
        if self._retina is None or self._ommatidia_coms is None:
            return _empty_vision_features()

        vision_input = _finite_float32(vision_input)
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
        forward_speed: float = 0.0,
        yaw_rate: float = 0.0,
        policy_action_idx: int | None = None,
        policy_action_label: str = "",
        switched_action: bool = False,
        low_level_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fly_position = self._fly_position()
        info = {
            "sim_time": float(self.sim.time),
            "decision_count": self._decision_count,
            "fly_position": fly_position.copy(),
            "target_position": self.target_position,
            "obstacle_positions": self.current_obstacle_positions.copy(),
            "active_obstacle_count": int(self.current_active_obstacle_count),
            "active_obstacle_positions": self.current_obstacle_positions[
                : self.current_active_obstacle_count
            ].copy(),
            "distance_to_target": self._distance_to_target(fly_position),
            "processed_features": features,
            "reward_terms": reward_terms,
            "success": terminated,
            "is_success": terminated,
            "terminated": terminated,
            "truncated": truncated,
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

    def _fly_heading(self) -> float:
        quat = self.sim.mj_data.qpos[
            self._root_qposadr + 3 : self._root_qposadr + 7
        ]
        return _heading_from_quat_wxyz(quat)

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


def _sample_navigation_layout(
    config: NavigationTaskConfig,
    rng: np.random.Generator,
    *,
    base_spawn_position: np.ndarray | None = None,
    base_spawn_yaw: float | None = None,
    base_odor_source: np.ndarray | None = None,
    base_obstacle_positions: np.ndarray | None = None,
) -> NavigationLayout:
    spawn_base = (
        config.spawn_position
        if base_spawn_position is None
        else np.asarray(base_spawn_position, dtype=float)
    )
    spawn_yaw_base = config.spawn_yaw if base_spawn_yaw is None else base_spawn_yaw
    odor_base = (
        config.odor_source
        if base_odor_source is None
        else np.asarray(base_odor_source, dtype=float)
    )
    obstacle_base = (
        config.obstacle_positions
        if base_obstacle_positions is None
        else np.asarray(base_obstacle_positions, dtype=float)
    )

    if not config.randomize_layout:
        obstacle_positions = obstacle_base.copy()
        return NavigationLayout(
            spawn_position=spawn_base.copy(),
            spawn_yaw=float(spawn_yaw_base),
            odor_source=odor_base.copy(),
            obstacle_positions=obstacle_positions,
            active_obstacle_count=obstacle_positions.shape[0],
        )

    for _attempt in range(config.layout_sampling_attempts):
        spawn_position = spawn_base.copy()
        spawn_yaw = float(spawn_yaw_base)
        odor_source = odor_base.copy()
        obstacle_positions = obstacle_base.copy()

        spawn_position[:2] += rng.uniform(
            -config.spawn_xy_jitter,
            config.spawn_xy_jitter,
            size=2,
        )
        spawn_yaw += float(
            rng.uniform(
                -config.spawn_yaw_jitter,
                config.spawn_yaw_jitter,
            )
        )
        odor_source[0, 0] = rng.uniform(*config.odor_source_x_range)
        odor_source[0, 1] = rng.uniform(*config.odor_source_y_range)

        active_obstacle_count = obstacle_positions.shape[0]
        if config.obstacle_count_range is not None:
            min_count, max_count = config.obstacle_count_range
            active_obstacle_count = int(rng.integers(min_count, max_count + 1))
            obstacle_positions[:] = config.inactive_obstacle_position

        if active_obstacle_count:
            obstacle_positions[:active_obstacle_count, 0] = rng.uniform(
                *config.obstacle_x_range,
                size=active_obstacle_count,
            )
            obstacle_positions[:active_obstacle_count, 1] = rng.uniform(
                *config.obstacle_y_range,
                size=active_obstacle_count,
            )

        layout = NavigationLayout(
            spawn_position=spawn_position,
            spawn_yaw=spawn_yaw,
            odor_source=odor_source,
            obstacle_positions=obstacle_positions,
            active_obstacle_count=active_obstacle_count,
        )
        if _navigation_layout_is_valid(config, layout):
            return layout

    raise RuntimeError(
        "Failed to sample a valid navigation layout after "
        f"{config.layout_sampling_attempts} attempts. "
        "Relax the layout clearance constraints or widen the sampling ranges."
    )


def _navigation_layout_is_valid(
    config: NavigationTaskConfig,
    layout: NavigationLayout,
) -> bool:
    spawn_xy = layout.spawn_position[:2]
    target_xy = layout.odor_source[0, :2]
    active_obstacles = layout.active_obstacle_positions

    if (
        np.linalg.norm(target_xy - spawn_xy)
        < config.layout_min_target_spawn_distance
    ):
        return False

    if active_obstacles.size == 0:
        return True

    if config.layout_min_obstacle_spawn_distance > 0:
        distances = np.linalg.norm(active_obstacles - spawn_xy, axis=1)
        if np.any(distances < config.layout_min_obstacle_spawn_distance):
            return False

    if config.layout_min_obstacle_target_distance > 0:
        distances = np.linalg.norm(active_obstacles - target_xy, axis=1)
        if np.any(distances < config.layout_min_obstacle_target_distance):
            return False

    if (
        config.layout_min_obstacle_obstacle_distance > 0
        and active_obstacles.shape[0] > 1
    ):
        for obstacle_idx in range(active_obstacles.shape[0] - 1):
            distances = np.linalg.norm(
                active_obstacles[obstacle_idx + 1 :] - active_obstacles[obstacle_idx],
                axis=1,
            )
            if np.any(distances < config.layout_min_obstacle_obstacle_distance):
                return False

    return True


def _yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = yaw / 2
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _heading_from_quat_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat, dtype=float)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def navigation_policy_action_from_raw(
    action: Any,
    *,
    action_space: str,
) -> tuple[int | None, str]:
    if action_space == "continuous":
        return None, "low_level"
    action_array = np.asarray(action)
    if action_array.shape == (1,):
        action_value = action_array[0]
    elif action_array.shape == ():
        action_value = action_array.item()
    else:
        return None, ""
    action_idx = int(action_value)
    if action_idx < 0 or action_idx >= len(ACTION_PRESET_NAMES):
        return action_idx, ""
    return action_idx, ACTION_PRESET_NAMES[action_idx]


def compute_motion_diagnostics(
    prev_fly_xy: np.ndarray | None,
    prev_heading: float | None,
    current_fly_xy: np.ndarray,
    current_heading: float,
    decision_interval: float,
) -> tuple[float, float]:
    decision_dt = max(float(decision_interval), np.finfo(np.float32).eps)
    if prev_fly_xy is None or prev_heading is None:
        return 0.0, 0.0
    delta_xy = current_fly_xy - prev_fly_xy
    previous_forward = np.array(
        [math.cos(prev_heading), math.sin(prev_heading)],
        dtype=np.float32,
    )
    forward_speed = float(np.dot(delta_xy, previous_forward) / decision_dt)
    yaw_rate = float(_wrap_angle(current_heading - prev_heading) / decision_dt)
    return forward_speed, yaw_rate


def _finite_float32(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    arr = np.asarray(value)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if np.all(np.isfinite(arr)):
        return arr
    return np.nan_to_num(
        arr,
        copy=True,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)


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


def _odor_intensity_from_sources(
    sensor_positions: np.ndarray,
    odor_source: np.ndarray,
    peak_odor_intensity: np.ndarray,
) -> np.ndarray:
    sensor_positions = np.asarray(sensor_positions, dtype=float)
    odor_source = np.asarray(odor_source, dtype=float)
    peak_odor_intensity = np.asarray(peak_odor_intensity, dtype=float)
    n_sensors = sensor_positions.shape[0]
    odor_sources = np.repeat(
        np.repeat(
            odor_source[:, np.newaxis, np.newaxis, :],
            peak_odor_intensity.shape[1],
            axis=1,
        ),
        n_sensors,
        axis=2,
    )
    peak_intensity = np.repeat(
        peak_odor_intensity[:, :, np.newaxis],
        n_sensors,
        axis=2,
    )
    sensor_positions = sensor_positions[np.newaxis, np.newaxis, :, :]
    distance = np.linalg.norm(sensor_positions - odor_sources, axis=3)
    return (peak_intensity * distance**-2).sum(axis=0)


def _odor_by_side(
    odor_intensity: np.ndarray,
    palps_antennae_weights: tuple[float, float],
) -> np.ndarray:
    odor_intensity = _finite_float32(odor_intensity)
    if odor_intensity.size == 0:
        return np.zeros(2, dtype=np.float32)
    odor_by_sensor = odor_intensity.reshape((-1, 2, 2))
    weights = _finite_float32(palps_antennae_weights)
    if weights.shape != (2,) or float(weights.sum()) <= 0.0:
        weights = np.ones(2, dtype=np.float32)
    weighted = np.average(
        odor_by_sensor,
        axis=1,
        weights=weights,
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
