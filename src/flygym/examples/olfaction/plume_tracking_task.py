from __future__ import annotations

import numpy as np
import mujoco as mj

from flygym.compose import Fly, OdorPlumeWorld
from flygym.examples.locomotion import HybridTurningController, apply_locomotion_action
from flygym.examples.olfaction.common import (
    add_fly_to_odor_world,
    make_olfaction_observation,
)
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D


class PlumeNavigationTask:
    """V2 plume navigation task using a hybrid turning locomotion controller."""

    def __init__(
        self,
        fly: Fly,
        arena: OdorPlumeWorld,
        *,
        cameras=None,
        spawn_position: tuple[float, float, float] = (40, 80, 0.8),
        spawn_rotation: Rotation3D | None = None,
        render_plume_alpha: float = 0.75,
        intensity_display_vmax: float = 1.0,
        camera_res: tuple[int, int] = (240, 320),
        playback_speed: float = 0.2,
        output_fps: int = 25,
    ) -> None:
        if spawn_rotation is None:
            spawn_rotation = Rotation3D("quat", (1, 0, 0, 0))
        if fly.name not in arena.fly_lookup:
            add_fly_to_odor_world(arena, fly, spawn_position, spawn_rotation)

        self.fly = fly
        self.arena = arena
        self.sim = Simulation(arena)
        mj.mj_forward(self.sim.mj_model, self.sim.mj_data)
        self.controller = HybridTurningController(
            timestep=self.sim.timestep,
            output_dof_order=fly.get_actuated_jointdofs_order("position"),
        )
        self._render_plume_alpha = render_plume_alpha
        self._intensity_display_vmax = intensity_display_vmax
        self._grid_idx_by_camera = {}
        self._cached_plume_by_camera = {}
        self._plume_last_update_time = -np.inf

        if cameras:
            self.renderer = self.sim.set_renderer(
                cameras,
                camera_res=camera_res,
                playback_speed=playback_speed,
                output_fps=output_fps,
            )
            self._cache_camera_projection()
        else:
            self.renderer = None

    @property
    def timestep(self) -> float:
        return self.sim.timestep

    @property
    def curr_time(self) -> float:
        return self.sim.time

    def reset(self) -> tuple[dict[str, np.ndarray], dict]:
        self.sim.reset()
        mj.mj_forward(self.sim.mj_model, self.sim.mj_data)
        self.controller.reset()
        self._cached_plume_by_camera = {}
        self._plume_last_update_time = -np.inf
        return self.get_observation(), {}

    def get_observation(self) -> dict[str, np.ndarray]:
        return make_olfaction_observation(self.sim, self.fly.name)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, np.ndarray | bool]]:
        locomotion_action = self.controller.step(
            np.asarray(action), self.sim, self.fly.name
        )
        apply_locomotion_action(self.sim, self.fly.name, locomotion_action)
        self.sim.step()
        obs = self.get_observation()
        info = {
            "net_corrections": self.controller.last_info["net_corrections"].copy(),
            "joints": locomotion_action.joint_angles.copy(),
            "adhesion": locomotion_action.adhesion_onoff.copy(),
            "flip": False,
        }
        return obs, 0.0, False, bool(np.isnan(obs["odor_intensity"]).any()), info

    def render(self) -> list[np.ndarray | None]:
        if self.renderer is None:
            return [None]
        if not self.sim.render_as_needed():
            return [None for _ in self.renderer.frames]

        camera_names = list(self.renderer.frames)
        self._overlay_plume(camera_names)
        return [self.renderer.frames[name][-1] for name in camera_names]

    def _cache_camera_projection(self) -> None:
        for cam_name, cam_id in self.renderer._cameras_names2id.items():
            self._grid_idx_by_camera[cam_name] = self._grid_indices_for_camera(cam_id)

    def _grid_indices_for_camera(self, cam_id: int) -> np.ndarray:
        height, width = self.renderer.camera_res
        fovy = np.deg2rad(self.sim.mj_model.cam_fovy[cam_id])
        fovx = 2 * np.arctan(np.tan(fovy / 2) * width / height)
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        x = (2 * (cols + 0.5) / width - 1) * np.tan(fovx / 2)
        y = (1 - 2 * (rows + 0.5) / height) * np.tan(fovy / 2)
        camera_dirs = np.stack((x, y, -np.ones_like(x)), axis=-1)

        cam_pos = self.sim.mj_data.cam_xpos[cam_id]
        cam_xmat = self.sim.mj_data.cam_xmat[cam_id].reshape(3, 3)
        world_dirs = camera_dirs @ cam_xmat.T
        scale = -cam_pos[2] / world_dirs[:, :, 2]
        world_pos = cam_pos + world_dirs * scale[:, :, np.newaxis]
        grid_idx = np.floor(world_pos[:, :, :2] / self.arena.dimension_scale_factor)
        invalid = (
            (scale <= 0)
            | (grid_idx[:, :, 0] < 0)
            | (grid_idx[:, :, 1] < 0)
            | (grid_idx[:, :, 0] >= self.arena.plume_grid.shape[2])
            | (grid_idx[:, :, 1] >= self.arena.plume_grid.shape[1])
        )
        grid_idx[invalid] = -1
        return grid_idx.astype(np.int32)

    def _overlay_plume(self, camera_names: list[str]) -> None:
        update_needed = (
            self.curr_time - self._plume_last_update_time
            > self.arena.plume_update_interval
        )
        if update_needed or not self._cached_plume_by_camera:
            t_idx = int(self.curr_time * self.arena.plume_simulation_fps)
            plume_grid = self.arena.plume_grid[t_idx, :, :].astype(np.float32)
            self._cached_plume_by_camera = {
                name: _resample_plume_image(self._grid_idx_by_camera[name], plume_grid)
                for name in camera_names
            }
            self._plume_last_update_time = self.curr_time

        for idx, name in enumerate(camera_names):
            frame = self.renderer.frames[name][-1]
            plume_img = self._cached_plume_by_camera[name][:, :, np.newaxis]
            plume_img = np.nan_to_num(plume_img) * self._render_plume_alpha
            frame = np.clip(frame - plume_img * 255, 0, 255).astype(np.uint8)
            if idx == 0:
                self._add_intensity_indicator(frame)
            self.renderer.frames[name][-1] = frame

    def _add_intensity_indicator(self, frame: np.ndarray) -> None:
        odor = self.get_observation()["odor_intensity"]
        mean_intensity = 0.0 if np.isnan(odor).all() else np.nanmean(odor)
        width = int(
            frame.shape[1]
            * np.clip(mean_intensity / self._intensity_display_vmax, 0, 1)
        )
        frame[-10:, :width] = (255, 0, 0)


def _resample_plume_image(grid_idx_all: np.ndarray, plume_grid: np.ndarray) -> np.ndarray:
    plume_img = np.full(grid_idx_all.shape[:2], np.nan, dtype=np.float32)
    x_idx = grid_idx_all[:, :, 0]
    y_idx = grid_idx_all[:, :, 1]
    valid = x_idx >= 0
    plume_img[valid] = plume_grid[y_idx[valid], x_idx[valid]]
    return plume_img
