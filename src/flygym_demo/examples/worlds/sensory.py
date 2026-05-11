from __future__ import annotations

import mujoco as mj
import numpy as np

from .terrain import FlatDemoWorld


class OdorWorld(FlatDemoWorld):
    def __init__(
        self,
        name: str = "odor_world",
        size: tuple[float, float] = (300, 300),
        odor_source: np.ndarray = np.array([[10, 0, 0]], dtype=float),
        peak_odor_intensity: np.ndarray = np.array([[1]], dtype=float),
        diffuse_func=lambda x: np.where(x == 0, np.inf, x**-2),
        marker_size: float = 0.25,
    ) -> None:
        super().__init__(name=name, half_size=max(size))
        self.odor_source = np.asarray(odor_source, dtype=np.float64)
        self.peak_odor_intensity = np.asarray(peak_odor_intensity, dtype=np.float64)
        if self.odor_source.shape[0] != self.peak_odor_intensity.shape[0]:
            raise ValueError("odor_source and peak_odor_intensity source counts must match")
        self.diffuse_func = diffuse_func
        colors = [(0.12, 0.47, 0.71, 1), (1.0, 0.5, 0.05, 1), (0.17, 0.63, 0.17, 1)]
        for i, pos in enumerate(self.odor_source):
            marker_body = self.mjcf_root.worldbody.add("body", name=f"odor_source_marker_{i}", pos=pos, mocap=True)
            marker_body.add("geom", type="capsule", size=(marker_size, marker_size), rgba=colors[i % len(colors)])

    @property
    def odor_dimensions(self) -> int:
        return self.peak_odor_intensity.shape[1]

    def get_olfaction(self, sensor_pos: np.ndarray) -> np.ndarray:
        sensor_pos = np.asarray(sensor_pos, dtype=np.float64)
        source = self.odor_source[:, np.newaxis, np.newaxis, :]
        source = np.repeat(source, self.odor_dimensions, axis=1)
        source = np.repeat(source, sensor_pos.shape[0], axis=2)
        sensor = sensor_pos[np.newaxis, np.newaxis, :, :]
        dist = np.linalg.norm(sensor - source, axis=3)
        peak = np.repeat(self.peak_odor_intensity[:, :, np.newaxis], sensor_pos.shape[0], axis=2)
        return (peak * self.diffuse_func(dist)).sum(axis=0)


class OdorPlumeWorld(FlatDemoWorld):
    """World backed by a precomputed plume HDF5 file."""

    def __init__(
        self,
        plume_data_path,
        name: str = "odor_plume_world",
        arena_size: tuple[float, float] = (80, 20),
        plume_simulation_fps: float = 20.0,
    ) -> None:
        super().__init__(name=name, half_size=max(arena_size))
        self.plume_data_path = plume_data_path
        self.arena_size = np.asarray(arena_size, dtype=np.float64)
        self.plume_simulation_fps = plume_simulation_fps
        self._dataset = None

    def _open(self):
        if self._dataset is None:
            import h5py

            self._dataset = h5py.File(self.plume_data_path, "r")
        return self._dataset

    def get_olfaction(self, sensor_pos: np.ndarray, t: float = 0.0) -> np.ndarray:
        ds = self._open()
        plume = ds["plume"]
        frame_idx = min(int(t * self.plume_simulation_fps), plume.shape[0] - 1)
        frame = plume[frame_idx]
        xy = np.asarray(sensor_pos)[:, :2]
        px = np.clip(((xy[:, 0] / self.arena_size[0]) + 0.5) * (frame.shape[1] - 1), 0, frame.shape[1] - 1).astype(int)
        py = np.clip(((xy[:, 1] / self.arena_size[1]) + 0.5) * (frame.shape[0] - 1), 0, frame.shape[0] - 1).astype(int)
        return frame[py, px][np.newaxis, :]


class MovingObjectWorld(FlatDemoWorld):
    def __init__(
        self,
        name: str = "moving_object_world",
        init_pos: tuple[float, float, float] = (5, 0, 2),
        radius: float = 1.0,
        speed: float = 10.0,
        lateral_magnitude: float = 0.5,
    ) -> None:
        super().__init__(name=name, half_size=100)
        self.init_pos = np.asarray(init_pos, dtype=np.float64)
        self.radius = radius
        self.speed = speed
        self.lateral_magnitude = lateral_magnitude
        body = self.mjcf_root.worldbody.add("body", name="moving_object", pos=init_pos, mocap=True)
        body.add("geom", name="moving_object_geom", type="sphere", size=(radius,), rgba=(0.02, 0.02, 0.02, 1))
        self._body = body
        self._mocap_id = None

    def position_at(self, t: float) -> np.ndarray:
        x = self.init_pos[0] + self.speed * t
        y = self.init_pos[1] + self.lateral_magnitude * self.speed * np.sin(t)
        return np.array([x, y, self.init_pos[2]])

    def reset(self, sim) -> None:
        body_id = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_BODY, self._body.full_identifier)
        self._mocap_id = sim.mj_model.body_mocapid[body_id]
        if self._mocap_id >= 0:
            sim.mj_data.mocap_pos[self._mocap_id] = self.init_pos

    def step(self, sim, dt: float) -> None:
        if self._mocap_id is None:
            self.reset(sim)
        if self._mocap_id is not None and self._mocap_id >= 0:
            sim.mj_data.mocap_pos[self._mocap_id] = self.position_at(sim.time)


class MovingFlyWorld(MovingObjectWorld):
    """A lightweight moving fly-shaped visual target for vision tutorials."""

    def __init__(self, name: str = "moving_fly_world", **kwargs) -> None:
        super().__init__(name=name, radius=0.5, **kwargs)
        self._body.find("geom", "moving_object_geom").remove()
        self._body.add("geom", name="moving_fly_body", type="capsule", size=(0.25, 1.2), rgba=(0.02, 0.02, 0.02, 1), euler=(0, np.pi / 2, 0))
        self._body.add("geom", name="moving_fly_head", type="sphere", size=(0.35,), pos=(0.9, 0, 0), rgba=(0.02, 0.02, 0.02, 1))
