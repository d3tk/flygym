from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flygym.compose import OdorWorld


@dataclass(slots=True)
class ObstacleOdorWorldConfig:
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

    def __post_init__(self) -> None:
        self.obstacle_positions = np.asarray(self.obstacle_positions, dtype=float)
        self.obstacle_colors = np.asarray(self.obstacle_colors, dtype=float)
        if (
            self.obstacle_positions.ndim != 2
            or self.obstacle_positions.shape[1] != 2
        ):
            raise ValueError("obstacle_positions must have shape (n_obstacles, 2).")
        if self.obstacle_radius <= 0:
            raise ValueError("obstacle_radius must be positive.")
        if self.obstacle_height <= 0:
            raise ValueError("obstacle_height must be positive.")
        if self.obstacle_colors.shape not in {
            (4,),
            (self.obstacle_positions.shape[0], 4),
        }:
            raise ValueError(
                "obstacle_colors must have shape (4,) or (n_obstacles, 4)."
            )


class ObstacleOdorWorld(OdorWorld):
    """Odor world with cylindrical obstacles included in explicit contact pairs."""

    def __init__(
        self,
        *,
        name: str = "obstacle_odor_world",
        half_size: float = 1000,
        odor_source: np.ndarray | None = None,
        peak_odor_intensity: np.ndarray | None = None,
        obstacle_positions: np.ndarray | None = None,
        obstacle_radius: float = 1.0,
        obstacle_height: float = 4.0,
        obstacle_colors: np.ndarray | tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        marker_colors: np.ndarray | None = None,
        marker_size: float = 0.25,
    ) -> None:
        super().__init__(
            name=name,
            half_size=half_size,
            odor_source=odor_source,
            peak_odor_intensity=peak_odor_intensity,
            marker_colors=marker_colors,
            marker_size=marker_size,
        )
        self.obstacle_config = ObstacleOdorWorldConfig(
            obstacle_positions=(
                ObstacleOdorWorldConfig().obstacle_positions
                if obstacle_positions is None
                else obstacle_positions
            ),
            obstacle_radius=obstacle_radius,
            obstacle_height=obstacle_height,
            obstacle_colors=np.asarray(obstacle_colors, dtype=float),
        )
        self.obstacle_positions = self.obstacle_config.obstacle_positions.copy()
        self.obstacle_bodies = []
        self.obstacle_geoms = []
        self._add_obstacles()

    def obstacle_body_full_identifiers(self) -> list[str]:
        return [body.full_identifier for body in self.obstacle_bodies]

    def odor_marker_body_full_identifiers(self) -> list[str]:
        return [
            self.mjcf_root.find("body", f"odor_source_marker_{idx}").full_identifier
            for idx in range(self.num_odor_sources)
        ]

    def _add_obstacles(self) -> None:
        cfg = self.obstacle_config
        colors = cfg.obstacle_colors
        if colors.shape == (4,):
            colors = np.repeat(colors[np.newaxis, :], cfg.obstacle_positions.shape[0], 0)

        material = self.mjcf_root.asset.add(
            "material",
            name="obstacle",
            reflectance=0.1,
        )
        z_pos = cfg.obstacle_height / 2
        for idx, (xy, rgba) in enumerate(zip(cfg.obstacle_positions, colors)):
            body = self.mjcf_root.worldbody.add(
                "body",
                name=f"obstacle_{idx}",
                mocap=True,
                pos=(float(xy[0]), float(xy[1]), float(z_pos)),
            )
            geom = body.add(
                "geom",
                type="cylinder",
                name=f"obstacle_{idx}",
                size=(float(cfg.obstacle_radius), float(cfg.obstacle_height / 2)),
                rgba=tuple(float(v) for v in rgba),
                material=material,
                contype=0,
                conaffinity=0,
            )
            self.obstacle_bodies.append(body)
            self.obstacle_geoms.append(geom)
            self.ground_geoms.append(geom)
