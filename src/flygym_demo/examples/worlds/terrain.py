from __future__ import annotations

from typing import Optional

import numpy as np

from .base import DemoWorld, add_checker_plane


class GappedTerrainWorld(DemoWorld):
    def __init__(
        self,
        name: str = "gapped_terrain_world",
        x_range: tuple[float, float] = (-10, 25),
        y_range: tuple[float, float] = (-20, 20),
        gap_width: float = 0.3,
        block_width: float = 1.0,
        gap_depth: float = 2.0,
        ground_alpha: float = 1.0,
        scale_bar_pos: Optional[tuple[float, float, float]] = None,
    ) -> None:
        super().__init__(name=name)
        self.x_range = x_range
        self.y_range = y_range
        block_centers = np.arange(
            x_range[0] + block_width / 2, x_range[1], block_width + gap_width
        )
        box_size = (block_width / 2, (y_range[1] - y_range[0]) / 2, gap_depth / 2)
        for x_pos in block_centers:
            geom = self.mjcf_root.worldbody.add(
                "geom",
                type="box",
                name=f"ground_element_x{x_pos:.3f}",
                size=box_size,
                pos=(x_pos, 0, 0),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
                contype=0,
                conaffinity=0,
            )
            self.ground_geoms.append(geom)
        self.mjcf_root.worldbody.add(
            "geom",
            type="plane",
            name="ground_base",
            pos=(np.mean(x_range), 0, -gap_depth / 2),
            rgba=(0.3, 0.3, 0.3, ground_alpha),
            size=((x_range[1] - x_range[0]) / 2, max(abs(y_range[0]), abs(y_range[1])), 1),
        )
        if scale_bar_pos:
            self.mjcf_root.worldbody.add(
                "geom", type="cylinder", size=(0.05, 0.5), pos=scale_bar_pos, rgba=(0, 0, 0, 1)
            )


class BlocksTerrainWorld(DemoWorld):
    def __init__(
        self,
        name: str = "blocks_terrain_world",
        x_range: tuple[float, float] = (-10, 25),
        y_range: tuple[float, float] = (-20, 20),
        block_size: float = 1.3,
        height_range: tuple[float, float] = (0.35, 0.35),
        ground_alpha: float = 1.0,
        rand_seed: int = 0,
    ) -> None:
        super().__init__(name=name)
        self.x_range = x_range
        self.y_range = y_range
        rng = np.random.RandomState(rand_seed)
        x_centers = np.arange(x_range[0] + block_size / 2, x_range[1], block_size)
        y_centers = np.arange(y_range[0] + block_size / 2, y_range[1], block_size)
        for i, x_pos in enumerate(x_centers):
            for j, y_pos in enumerate(y_centers):
                height = 0.1 if (i % 2 == 1) != (j % 2 == 1) else 0.1 + rng.uniform(*height_range)
                geom = self.mjcf_root.worldbody.add(
                    "geom",
                    type="box",
                    name=f"ground_element_x{x_pos:.3f}_y{y_pos:.3f}",
                    size=(block_size * 0.55, block_size * 0.55, height / 2 + block_size / 2),
                    pos=(x_pos, y_pos, height / 2 - block_size / 2),
                    rgba=(0.3, 0.3, 0.3, ground_alpha),
                    contype=0,
                    conaffinity=0,
                )
                self.ground_geoms.append(geom)


class MixedTerrainWorld(DemoWorld):
    def __init__(
        self,
        name: str = "mixed_terrain_world",
        gap_width: float = 0.3,
        gapped_block_width: float = 1.0,
        gap_depth: float = 2.0,
        block_size: float = 1.3,
        height_range: tuple[float, float] = (0.35, 0.35),
        ground_alpha: float = 1.0,
        rand_seed: int = 0,
    ) -> None:
        super().__init__(name=name)
        rng = np.random.RandomState(rand_seed)
        y_range = (-20, 20)
        for x_range in [(-4, 5), (5, 14), (14, 23)]:
            x_centers = np.arange(
                x_range[0] + block_size / 2, x_range[0] + block_size * 3, block_size
            )
            y_centers = np.arange(y_range[0] + block_size / 2, y_range[1], block_size)
            for i, x_pos in enumerate(x_centers):
                for j, y_pos in enumerate(y_centers):
                    height = 0.1 if (i % 2 == 1) != (j % 2 == 1) else 0.1 + rng.uniform(*height_range)
                    geom = self.mjcf_root.worldbody.add(
                        "geom",
                        type="box",
                        name=f"ground_element_block_x{x_pos:.3f}_y{y_pos:.3f}",
                        size=(block_size * 0.55, block_size * 0.55, height / 2 + block_size / 2),
                        pos=(x_pos, y_pos, height / 2 - block_size / 2 - np.mean(height_range) - 0.1),
                        rgba=(0.3, 0.3, 0.3, ground_alpha),
                        contype=0,
                        conaffinity=0,
                    )
                    self.ground_geoms.append(geom)
            curr_x = x_range[0] + block_size * 3
            arena_width = y_range[1] - y_range[0]
            for width in [gapped_block_width / 2, gapped_block_width]:
                geom = self.mjcf_root.worldbody.add(
                    "geom",
                    type="box",
                    name=f"ground_element_gap_x{curr_x:.3f}",
                    size=(width / 2, arena_width / 2, gap_depth / 2),
                    pos=(curr_x + width / 2, 0, -gap_depth / 2),
                    rgba=(0.3, 0.3, 0.3, ground_alpha),
                    contype=0,
                    conaffinity=0,
                )
                self.ground_geoms.append(geom)
                curr_x += width + gap_width
            remaining = x_range[1] - curr_x
            if remaining > 0:
                geom = self.mjcf_root.worldbody.add(
                    "geom",
                    type="box",
                    name=f"ground_element_flat_x{curr_x:.3f}",
                    size=(remaining / 2, arena_width / 2, gap_depth / 2),
                    pos=(curr_x + remaining / 2, 0, -gap_depth / 2),
                    rgba=(0.3, 0.3, 0.3, ground_alpha),
                    contype=0,
                    conaffinity=0,
                )
                self.ground_geoms.append(geom)
            self.mjcf_root.worldbody.add(
                "geom",
                type="plane",
                name=f"ground_base_{x_range[0]}",
                pos=(np.mean(x_range), 0, -gap_depth / 2),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
                size=((x_range[1] - x_range[0]) / 2, max(abs(y_range[0]), abs(y_range[1])), 1),
            )


class FlatDemoWorld(DemoWorld):
    def __init__(self, name: str = "flat_demo_world", half_size: float = 1000.0) -> None:
        super().__init__(name=name)
        self.ground_geoms.append(add_checker_plane(self.mjcf_root, "ground_plane", (half_size, half_size)))
