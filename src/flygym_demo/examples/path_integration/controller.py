from __future__ import annotations

import numpy as np

from flygym import Simulation
from flygym_demo.examples.common import ControllerOutput
from flygym_demo.examples.locomotion import HybridTurningController


class RandomExplorationController:
    """Stochastic bouts of straight walking and turning for path integration demos."""

    def __init__(
        self,
        walking_controller: HybridTurningController,
        bout_duration_s: tuple[float, float] = (0.2, 0.8),
        seed: int = 0,
    ) -> None:
        self.walking_controller = walking_controller
        self.bout_duration_s = bout_duration_s
        self.rng = np.random.RandomState(seed)
        self._remaining_s = 0.0
        self._drive = np.ones(2)

    def reset(self) -> None:
        self.walking_controller.reset()
        self._remaining_s = 0.0
        self._drive = np.ones(2)

    def _sample_bout(self) -> None:
        self._remaining_s = self.rng.uniform(*self.bout_duration_s)
        turn = self.rng.choice([-0.35, 0.0, 0.35], p=[0.25, 0.5, 0.25])
        self._drive = np.array([1 - turn, 1 + turn])

    def step(self, sim: Simulation, drive=None) -> ControllerOutput:
        if self._remaining_s <= 0:
            self._sample_bout()
        self._remaining_s -= sim.timestep
        out = self.walking_controller.step(sim, self._drive)
        out.metadata["exploration_drive"] = self._drive.copy()
        return out
