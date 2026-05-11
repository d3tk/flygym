from __future__ import annotations

import numpy as np

from flygym import Simulation
from flygym_demo.examples.common import ControllerOutput
from flygym_demo.examples.locomotion import HybridTurningController

from .sensors import read_odor


class SimpleOdorTaxisController:
    """Turn toward attractive odor and away from aversive odor using sensor asymmetry."""

    def __init__(
        self,
        walking_controller: HybridTurningController,
        odor_sensor_sites: dict[str, object],
        attractive_gain: float = 0.4,
        aversive_gain: float = 0.4,
    ) -> None:
        self.walking_controller = walking_controller
        self.odor_sensor_sites = odor_sensor_sites
        self.attractive_gain = attractive_gain
        self.aversive_gain = aversive_gain

    def reset(self) -> None:
        self.walking_controller.reset()

    def step(self, sim: Simulation, drive=None) -> ControllerOutput:
        odor = read_odor(sim, self.odor_sensor_sites)
        left = odor[:, [0, 2]].mean(axis=1)
        right = odor[:, [1, 3]].mean(axis=1)
        attractive_bias = (left[0] - right[0]) if odor.shape[0] >= 1 else 0
        aversive_bias = (right[1] - left[1]) if odor.shape[0] >= 2 else 0
        turn = self.attractive_gain * attractive_bias + self.aversive_gain * aversive_bias
        descending = np.clip(np.array([1 - turn, 1 + turn]), -0.5, 1.5)
        output = self.walking_controller.step(sim, descending)
        output.metadata["odor"] = odor.copy()
        output.metadata["odor_turn_bias"] = float(turn)
        return output
