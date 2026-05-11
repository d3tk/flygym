from __future__ import annotations

import numpy as np

from flygym import Simulation
from flygym_demo.examples.common import ControllerOutput
from flygym_demo.examples.locomotion import HybridTurningController


class SimpleVisualTaxisController:
    """Object-centering controller driven by a two-eye retina observation."""

    def __init__(
        self,
        walking_controller: HybridTurningController,
        retina,
        detection_threshold: float = 0.15,
        gain: float = 0.8,
    ) -> None:
        self.walking_controller = walking_controller
        self.retina = retina
        self.detection_threshold = detection_threshold
        self.gain = gain

    def reset(self) -> None:
        self.walking_controller.reset()

    def _turn_from_vision(self, vision: np.ndarray) -> float:
        # vision: (2 eyes, n_ommatidia, 2 channels)
        darkness = 1 - vision.max(axis=-1)
        object_mask = darkness > self.detection_threshold
        sizes = object_mask.mean(axis=1)
        if sizes.sum() == 0:
            return 0.0
        return self.gain * (sizes[1] - sizes[0])

    def step(self, sim: Simulation, drive=None) -> ControllerOutput:
        if drive is None:
            turn = 0.0
            vision = None
        else:
            vision = np.asarray(drive)
            turn = self._turn_from_vision(vision)
        descending = np.clip(np.array([1 - turn, 1 + turn]), -0.5, 1.5)
        output = self.walking_controller.step(sim, descending)
        output.metadata["vision_turn_bias"] = float(turn)
        if vision is not None:
            output.metadata["vision"] = vision.copy()
        return output
