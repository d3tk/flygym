from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from flygym import Simulation
from flygym.compose import ActuatorType
from flygym_demo.examples.common import ControllerOutput

from .cpg_controller import CPGNetwork
from .steps import PreprogrammedSteps, get_cpg_biases


_DEFAULT_CORRECTION_VECTORS = {
    "f": np.array([-0.03, 0.0, 0.0, -0.03, 0.0, 0.03, 0.03]),
    "m": np.array([-0.015, 0.001, 0.025, -0.02, 0.0, -0.02, 0.0]),
    "h": np.array([0.0, 0.0, 0.0, -0.02, 0.0, 0.01, -0.02]),
}
_DEFAULT_CORRECTION_RATES = {"retraction": (800.0, 700.0), "stumbling": (2200.0, 1800.0)}


class HybridController:
    """CPG walking with lightweight sensory corrections for v2 simulations."""

    def __init__(
        self,
        timestep: float,
        *,
        preprogrammed_steps: PreprogrammedSteps | None = None,
        gait: str = "tripod",
        intrinsic_freqs: np.ndarray | None = None,
        intrinsic_amps: np.ndarray | None = None,
        convergence_coefs: np.ndarray | None = None,
        correction_vectors: dict[str, np.ndarray] | None = None,
        correction_rates: dict[str, tuple[float, float]] | None = None,
        stumbling_force_threshold: float = -1.0,
        seed: int = 0,
    ) -> None:
        self.preprogrammed_steps = preprogrammed_steps or PreprogrammedSteps()
        self.base_intrinsic_freqs = (
            np.ones(6) * 12 if intrinsic_freqs is None else np.asarray(intrinsic_freqs)
        )
        self.base_intrinsic_amps = (
            np.ones(6) if intrinsic_amps is None else np.asarray(intrinsic_amps)
        )
        phase_biases = get_cpg_biases(gait)
        self.cpg_network = CPGNetwork(
            timestep=timestep,
            intrinsic_freqs=self.base_intrinsic_freqs.copy(),
            intrinsic_amps=self.base_intrinsic_amps.copy(),
            coupling_weights=(phase_biases > 0) * 10,
            phase_biases=phase_biases,
            convergence_coefs=np.ones(6) * 20
            if convergence_coefs is None
            else convergence_coefs,
            seed=seed,
        )
        self.timestep = timestep
        self.correction_vectors = correction_vectors or _DEFAULT_CORRECTION_VECTORS
        self.correction_rates = correction_rates or _DEFAULT_CORRECTION_RATES
        self.stumbling_force_threshold = stumbling_force_threshold
        self.legs = self.preprogrammed_steps.legs
        self.right_leg_inversion = np.array([1, -1, -1, 1, -1, 1, 1])
        self.phasic_multiplier = self._init_phasic_gain()
        self.reset()

    def reset(self) -> None:
        self.cpg_network.reset()
        self.retraction_correction = np.zeros(6, dtype=np.float64)
        self.stumbling_correction = np.zeros(6, dtype=np.float64)

    def _init_phasic_gain(self, swing_extension=np.pi / 4):
        phasic_multiplier = {}
        for leg in self.legs:
            swing_start, swing_end = self.preprogrammed_steps.swing_period[leg]
            points = [
                swing_start,
                np.mean([swing_start, swing_end]),
                swing_end + swing_extension,
                np.mean([swing_end, 2 * np.pi]),
                2 * np.pi,
            ]
            vals = [0, 0.8, 0, -0.1, 0]
            phasic_multiplier[leg] = interp1d(points, vals, kind="linear", fill_value="extrapolate")
        return phasic_multiplier

    def _find_leg_for_retraction(self, sim: Simulation, fly_name: str) -> int | None:
        fly = sim.world.fly_lookup[fly_name]
        body_order = fly.get_bodysegs_order()
        body_positions = sim.get_body_positions(fly_name)
        thorax_z = body_positions[body_order.index(fly.root_segment), 2]
        tarsus_z = []
        for leg in self.legs:
            name = f"{leg}_tarsus5"
            idx = next((i for i, seg in enumerate(body_order) if seg.name == name), None)
            if idx is None:
                return None
            tarsus_z.append(body_positions[idx, 2])
        clearance = thorax_z - np.asarray(tarsus_z)
        sorted_idx = np.argsort(clearance)
        if clearance[sorted_idx[-1]] > clearance[sorted_idx[-3]] + 0.05:
            return int(sorted_idx[-1])
        return None

    def _find_stumbling_legs(self, sim: Simulation, fly_name: str) -> np.ndarray:
        try:
            contact_active, forces, *_ = sim.get_ground_contact_info(fly_name)
        except Exception:
            return np.zeros(6, dtype=bool)
        forward_force = forces[:, 0]
        return (contact_active > 0) & (forward_force < self.stumbling_force_threshold)

    def _update_corrections(self, sim: Simulation, fly_name: str) -> dict[str, np.ndarray]:
        retraction_leg = self._find_leg_for_retraction(sim, fly_name)
        stumbling_legs = self._find_stumbling_legs(sim, fly_name)

        rates = self.correction_rates["retraction"]
        if retraction_leg is not None:
            self.retraction_correction[retraction_leg] += rates[0] * self.timestep
        self.retraction_correction -= rates[1] * self.timestep
        self.retraction_correction = np.clip(self.retraction_correction, 0, 1)

        rates = self.correction_rates["stumbling"]
        self.stumbling_correction[stumbling_legs] += rates[0] * self.timestep
        self.stumbling_correction[~stumbling_legs] -= rates[1] * self.timestep
        self.stumbling_correction = np.clip(self.stumbling_correction, 0, 1)

        return {
            "retraction_leg": np.array(-1 if retraction_leg is None else retraction_leg),
            "stumbling_legs": stumbling_legs.copy(),
            "retraction_correction": self.retraction_correction.copy(),
            "stumbling_correction": self.stumbling_correction.copy(),
        }

    def _correction_targets_for_order(self, dof_order, phases) -> np.ndarray:
        values = []
        for dof in dof_order:
            leg = dof.child.pos
            leg_idx = self.legs.index(leg)
            legacy_key = (dof.parent.link, dof.child.link, dof.axis.value)
            dof_idx = self.preprogrammed_steps.dofs_per_leg.index(legacy_key)
            vec = self.correction_vectors[leg[1]].copy()
            if leg[0] == "r":
                vec *= self.right_leg_inversion
            phasic = float(self.phasic_multiplier[leg](phases[leg_idx] % (2 * np.pi)))
            corr = self.retraction_correction[leg_idx] + self.stumbling_correction[leg_idx] * phasic
            values.append(vec[dof_idx] * corr)
        return np.asarray(values, dtype=np.float32)

    def _apply_drive(self, drive) -> dict[str, np.ndarray]:
        self.cpg_network.intrinsic_freqs = self.base_intrinsic_freqs.copy()
        self.cpg_network.intrinsic_amps = self.base_intrinsic_amps.copy()
        return {}

    def step(self, sim: Simulation, drive=None) -> ControllerOutput:
        fly_name = next(iter(sim.world.fly_lookup))
        fly = sim.world.fly_lookup[fly_name]
        drive_metadata = self._apply_drive(drive)
        self.cpg_network.step()

        metadata = self._update_corrections(sim, fly_name)
        dof_order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        targets = self.preprogrammed_steps.get_joint_angles_for_order(
            self.cpg_network.curr_phases, self.cpg_network.curr_magnitudes, dof_order
        )
        targets += self._correction_targets_for_order(dof_order, self.cpg_network.curr_phases)
        adhesion = self.preprogrammed_steps.get_adhesion_for_phases(
            self.cpg_network.curr_phases
        )
        metadata.update(
            {
                "phases": self.cpg_network.curr_phases.copy(),
                "magnitudes": self.cpg_network.curr_magnitudes.copy(),
                **drive_metadata,
            }
        )
        return ControllerOutput(targets, adhesion, metadata)


class HybridTurningController(HybridController):
    """Hybrid walking controller receiving a two-value left/right descending drive."""

    def __init__(self, *args, amplitude_range: tuple[float, float] = (-0.5, 1.5), **kwargs):
        super().__init__(*args, **kwargs)
        self.amplitude_range = amplitude_range

    def _apply_drive(self, drive) -> dict[str, np.ndarray]:
        if drive is None:
            drive_arr = np.ones(2)
        else:
            drive_arr = np.asarray(drive, dtype=np.float64).reshape(2)
        drive_arr = np.clip(drive_arr, *self.amplitude_range)
        per_leg_drive = np.array(
            [drive_arr[0], drive_arr[0], drive_arr[0], drive_arr[1], drive_arr[1], drive_arr[1]]
        )
        self.cpg_network.intrinsic_freqs = self.base_intrinsic_freqs * np.clip(
            per_leg_drive, 0, None
        )
        self.cpg_network.intrinsic_amps = self.base_intrinsic_amps * per_leg_drive
        return {"drive": drive_arr.copy()}
