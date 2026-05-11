from __future__ import annotations

import numpy as np

from flygym import Simulation
from flygym.compose import ActuatorType
from flygym_demo.examples.common import ControllerOutput

from .steps import PreprogrammedSteps, get_cpg_biases


def calculate_ddt(theta, r, w, phi, nu, R, alpha):
    intrinsic_term = 2 * np.pi * nu
    phase_diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    coupling_term = (r * w * np.sin(phase_diff - phi)).sum(axis=1)
    dtheta_dt = intrinsic_term + coupling_term
    dr_dt = alpha * (R - r)
    return dtheta_dt, dr_dt


class CPGNetwork:
    def __init__(
        self,
        timestep: float,
        intrinsic_freqs: np.ndarray,
        intrinsic_amps: np.ndarray,
        coupling_weights: np.ndarray,
        phase_biases: np.ndarray,
        convergence_coefs: np.ndarray,
        init_phases: np.ndarray | None = None,
        init_magnitudes: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        self.timestep = timestep
        self.num_cpgs = intrinsic_freqs.size
        self.intrinsic_freqs = np.asarray(intrinsic_freqs, dtype=np.float64)
        self.intrinsic_amps = np.asarray(intrinsic_amps, dtype=np.float64)
        self.coupling_weights = np.asarray(coupling_weights, dtype=np.float64)
        self.phase_biases = np.asarray(phase_biases, dtype=np.float64)
        self.convergence_coefs = np.asarray(convergence_coefs, dtype=np.float64)
        self.random_state = np.random.RandomState(seed)
        self._init_phases = init_phases
        self._init_magnitudes = init_magnitudes
        self.reset()

    def step(self) -> None:
        dtheta_dt, dr_dt = calculate_ddt(
            theta=self.curr_phases,
            r=self.curr_magnitudes,
            w=self.coupling_weights,
            phi=self.phase_biases,
            nu=self.intrinsic_freqs,
            R=self.intrinsic_amps,
            alpha=self.convergence_coefs,
        )
        self.curr_phases += dtheta_dt * self.timestep
        self.curr_magnitudes += dr_dt * self.timestep

    def reset(
        self,
        init_phases: np.ndarray | None = None,
        init_magnitudes: np.ndarray | None = None,
    ) -> None:
        if init_phases is None:
            init_phases = self._init_phases
        if init_magnitudes is None:
            init_magnitudes = self._init_magnitudes
        if init_phases is None:
            self.curr_phases = self.random_state.random(self.num_cpgs) * 2 * np.pi
        else:
            self.curr_phases = np.asarray(init_phases, dtype=np.float64).copy()
        if init_magnitudes is None:
            self.curr_magnitudes = np.zeros(self.num_cpgs, dtype=np.float64)
        else:
            self.curr_magnitudes = np.asarray(init_magnitudes, dtype=np.float64).copy()


class CPGController:
    def __init__(
        self,
        timestep: float,
        *,
        preprogrammed_steps: PreprogrammedSteps | None = None,
        gait: str = "tripod",
        intrinsic_freqs: np.ndarray | None = None,
        intrinsic_amps: np.ndarray | None = None,
        convergence_coefs: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        self.preprogrammed_steps = preprogrammed_steps or PreprogrammedSteps()
        phase_biases = get_cpg_biases(gait)
        coupling_weights = (phase_biases > 0) * 10
        self.cpg_network = CPGNetwork(
            timestep=timestep,
            intrinsic_freqs=np.ones(6) * 12 if intrinsic_freqs is None else intrinsic_freqs,
            intrinsic_amps=np.ones(6) if intrinsic_amps is None else intrinsic_amps,
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_coefs=np.ones(6) * 20
            if convergence_coefs is None
            else convergence_coefs,
            seed=seed,
        )

    def reset(self) -> None:
        self.cpg_network.reset()

    def step(self, sim: Simulation, drive=None) -> ControllerOutput:
        fly_name = next(iter(sim.world.fly_lookup))
        fly = sim.world.fly_lookup[fly_name]
        self.cpg_network.step()
        dof_order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        targets = self.preprogrammed_steps.get_joint_angles_for_order(
            self.cpg_network.curr_phases, self.cpg_network.curr_magnitudes, dof_order
        )
        adhesion = self.preprogrammed_steps.get_adhesion_for_phases(
            self.cpg_network.curr_phases
        )
        return ControllerOutput(
            targets,
            adhesion,
            {
                "phases": self.cpg_network.curr_phases.copy(),
                "magnitudes": self.cpg_network.curr_magnitudes.copy(),
            },
        )
