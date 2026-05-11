from __future__ import annotations

import pickle
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from flygym.anatomy import JointDOF


class PreprogrammedSteps:
    """Preprogrammed single-leg steps extracted from FlyGym 1.x recordings."""

    legs = ["lf", "lm", "lh", "rf", "rm", "rh"]
    legacy_legs = ["LF", "LM", "LH", "RF", "RM", "RH"]
    dofs_per_leg = [
        ("thorax", "coxa", "pitch"),
        ("thorax", "coxa", "roll"),
        ("thorax", "coxa", "yaw"),
        ("coxa", "trochanterfemur", "pitch"),
        ("coxa", "trochanterfemur", "roll"),
        ("trochanterfemur", "tibia", "pitch"),
        ("tibia", "tarsus1", "pitch"),
    ]
    _legacy_dof_names = [
        "Coxa",
        "Coxa_roll",
        "Coxa_yaw",
        "Femur",
        "Femur_roll",
        "Tibia",
        "Tarsus1",
    ]

    def __init__(
        self,
        path: str | Path | None = None,
        neutral_pose_phases: tuple[float, ...] = (np.pi,) * 6,
    ) -> None:
        if path is None:
            path = (
                Path(str(files("flygym_demo.examples")))
                / "assets/behavior/single_steps_untethered.pkl"
            )
        with open(path, "rb") as f:
            single_steps_data = pickle.load(f)

        self._length = len(single_steps_data["joint_LFCoxa"])
        self._timestep = single_steps_data["meta"]["timestep"]
        self.duration = self._length * self._timestep

        phase_grid = np.linspace(0, 2 * np.pi, self._length)
        self._psi_funcs = {}
        for legacy_leg, leg in zip(self.legacy_legs, self.legs):
            joint_angles = np.array(
                [
                    single_steps_data[f"joint_{legacy_leg}{dof}"]
                    for dof in self._legacy_dof_names
                ]
            )
            self._psi_funcs[leg] = CubicSpline(
                phase_grid, joint_angles, axis=1, bc_type="periodic"
            )

        self.neutral_pos = {
            leg: self._psi_funcs[leg](theta_neutral)[:, np.newaxis]
            for leg, theta_neutral in zip(self.legs, neutral_pose_phases)
        }

        swing_stance_time_dict = single_steps_data["swing_stance_time"]
        self.swing_period = {}
        for legacy_leg, leg in zip(self.legacy_legs, self.legs):
            my_swing_period = np.array([0, swing_stance_time_dict["stance"][legacy_leg]])
            my_swing_period /= self.duration
            my_swing_period *= 2 * np.pi
            self.swing_period[leg] = my_swing_period

    def get_joint_angles(
        self, leg: str, phase: float | np.ndarray, magnitude: float | np.ndarray = 1
    ) -> np.ndarray:
        leg = leg.lower()
        phase_arr = np.asarray([phase]) if np.asarray(phase).shape == () else phase
        offset = self._psi_funcs[leg](phase_arr) - self.neutral_pos[leg]
        joint_angles = self.neutral_pos[leg] + magnitude * offset
        return joint_angles.squeeze()

    def get_joint_angles_for_order(
        self,
        phases: np.ndarray,
        magnitudes: np.ndarray | None,
        output_dof_order: list[JointDOF],
    ) -> np.ndarray:
        if magnitudes is None:
            magnitudes = np.ones(len(self.legs))

        by_leg = {
            leg: self.get_joint_angles(leg, phases[i], magnitudes[i])
            for i, leg in enumerate(self.legs)
        }
        targets = []
        for dof in output_dof_order:
            leg = dof.child.pos
            dof_key = (dof.parent.link, dof.child.link, dof.axis.value)
            targets.append(by_leg[leg][self.dofs_per_leg.index(dof_key)])
        return np.asarray(targets, dtype=np.float32)

    def get_adhesion_onoff(self, leg: str, phase: float) -> bool:
        swing_start, swing_end = self.swing_period[leg.lower()]
        return not (swing_start < phase % (2 * np.pi) < swing_end)

    def get_adhesion_for_phases(self, phases: np.ndarray) -> np.ndarray:
        return np.array(
            [self.get_adhesion_onoff(leg, phase) for leg, phase in zip(self.legs, phases)],
            dtype=np.float32,
        )

    @property
    def default_pose(self) -> np.ndarray:
        return np.concatenate([self.neutral_pos[leg] for leg in self.legs]).ravel()


def get_cpg_biases(gait: str) -> np.ndarray:
    gait = gait.lower()
    if gait == "tripod":
        phase_biases = np.array(
            [
                [0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0],
            ],
            dtype=np.float64,
        )
        return phase_biases * np.pi
    if gait == "tetrapod":
        phase_biases = np.array(
            [
                [0, 1, 2, 2, 0, 1],
                [2, 0, 1, 1, 2, 0],
                [1, 2, 0, 0, 1, 2],
                [1, 2, 0, 0, 1, 2],
                [0, 1, 2, 2, 0, 1],
                [2, 0, 1, 1, 2, 0],
            ],
            dtype=np.float64,
        )
        return phase_biases * 2 * np.pi / 3
    if gait == "wave":
        phase_biases = np.array(
            [
                [0, 1, 2, 3, 4, 5],
                [5, 0, 1, 2, 3, 4],
                [4, 5, 0, 1, 2, 3],
                [3, 4, 5, 0, 1, 2],
                [2, 3, 4, 5, 0, 1],
                [1, 2, 3, 4, 5, 0],
            ],
            dtype=np.float64,
        )
        return phase_biases * 2 * np.pi / 6
    raise ValueError(f"Unknown gait: {gait}")
