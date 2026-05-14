from __future__ import annotations

from enum import Enum

import numpy as np


class WalkingState(Enum):
    FORWARD = "forward"
    TURN_LEFT = "left turn"
    TURN_RIGHT = "right turn"
    STOP = "stop"


class TurningObjective(Enum):
    UPWIND = "upwind"
    DOWNWIND = "downwind"


class PlumeNavigationController:
    """Encounter-driven plume navigation controller from Demir et al. 2020."""

    def __init__(
        self,
        dt: float,
        forward_dn_drive: tuple[float, float] = (1.0, 1.0),
        left_turn_dn_drive: tuple[float, float] = (-0.4, 1.2),
        right_turn_dn_drive: tuple[float, float] = (1.2, -0.4),
        stop_dn_drive: tuple[float, float] = (0.0, 0.0),
        turn_duration: float = 0.25,
        lambda_ws_0: float = 0.78,
        delta_lambda_ws: float = -0.8,
        tau_s: float = 0.2,
        alpha: float = 0.8,
        tau_freq_conv: float = 2,
        cumulative_evidence_window: float = 2.0,
        lambda_sw_0: float = 0.5,
        delta_lambda_sw: float = 1,
        tau_w: float = 0.52,
        lambda_turn: float = 1.33,
        random_seed: int = 0,
    ) -> None:
        self.dt = dt
        self.dn_drives = {
            WalkingState.FORWARD: np.array(forward_dn_drive),
            WalkingState.TURN_LEFT: np.array(left_turn_dn_drive),
            WalkingState.TURN_RIGHT: np.array(right_turn_dn_drive),
            WalkingState.STOP: np.array(stop_dn_drive),
        }

        self.lambda_ws_0 = lambda_ws_0
        self.delta_lambda_ws = delta_lambda_ws
        self.tau_s = tau_s

        self.curr_time = 0.0
        self.curr_state = WalkingState.FORWARD
        self.curr_state_start_time = 0.0
        self.last_encounter_time = -np.inf
        self.encounter_history = []

        self.cumulative_evidence_window = cumulative_evidence_window
        self.cumulative_evidence_len = int(cumulative_evidence_window / dt)
        self.lambda_sw_0 = lambda_sw_0
        self.delta_lambda_sw = delta_lambda_sw
        self.tau_w = tau_w
        self.encounter_weights = -np.arange(self.cumulative_evidence_len)[::-1] * dt

        self.turn_duration = turn_duration
        self.alpha = alpha
        self.tau_freq_conv = tau_freq_conv
        self.freq_kernel = np.exp(self.encounter_weights / tau_freq_conv)
        self.lambda_turn = lambda_turn

        self._turn_debug_str_buffer = ""
        self.random_state = np.random.RandomState(random_seed)

    def decide_state(
        self, encounter_flag: bool, fly_heading: np.ndarray
    ) -> tuple[WalkingState, np.ndarray, str]:
        self.encounter_history.append(encounter_flag)
        if encounter_flag:
            self.last_encounter_time = self.curr_time

        debug_str = ""

        if self.curr_state == WalkingState.FORWARD:
            p_nochange = np.exp(-self.lambda_turn * self.dt)
            if self.random_state.rand() > p_nochange:
                encounter_hist = np.array(
                    self.encounter_history[-self.cumulative_evidence_len :]
                )
                kernel = self.freq_kernel[-len(encounter_hist) :]
                w_freq = np.sum(kernel * encounter_hist) * self.dt
                w_freq *= self.exp_integral_norm_factor(
                    len(encounter_hist) * self.dt,
                    self.tau_freq_conv,
                )
                p_upwind = 1 / (1 + np.exp(-self.alpha * w_freq))
                if self.random_state.rand() < p_upwind:
                    turn_objective = TurningObjective.UPWIND
                    debug_str = (
                        f"Wfreq={w_freq:.2f}  "
                        f"P(upwind)={p_upwind:.2f}, turning UPWIND"
                    )
                else:
                    turn_objective = TurningObjective.DOWNWIND
                    debug_str = (
                        f"Wfreq={w_freq:.2f}  "
                        f"P(upwind)={p_upwind:.2f}, turning DOWNWIND"
                    )
                self._turn_debug_str_buffer = debug_str
                self.curr_state = self._turn_state(turn_objective, fly_heading)
                self.curr_state_start_time = self.curr_time

        if self.curr_state == WalkingState.FORWARD:
            lambda_ws = self.lambda_ws_0 + self.delta_lambda_ws * np.exp(
                -(self.curr_time - self.last_encounter_time) / self.tau_s
            )
            p_nochange = np.exp(-lambda_ws * self.dt)
            p_stop_1s = 1 - np.exp(-lambda_ws)
            debug_str = (
                f"lambda(w->s)={lambda_ws:.2f}  P(stop within 1s)={p_stop_1s:.2f}"
            )
            if self.random_state.rand() > p_nochange:
                self.curr_state = WalkingState.STOP
                self.curr_state_start_time = self.curr_time

        if self.curr_state in (WalkingState.TURN_LEFT, WalkingState.TURN_RIGHT):
            debug_str = self._turn_debug_str_buffer
            if self.curr_time - self.curr_state_start_time > self.turn_duration:
                self.curr_state = WalkingState.FORWARD
                self.curr_state_start_time = self.curr_time

        if self.curr_state == WalkingState.STOP:
            encounter_hist = np.array(
                self.encounter_history[-self.cumulative_evidence_len :]
            )
            time_diff = self.encounter_weights[-len(encounter_hist) :]
            evidence = np.sum(np.exp(time_diff / self.tau_w) * encounter_hist) * self.dt
            evidence *= self.exp_integral_norm_factor(
                len(encounter_hist) * self.dt,
                self.tau_w,
            )
            lambda_sw = self.lambda_sw_0 + self.delta_lambda_sw * evidence
            p_nochange = np.exp(-lambda_sw * self.dt)
            p_walk_1s = 1 - np.exp(-lambda_sw)
            debug_str = (
                f"lambda(s->w)={lambda_sw:.2f}  P(walk within 1s)={p_walk_1s:.2f}"
            )
            if self.random_state.rand() > p_nochange:
                self.curr_state = WalkingState.FORWARD
                self.curr_state_start_time = self.curr_time

        self.curr_time += self.dt
        return self.curr_state, self.dn_drives[self.curr_state], debug_str

    def _turn_state(
        self, turn_objective: TurningObjective, fly_heading: np.ndarray
    ) -> WalkingState:
        upwind_is_left = fly_heading[1] >= 0
        if turn_objective == TurningObjective.UPWIND:
            return WalkingState.TURN_LEFT if upwind_is_left else WalkingState.TURN_RIGHT
        return WalkingState.TURN_RIGHT if upwind_is_left else WalkingState.TURN_LEFT

    @staticmethod
    def exp_integral_norm_factor(window: float, tau: float) -> float:
        if window <= 0:
            raise ValueError("window must be positive")
        return 1 / (1 - np.exp(-window / tau))
