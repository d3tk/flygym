from __future__ import annotations

import numpy as np
import networkx as nx

from flygym import Simulation
from flygym.compose import ActuatorType
from flygym_demo.examples.common import ControllerOutput

from .steps import PreprogrammedSteps


def filter_edges(graph, rule, src_node=None):
    return [
        (src, tgt)
        for src, tgt, rule_type in graph.edges(data="rule")
        if (rule_type == rule) and (src_node is None or src == src_node)
    ]


def construct_rules_graph():
    edges = {
        "rule1": {"lm": ["lf"], "lh": ["lm"], "rm": ["rf"], "rh": ["rm"]},
        "rule2": {
            "lf": ["rf"],
            "lm": ["rm", "lf"],
            "lh": ["rh", "lm"],
            "rf": ["lf"],
            "rm": ["lm", "rf"],
            "rh": ["lh", "rm"],
        },
        "rule3": {
            "lf": ["rf", "lm"],
            "lm": ["rm", "lh"],
            "lh": ["rh"],
            "rf": ["lf", "rm"],
            "rm": ["lm", "rh"],
            "rh": ["lh"],
        },
    }
    graph = nx.MultiDiGraph()
    for rule_type, d in edges.items():
        for src, targets in d.items():
            for tgt in targets:
                if rule_type == "rule1":
                    rule_type_detailed = rule_type
                else:
                    side = "ipsi" if src[0] == tgt[0] else "contra"
                    rule_type_detailed = f"{rule_type}_{side}"
                graph.add_edge(src, tgt, rule=rule_type_detailed)
    return graph


class RuleBasedController:
    legs = ["lf", "lm", "lh", "rf", "rm", "rh"]

    def __init__(
        self,
        timestep: float,
        rules_graph: nx.MultiDiGraph | None = None,
        weights: dict[str, float] | None = None,
        preprogrammed_steps: PreprogrammedSteps | None = None,
        margin: float = 0.001,
        seed: int = 0,
    ) -> None:
        self.timestep = timestep
        self.rules_graph = rules_graph or construct_rules_graph()
        self.weights = weights or {
            "rule1": -10,
            "rule2_ipsi": 2.5,
            "rule2_contra": 1.0,
            "rule3_ipsi": 3.0,
            "rule3_contra": 2.0,
        }
        self.preprogrammed_steps = preprogrammed_steps or PreprogrammedSteps()
        self.margin = margin
        self.random_state = np.random.RandomState(seed)
        self._phase_inc_per_step = 2 * np.pi * (
            timestep / self.preprogrammed_steps.duration
        )
        self._leg2id = {leg: i for i, leg in enumerate(self.legs)}
        self.reset()

    def reset(self) -> None:
        self.curr_step = 0
        self.rule1_scores = np.zeros(6)
        self.rule2_scores = np.zeros(6)
        self.rule3_scores = np.zeros(6)
        self.leg_phases = np.zeros(6)
        self.mask_is_stepping = np.zeros(6, dtype=bool)

    @property
    def combined_scores(self):
        return self.rule1_scores + self.rule2_scores + self.rule3_scores

    def _get_eligible_legs(self):
        score_thr = self.combined_scores.max()
        score_thr = max(0, score_thr - np.abs(score_thr) * self.margin)
        mask = (
            (self.combined_scores >= score_thr)
            & (self.combined_scores > 0)
            & ~self.mask_is_stepping
        )
        return np.where(mask)[0]

    def _select_stepping_leg(self):
        eligible_legs = self._get_eligible_legs()
        if len(eligible_legs) == 0:
            return None
        return self.random_state.choice(eligible_legs)

    def _apply_rule1(self):
        self.rule1_scores[:] = 0
        for i, leg in enumerate(self.legs):
            swing_end = self.preprogrammed_steps.swing_period[leg][1]
            is_swinging = 0 < self.leg_phases[i] < swing_end
            for _, tgt in filter_edges(self.rules_graph, "rule1", src_node=leg):
                self.rule1_scores[self._leg2id[tgt]] = (
                    self.weights["rule1"] if is_swinging else 0
                )

    def _get_stance_progress_ratio(self, leg):
        _, swing_end = self.preprogrammed_steps.swing_period[leg]
        stance_duration = 2 * np.pi - swing_end
        curr = self.leg_phases[self._leg2id[leg]] - swing_end
        return max(0, curr) / stance_duration

    def _apply_rule2(self):
        self.rule2_scores[:] = 0
        for leg in self.legs:
            stance_progress_ratio = self._get_stance_progress_ratio(leg)
            if stance_progress_ratio == 0:
                continue
            for side in ["ipsi", "contra"]:
                for _, tgt in filter_edges(self.rules_graph, f"rule2_{side}", leg):
                    self.rule2_scores[self._leg2id[tgt]] += self.weights[
                        f"rule2_{side}"
                    ] * (1 - stance_progress_ratio)

    def _apply_rule3(self):
        self.rule3_scores[:] = 0
        for leg in self.legs:
            stance_progress_ratio = self._get_stance_progress_ratio(leg)
            if stance_progress_ratio == 0:
                continue
            for side in ["ipsi", "contra"]:
                for _, tgt in filter_edges(self.rules_graph, f"rule3_{side}", leg):
                    self.rule3_scores[self._leg2id[tgt]] += self.weights[
                        f"rule3_{side}"
                    ] * stance_progress_ratio

    def _update_rules(self):
        if self.curr_step == 0:
            stepping_leg_id = self.random_state.choice([0, 1, 3, 4])
        else:
            stepping_leg_id = self._select_stepping_leg()
        if stepping_leg_id is not None:
            self.mask_is_stepping[stepping_leg_id] = True
        self.leg_phases[self.mask_is_stepping] += self._phase_inc_per_step
        completed = self.leg_phases >= 2 * np.pi
        self.leg_phases[completed] = 0
        self.mask_is_stepping[completed] = False
        self._apply_rule1()
        self._apply_rule2()
        self._apply_rule3()
        self.curr_step += 1

    def step(self, sim: Simulation, drive=None) -> ControllerOutput:
        fly_name = next(iter(sim.world.fly_lookup))
        fly = sim.world.fly_lookup[fly_name]
        self._update_rules()
        dof_order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        targets = self.preprogrammed_steps.get_joint_angles_for_order(
            self.leg_phases, None, dof_order
        )
        adhesion = self.preprogrammed_steps.get_adhesion_for_phases(self.leg_phases)
        return ControllerOutput(
            targets,
            adhesion,
            {
                "leg_phases": self.leg_phases.copy(),
                "combined_scores": self.combined_scores.copy(),
                "stepping": self.mask_is_stepping.copy(),
            },
        )
