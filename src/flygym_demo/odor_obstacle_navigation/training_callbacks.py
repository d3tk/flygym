from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


@dataclass(slots=True)
class ResolvedEnvTiming:
    timestep: float
    decision_steps: int
    decision_dt: float
    max_decision_steps: int
    episode_time_seconds: float


def resolved_env_timing_from_config(config) -> ResolvedEnvTiming:
    return ResolvedEnvTiming(
        timestep=float(config.timestep),
        decision_steps=int(config.decision_steps),
        decision_dt=float(config.decision_interval),
        max_decision_steps=int(config.max_decision_steps),
        episode_time_seconds=float(config.episode_time),
    )


def vecnormalize_path_for_checkpoint(checkpoint_path: Path) -> Path:
    stem = checkpoint_path.stem
    if not stem.endswith("_steps"):
        raise ValueError(f"Unexpected checkpoint stem: {stem}")
    prefix, step_count = stem[: -len("_steps")].rsplit("_", 1)
    return checkpoint_path.with_name(f"{prefix}_vecnormalize_{step_count}_steps.pkl")


ACTION_METRIC_LABELS = ("left", "right", "forward", "groom")
SHARED_ROLLOUT_DIAGNOSTIC_DEFAULTS = {
    "reward/progress": 0.0,
    "reward/time_penalty": 0.0,
    "reward/success_bonus": 0.0,
    "reward/extrinsic": 0.0,
    "reward/total": 0.0,
    "options/left_fraction": 0.0,
    "options/right_fraction": 0.0,
    "options/forward_fraction": 0.0,
    "options/groom_fraction": 0.0,
    "options/entropy": 0.0,
    "options/switch_rate": 0.0,
    "behavior/forward_speed": 0.0,
    "behavior/yaw_rate": 0.0,
    "behavior/alive_fraction": 0.0,
    "behavior/low_height_failures": 0.0,
    "behavior/out_of_bounds_failures": 0.0,
    "behavior/physics_failures": 0.0,
}
INTRINSIC_ROLLOUT_DIAGNOSTIC_DEFAULTS = {
    "reward/intrinsic": 0.0,
    "curiosity/loss": 0.0,
}
RND_ROLLOUT_DIAGNOSTIC_DEFAULTS = {"curiosity/rnd_error": 0.0}
SHAPING_ROLLOUT_DIAGNOSTIC_DEFAULTS = {
    "reward/forward": 0.0,
    "reward/switch_cost": 0.0,
}
ROLLOUT_DIAGNOSTIC_DEFAULTS = {
    **SHARED_ROLLOUT_DIAGNOSTIC_DEFAULTS,
    **INTRINSIC_ROLLOUT_DIAGNOSTIC_DEFAULTS,
    **RND_ROLLOUT_DIAGNOSTIC_DEFAULTS,
    **SHAPING_ROLLOUT_DIAGNOSTIC_DEFAULTS,
}


def _finite_mean(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return 0.0
    return float(np.mean(finite_values))


def _extrinsic_reward_value(terms: dict[str, float]) -> float:
    if "extrinsic_total" in terms:
        return float(terms["extrinsic_total"])
    if "extrinsic_weighted" in terms:
        return float(terms["extrinsic_weighted"])
    return float(terms.get("total", 0.0))


def _continuous_option_label(info: dict) -> str:
    low_level_action = info.get("low_level_action", {})
    if not isinstance(low_level_action, dict):
        return ""
    if bool(low_level_action.get("groom", False)):
        return "groom"
    try:
        left_force = float(low_level_action["left_force"])
        right_force = float(low_level_action["right_force"])
    except (KeyError, TypeError, ValueError):
        return ""
    if abs(left_force - right_force) <= 1e-6:
        return "forward"
    if right_force > left_force:
        return "left"
    return "right"


def _option_label_from_info(info: dict) -> str:
    policy_action_label = str(info.get("policy_action_label", ""))
    if policy_action_label in ACTION_METRIC_LABELS:
        return policy_action_label
    if policy_action_label == "low_level":
        return _continuous_option_label(info)
    return ""


def _rollout_default_metrics(
    infos: list[dict],
    *,
    reward_mode: str,
    intrinsic_method: str | None,
    include_reward_shaping: bool | None,
) -> dict[str, float]:
    defaults = dict(SHARED_ROLLOUT_DIAGNOSTIC_DEFAULTS)
    has_intrinsic_reward = reward_mode in {"intrinsic", "mixed"} or any(
        "intrinsic" in dict(info.get("reward_terms", {})) for info in infos
    )
    if has_intrinsic_reward:
        defaults.update(INTRINSIC_ROLLOUT_DIAGNOSTIC_DEFAULTS)
    if has_intrinsic_reward and intrinsic_method == "rnd":
        defaults.update(RND_ROLLOUT_DIAGNOSTIC_DEFAULTS)
    has_shaping = include_reward_shaping
    if has_shaping is None:
        has_shaping = any(
            any(
                key in dict(info.get("reward_terms", {}))
                for key in ("forward", "switch_cost")
            )
            for info in infos
        )
    if has_shaping:
        defaults.update(SHAPING_ROLLOUT_DIAGNOSTIC_DEFAULTS)
    return defaults


def aggregate_rollout_diagnostics(
    infos: list[dict],
    *,
    reward_mode: str = "extrinsic",
    intrinsic_method: str | None = None,
    include_reward_shaping: bool | None = None,
) -> dict[str, float]:
    defaults = _rollout_default_metrics(
        infos,
        reward_mode=reward_mode,
        intrinsic_method=intrinsic_method,
        include_reward_shaping=include_reward_shaping,
    )
    if not infos:
        return defaults

    reward_terms = [dict(info.get("reward_terms", {})) for info in infos]
    labels = [_option_label_from_info(info) for info in infos]
    label_counts = Counter(label for label in labels if label)
    total_labels = max(sum(label_counts.values()), 1)
    action_probs = {
        label: label_counts.get(label, 0) / total_labels
        for label in ACTION_METRIC_LABELS
    }
    entropy = -sum(
        prob * float(np.log(prob)) for prob in action_probs.values() if prob > 0.0
    )

    failure_reasons = [
        info.get("failure_reason")
        for info in infos
        if info.get("failure_reason") is not None
    ]
    failure_counts = Counter(failure_reasons)
    out_of_bounds_failures = failure_counts.get("x_out_of_bounds", 0) + failure_counts.get(
        "y_out_of_bounds", 0
    )

    metrics = {
        "reward/progress": _finite_mean(
            [float(terms.get("progress", 0.0)) for terms in reward_terms]
        ),
        "reward/time_penalty": _finite_mean(
            [float(terms.get("time", 0.0)) for terms in reward_terms]
        ),
        "reward/success_bonus": _finite_mean(
            [float(terms.get("success", 0.0)) for terms in reward_terms]
        ),
        "reward/intrinsic": _finite_mean(
            [float(terms.get("intrinsic", 0.0)) for terms in reward_terms]
        ),
        "reward/extrinsic": _finite_mean(
            [_extrinsic_reward_value(terms) for terms in reward_terms]
        ),
        "reward/forward": _finite_mean(
            [float(terms.get("forward", 0.0)) for terms in reward_terms]
        ),
        "reward/switch_cost": _finite_mean(
            [float(terms.get("switch_cost", 0.0)) for terms in reward_terms]
        ),
        "reward/total": _finite_mean(
            [float(terms.get("total", 0.0)) for terms in reward_terms]
        ),
        "curiosity/loss": _finite_mean(
            [float(info.get("curiosity_loss", 0.0)) for info in infos]
        ),
        "curiosity/rnd_error": _finite_mean(
            [float(info.get("rnd_error", 0.0)) for info in infos]
        ),
        "options/left_fraction": float(action_probs["left"]),
        "options/right_fraction": float(action_probs["right"]),
        "options/forward_fraction": float(action_probs["forward"]),
        "options/groom_fraction": float(action_probs["groom"]),
        "options/entropy": float(entropy),
        "options/switch_rate": _finite_mean(
            [float(bool(info.get("switched_action", False))) for info in infos]
        ),
        "behavior/forward_speed": _finite_mean(
            [float(info.get("forward_speed", 0.0)) for info in infos]
        ),
        "behavior/yaw_rate": _finite_mean(
            [float(info.get("yaw_rate", 0.0)) for info in infos]
        ),
        "behavior/alive_fraction": _finite_mean(
            [float(info.get("failure_reason") is None) for info in infos]
        ),
        "behavior/low_height_failures": float(failure_counts.get("low_height", 0)),
        "behavior/out_of_bounds_failures": float(out_of_bounds_failures),
        "behavior/physics_failures": float(failure_counts.get("physics_error", 0)),
    }
    return {key: metrics.get(key, value) for key, value in defaults.items()}


class TrainingSpeedCallback(BaseCallback):
    def __init__(
        self,
        csv_path: Path,
        timing: ResolvedEnvTiming,
        *,
        log_to_wandb: bool = False,
        resume: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.csv_path = csv_path
        self.timing = timing
        self.log_to_wandb = log_to_wandb
        self.resume = resume
        self._train_start_wall = 0.0
        self._rollout_start_wall = 0.0
        self._rollout_start_timesteps = 0
        self._rollout_index = 0
        self._process = None
        self._fieldnames = [
            "rollout_index",
            "wall_time_seconds",
            "total_timesteps",
            "rollout_timesteps",
            "rollout_wall_seconds",
            "total_timesteps_per_sec",
            "rollout_timesteps_per_sec",
            "simulated_seconds_per_sec",
            "simulator_substeps_per_sec",
            "parent_cpu_percent",
        ]

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.resume and self.csv_path.is_file():
            with self.csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                self._rollout_index = int(rows[-1]["rollout_index"]) + 1
        else:
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
                writer.writeheader()

        self._train_start_wall = time.perf_counter()
        if psutil is not None:
            self._process = psutil.Process(os.getpid())
            self._process.cpu_percent(interval=None)

    def _on_rollout_start(self) -> None:
        self._rollout_start_wall = time.perf_counter()
        self._rollout_start_timesteps = int(self.num_timesteps)

    def _on_rollout_end(self) -> None:
        now = time.perf_counter()
        total_wall = max(now - self._train_start_wall, 1e-12)
        rollout_wall = max(now - self._rollout_start_wall, 1e-12)
        rollout_timesteps = int(self.num_timesteps - self._rollout_start_timesteps)
        total_steps_per_sec = float(self.num_timesteps) / total_wall
        rollout_steps_per_sec = float(rollout_timesteps) / rollout_wall
        parent_cpu_percent = ""
        if self._process is not None:
            parent_cpu_percent = self._process.cpu_percent(interval=None)

        row = {
            "rollout_index": self._rollout_index,
            "wall_time_seconds": total_wall,
            "total_timesteps": int(self.num_timesteps),
            "rollout_timesteps": rollout_timesteps,
            "rollout_wall_seconds": rollout_wall,
            "total_timesteps_per_sec": total_steps_per_sec,
            "rollout_timesteps_per_sec": rollout_steps_per_sec,
            "simulated_seconds_per_sec": total_steps_per_sec * self.timing.decision_dt,
            "simulator_substeps_per_sec": (
                total_steps_per_sec * self.timing.decision_steps
            ),
            "parent_cpu_percent": parent_cpu_percent,
        }

        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writerow(row)

        if self.log_to_wandb and wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "speed/rollout_index": row["rollout_index"],
                    "speed/wall_time_seconds": row["wall_time_seconds"],
                    "speed/rollout_wall_seconds": row["rollout_wall_seconds"],
                    "speed/total_timesteps_per_sec": row["total_timesteps_per_sec"],
                    "speed/rollout_timesteps_per_sec": row[
                        "rollout_timesteps_per_sec"
                    ],
                    "speed/simulated_seconds_per_sec": row[
                        "simulated_seconds_per_sec"
                    ],
                    "speed/simulator_substeps_per_sec": row[
                        "simulator_substeps_per_sec"
                    ],
                    "speed/parent_cpu_percent": row["parent_cpu_percent"],
                }
            )
        self._rollout_index += 1

    def _on_step(self) -> bool:
        return True


class FlyGymDiagnosticsCallback(BaseCallback):
    """Aggregate env-info diagnostics once per rollout."""

    def __init__(
        self,
        *,
        reward_mode: str = "extrinsic",
        intrinsic_method: str | None = None,
        include_reward_shaping: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.reward_mode = reward_mode
        self.intrinsic_method = intrinsic_method
        self.include_reward_shaping = include_reward_shaping
        self._infos: list[dict] = []

    def _on_rollout_start(self) -> None:
        self._infos = []

    def _on_step(self) -> bool:
        self._infos.extend(dict(info) for info in self.locals.get("infos", []))
        return True

    def _on_rollout_end(self) -> None:
        for key, value in aggregate_rollout_diagnostics(
            self._infos,
            reward_mode=self.reward_mode,
            intrinsic_method=self.intrinsic_method,
            include_reward_shaping=self.include_reward_shaping,
        ).items():
            self.logger.record(key, value)


class VecNormalizeFiniteCheckCallback(BaseCallback):
    """Fail early if normalized policy inputs or VecNormalize stats go non-finite."""

    def __init__(self, *, check_freq: int = 1, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        if check_freq <= 0:
            raise ValueError("check_freq must be positive.")
        self.check_freq = int(check_freq)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        issues: list[str] = []
        self._append_nonfinite_issue(
            issues,
            "rollout new_obs",
            self.locals.get("new_obs"),
        )
        last_obs = getattr(self.model, "_last_obs", None)
        self._append_nonfinite_issue(issues, "model._last_obs", last_obs)

        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            obs_rms = getattr(vec_env, "obs_rms", None)
            if obs_rms is not None:
                self._append_nonfinite_issue(
                    issues,
                    "VecNormalize.obs_rms.mean",
                    obs_rms.mean,
                )
                self._append_nonfinite_issue(
                    issues,
                    "VecNormalize.obs_rms.var",
                    obs_rms.var,
                )

        if issues:
            raise ValueError("Non-finite PPO normalization state: " + "; ".join(issues))
        return True

    @staticmethod
    def _append_nonfinite_issue(
        issues: list[str],
        name: str,
        values,
    ) -> None:
        if values is None:
            return
        array = np.asarray(values)
        count = int(array.size - np.count_nonzero(np.isfinite(array)))
        if count:
            issues.append(f"{name} has {count} non-finite values")


class NavigationEvalCallback(EvalCallback):
    """EvalCallback that logs eval success rate to W&B when enabled."""

    def __init__(self, *args, log_to_wandb: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log_to_wandb = log_to_wandb

    def _log_success_callback(self, locals_: dict, _globals: dict) -> None:
        if not locals_["done"]:
            return
        info = locals_["info"]
        maybe_is_success = info.get("is_success")
        if maybe_is_success is None and "success" in info:
            maybe_is_success = info["success"]
        if maybe_is_success is not None:
            self._is_success_buffer.append(bool(maybe_is_success))

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if (
            self.log_to_wandb
            and wandb is not None
            and wandb.run is not None
            and self.eval_freq > 0
            and self.n_calls % self.eval_freq == 0
            and len(self._is_success_buffer) > 0
        ):
            wandb.log(
                {"eval/success_rate": float(np.mean(self._is_success_buffer))},
                step=self.num_timesteps,
            )
        return continue_training


class VecNormalizeOnNewBestCallback(BaseCallback):
    """Persist VecNormalize stats next to SB3's best_model.zip (and run root)."""

    def __init__(self, run_dir: Path, best_model_dir: Path, *, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._run_dir_path = run_dir / "vecnormalize.pkl"
        self._best_dir_path = best_model_dir / "vecnormalize.pkl"

    def _on_step(self) -> bool:
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is None:
            return True
        vec_env.save(str(self._run_dir_path))
        self._best_dir_path.parent.mkdir(parents=True, exist_ok=True)
        vec_env.save(str(self._best_dir_path))
        return True
