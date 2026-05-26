from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time
from typing import Callable, Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from flygym_demo.odor_obstacle_navigation.runners import make_task_runner


@dataclass(slots=True)
class PPOConfig:
    total_timesteps: int = 100_000
    num_envs: int = 4
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (128, 128)
    normalize_observations: bool = True
    seed: int = 0
    device: str = "cpu"
    env_device: str = "cpu"
    env_runner: str = "sync"
    mp_start_method: str = "spawn"


@dataclass(slots=True)
class RolloutStats:
    returns: list[float]
    lengths: list[int]
    successes: list[float]
    wall_seconds: float
    phase_wall_seconds: dict[str, float]
    worker_cpu_percent_sum: float | None
    worker_cpu_percent_mean: float | None
    worker_cpu_percent_max: float | None
    worker_count: int


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)
        batch_count = values.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_space: str,
        action_dim: int,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        action_max_norm: float | None = None,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.action_space = action_space
        self.actor = _make_mlp(obs_dim, hidden_sizes, action_dim)
        self.critic = _make_mlp(obs_dim, hidden_sizes, 1)
        if action_space == "continuous":
            if action_low is None or action_high is None:
                raise ValueError("continuous policies require action bounds.")
            self.log_std = nn.Parameter(torch.zeros(action_dim))
            self.register_buffer(
                "action_low",
                torch.as_tensor(action_low, dtype=torch.float32),
            )
            self.register_buffer(
                "action_scale",
                torch.as_tensor((action_high - action_low) / 2, dtype=torch.float32),
            )
            self.action_max_norm = action_max_norm
        elif action_space == "discrete":
            self.log_std = None
            self.action_max_norm = None
            self.register_buffer("action_low", torch.empty(0, dtype=torch.float32))
            self.register_buffer("action_scale", torch.empty(0, dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported action_space: {action_space}")

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.action_space == "continuous":
            mean = self.actor(obs)
            if self.log_std is None:
                raise RuntimeError("continuous policy is missing log_std.")
            std = self.log_std.exp().expand_as(mean)
            normal = Normal(mean, std)
            if action is None:
                raw_action = mean if deterministic else normal.rsample()
                squashed = torch.tanh(raw_action)
                action = self.action_low + self.action_scale * (squashed + 1.0)
                action = self._project_action_norm(action)
                squashed = self._unscale_action(action)
                raw_action = torch.atanh(squashed)
            else:
                squashed = self._unscale_action(action)
                raw_action = torch.atanh(squashed)

            log_prob = normal.log_prob(raw_action)
            log_prob -= torch.log(self.action_scale * (1.0 - squashed.pow(2)) + 1e-6)
            entropy = normal.entropy().sum(dim=-1)
            value = self.get_value(obs)
            return action, log_prob.sum(dim=-1), entropy, value

        logits = self.actor(obs)
        categorical = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else categorical.sample()
        else:
            action = action.long()
            if action.ndim > 1:
                action = action.squeeze(-1)
        log_prob = categorical.log_prob(action)
        entropy = categorical.entropy()
        value = self.get_value(obs)
        return action, log_prob, entropy, value

    def _unscale_action(self, action: torch.Tensor) -> torch.Tensor:
        squashed = (action - self.action_low) / self.action_scale - 1.0
        return torch.clamp(squashed, -0.999999, 0.999999)

    def _project_action_norm(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_max_norm is None:
            return action
        norms = torch.linalg.vector_norm(action, dim=-1, keepdim=True)
        scales = torch.clamp(self.action_max_norm / (norms + 1e-8), max=1.0)
        return action * scales


class PPOTrainer:
    def __init__(self, task_factory: Callable[[], Any], config: PPOConfig) -> None:
        self.task_factory = task_factory
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        self.device = torch.device(config.device)
        self.runner = make_task_runner(
            task_factory,
            num_envs=config.num_envs,
            env_device=config.env_device,
            env_runner=config.env_runner,
            mp_start_method=config.mp_start_method,
        )
        try:
            self.obs, _ = self.runner.reset(seed=config.seed)
            self.obs_rms = RunningMeanStd(self.obs.shape[1:])
            self.obs_rms.update(self.obs)
        except BaseException:
            self.runner.close()
            raise

        action_low = np.asarray(self.runner.action_low, dtype=np.float32)
        action_high = np.asarray(self.runner.action_high, dtype=np.float32)
        self.action_space = str(getattr(self.runner, "action_space", "continuous"))
        self.action_max_norm = getattr(self.runner, "action_max_norm", None)
        if self.action_space == "continuous":
            self.action_shape = tuple(action_low.shape)
            self.action_dtype = np.float32
            policy_action_dim = action_low.shape[0]
            action_n = None
        elif self.action_space == "discrete":
            self.action_shape = ()
            self.action_dtype = np.int64
            action_n = getattr(self.runner, "action_n", None)
            if action_n is None:
                raise ValueError("discrete action space requires action_n.")
            policy_action_dim = int(action_n)
        else:
            raise ValueError(f"Unsupported action_space: {self.action_space}")
        self.action_n = action_n
        self.action_dim = policy_action_dim
        self.model = ActorCritic(
            obs_dim=self.obs.shape[1],
            action_space=self.action_space,
            action_dim=policy_action_dim,
            action_low=action_low,
            action_high=action_high,
            action_max_norm=self.action_max_norm,
            hidden_sizes=config.hidden_sizes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )
        self._cpu_sampler = _ParentCpuSampler()
        self._worker_cpu_sampler = _WorkerCpuSampler()
        self._worker_cpu_sampler.sample(self.worker_diagnostics())
        self.optimizer_step = 0

    def train(
        self,
        total_timesteps: int | None = None,
        *,
        metrics_callback: Callable[[dict[str, float]], None] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_interval_seconds: float | None = None,
        eval_interval: int | None = None,
        eval_episodes: int = 5,
        eval_seed: int | None = None,
    ) -> dict[str, float]:
        cfg = self.config
        total_timesteps = (
            cfg.total_timesteps if total_timesteps is None else total_timesteps
        )
        if eval_interval is not None and eval_interval <= 0:
            eval_interval = None
        if progress_interval_seconds is not None and progress_interval_seconds <= 0:
            progress_interval_seconds = None
        global_step = 0
        update_count = 0
        self._synchronize_device()
        train_started_at = time.perf_counter()
        next_eval_step = eval_interval
        episode_returns = np.zeros(cfg.num_envs, dtype=np.float32)
        episode_lengths = np.zeros(cfg.num_envs, dtype=np.int32)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        completed_successes: list[float] = []
        last_metrics: dict[str, float] = {}

        while global_step < total_timesteps:
            rollout, rollout_stats = self._collect_rollout(
                episode_returns,
                episode_lengths,
                completed_returns,
                completed_lengths,
                completed_successes,
                global_step_start=global_step,
                update_count=update_count + 1,
                progress_callback=progress_callback,
                progress_interval_seconds=progress_interval_seconds,
            )
            global_step += cfg.rollout_steps * cfg.num_envs
            update_count += 1
            _report_progress(
                progress_callback,
                {
                    "stage": "update_start",
                    "global_step": float(global_step),
                    "rollout_index": float(update_count),
                },
            )
            self._synchronize_device()
            update_started_at = time.perf_counter()
            update_metrics = self._update(
                rollout,
                metrics_callback=metrics_callback,
                rollout_stats=rollout_stats,
                global_step=global_step,
                update_count=update_count,
                train_started_at=train_started_at,
            )
            self._synchronize_device()
            update_wall_seconds = time.perf_counter() - update_started_at
            _report_progress(
                progress_callback,
                {
                    "stage": "update_end",
                    "global_step": float(global_step),
                    "rollout_index": float(update_count),
                    "update_wall_seconds": update_wall_seconds,
                    **update_metrics,
                },
            )
            metrics = self._make_train_metrics(
                update_metrics,
                rollout_stats,
                global_step=global_step,
                update_count=update_count,
                train_started_at=train_started_at,
            )
            if (
                eval_interval is not None
                and next_eval_step is not None
                and global_step >= next_eval_step
            ):
                _report_progress(
                    progress_callback,
                    {
                        "stage": "eval_start",
                        "global_step": float(global_step),
                        "rollout_index": float(update_count),
                        "eval_episodes": float(eval_episodes),
                    },
                )
                eval_started_at = time.perf_counter()
                metrics.update(
                    self._evaluate_metrics(
                        eval_episodes,
                        seed=None if eval_seed is None else eval_seed + update_count,
                    )
                )
                _report_progress(
                    progress_callback,
                    {
                        "stage": "eval_end",
                        "global_step": float(global_step),
                        "rollout_index": float(update_count),
                        "eval_wall_seconds": time.perf_counter() - eval_started_at,
                        "eval/success_rate": metrics["eval/success_rate"],
                        "eval/mean_reward": metrics["eval/mean_reward"],
                        "eval/mean_ep_length": metrics["eval/mean_ep_length"],
                    },
                )
                next_eval_step += eval_interval
            if metrics_callback is not None:
                metrics_callback(metrics)
            last_metrics = metrics

        self._synchronize_device()
        summary = {
            "global_step": float(global_step),
            "updates": float(update_count),
            "mean_episode_return": float(np.mean(completed_returns[-20:]))
            if completed_returns
            else 0.0,
            "mean_episode_length": float(np.mean(completed_lengths[-20:]))
            if completed_lengths
            else 0.0,
            "mean_episode_success": float(np.mean(completed_successes[-20:]))
            if completed_successes
            else 0.0,
        }
        summary.update(last_metrics)
        return summary

    def evaluate(
        self,
        num_episodes: int = 5,
        *,
        seed: int | None = None,
        deterministic: bool = True,
        render: bool = False,
    ) -> list[dict[str, float]]:
        results = []
        for episode_idx in range(num_episodes):
            task = self.task_factory()
            episode_seed = None if seed is None else seed + episode_idx
            obs, info = task.reset(seed=episode_seed)
            done = False
            total_reward = 0.0
            steps = 0
            while not done:
                action = self.act(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = task.step(action)
                if render:
                    task.render()
                total_reward += reward
                steps += 1
                done = terminated or truncated
            results.append(
                {
                    "return": float(total_reward),
                    "steps": float(steps),
                    "success": float(bool(info.get("success", False))),
                }
            )
            task.close()
        return results

    def act(self, obs: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        obs_batch = np.asarray(obs, dtype=np.float32)[np.newaxis, :]
        obs_tensor = self._obs_tensor(obs_batch)
        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(
                obs_tensor,
                deterministic=deterministic,
            )
        action_np = action.cpu().numpy()[0]
        if self.action_space == "discrete":
            return np.asarray(action_np, dtype=np.int64)
        return np.asarray(action_np, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimizer_step": self.optimizer_step,
                "obs_rms": self.obs_rms.state_dict(),
                "action_space": self.action_space,
                "action_shape": self.action_shape,
                "action_n": self.action_n,
                "action_max_norm": self.action_max_norm,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint_action_space = checkpoint.get("action_space", "continuous")
        if checkpoint_action_space != self.action_space:
            raise ValueError(
                "Checkpoint action space "
                f"{checkpoint_action_space!r} does not match trainer action space "
                f"{self.action_space!r}."
            )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.optimizer_step = int(checkpoint.get("optimizer_step", 0))
        self.obs_rms.load_state_dict(checkpoint["obs_rms"])

    def close(self) -> None:
        self.runner.close()

    def worker_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics = getattr(self.runner, "diagnostics", None)
        if diagnostics is None:
            return []
        return diagnostics()

    def diagnostics(self) -> dict[str, Any]:
        parent_affinity = None
        if hasattr(os, "sched_getaffinity"):
            parent_affinity = sorted(os.sched_getaffinity(0))
        return {
            "parent": {
                "pid": os.getpid(),
                "cpu_affinity": parent_affinity,
                "cpu_affinity_count": (
                    None if parent_affinity is None else len(parent_affinity)
                ),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
            },
            "runner": self.config.env_runner,
            "env_device": self.config.env_device,
            "requested_envs": self.config.num_envs,
            "workers": self.worker_diagnostics(),
        }

    def _collect_rollout(
        self,
        episode_returns: np.ndarray,
        episode_lengths: np.ndarray,
        completed_returns: list[float],
        completed_lengths: list[int],
        completed_successes: list[float],
        *,
        global_step_start: int,
        update_count: int,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        progress_interval_seconds: float | None,
    ) -> tuple[dict[str, torch.Tensor], RolloutStats]:
        cfg = self.config
        obs_buf = np.zeros(
            (cfg.rollout_steps, cfg.num_envs, self.obs.shape[1]), dtype=np.float32
        )
        action_buf = np.zeros(
            (cfg.rollout_steps, cfg.num_envs, *self.action_shape),
            dtype=self.action_dtype,
        )
        logprob_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
        reward_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
        done_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
        value_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
        rollout_returns: list[float] = []
        rollout_lengths: list[int] = []
        rollout_successes: list[float] = []
        phase_wall_seconds = {
            "obs_normalize": 0.0,
            "policy_forward": 0.0,
            "action_transfer": 0.0,
            "env_step": 0.0,
            "reset": 0.0,
            "bootstrap": 0.0,
            "advantage": 0.0,
            "tensorize": 0.0,
        }
        self._synchronize_device()
        rollout_started_at = time.perf_counter()
        next_progress_at = (
            rollout_started_at + progress_interval_seconds
            if progress_interval_seconds is not None
            else None
        )
        decision_steps, decision_interval = self._task_timing()
        _report_progress(
            progress_callback,
            {
                "stage": "rollout_start",
                "global_step": float(global_step_start),
                "rollout_index": float(update_count),
                "rollout_steps": float(cfg.rollout_steps),
                "num_envs": float(cfg.num_envs),
                "decision_steps": decision_steps,
                "decision_interval": decision_interval,
            },
        )

        for step_idx in range(cfg.rollout_steps):
            phase_started_at = time.perf_counter()
            self.obs_rms.update(self.obs)
            obs_tensor = self._obs_tensor(self.obs)
            phase_wall_seconds["obs_normalize"] += (
                time.perf_counter() - phase_started_at
            )
            phase_started_at = time.perf_counter()
            with torch.no_grad():
                action, logprob, _, value = self.model.get_action_and_value(obs_tensor)
            phase_wall_seconds["policy_forward"] += (
                time.perf_counter() - phase_started_at
            )

            phase_started_at = time.perf_counter()
            actions = self._action_numpy(action)
            obs_buf[step_idx] = self.obs
            action_buf[step_idx] = actions
            logprob_buf[step_idx] = logprob.cpu().numpy()
            value_buf[step_idx] = value.cpu().numpy()
            phase_wall_seconds["action_transfer"] += (
                time.perf_counter() - phase_started_at
            )

            phase_started_at = time.perf_counter()
            next_obs, rewards, terminated, truncated, infos = self.runner.step(actions)
            phase_wall_seconds["env_step"] += time.perf_counter() - phase_started_at
            done = np.logical_or(terminated, truncated)
            reward_buf[step_idx] = rewards
            done_buf[step_idx] = done.astype(np.float32)
            episode_returns += rewards
            episode_lengths += 1
            done_indices = np.flatnonzero(done)
            for env_idx in done_indices:
                episode_return = float(episode_returns[env_idx])
                episode_length = int(episode_lengths[env_idx])
                episode_success = float(bool(infos[env_idx].get("success", False)))
                completed_returns.append(episode_return)
                completed_lengths.append(episode_length)
                completed_successes.append(episode_success)
                rollout_returns.append(episode_return)
                rollout_lengths.append(episode_length)
                rollout_successes.append(episode_success)
                episode_returns[env_idx] = 0.0
                episode_lengths[env_idx] = 0
            if done_indices.size:
                phase_started_at = time.perf_counter()
                reset_obs, _ = self.runner.reset_at(done_indices)
                phase_wall_seconds["reset"] += time.perf_counter() - phase_started_at
                next_obs[done_indices] = reset_obs
            self.obs = next_obs
            now = time.perf_counter()
            if (
                next_progress_at is not None
                and now >= next_progress_at
                and step_idx + 1 < cfg.rollout_steps
            ):
                collected_timesteps = (step_idx + 1) * cfg.num_envs
                rollout_elapsed = now - rollout_started_at
                _report_progress(
                    progress_callback,
                    self._rollout_progress_metrics(
                        stage="rollout",
                        global_step=global_step_start + collected_timesteps,
                        update_count=update_count,
                        rollout_step=step_idx + 1,
                        rollout_elapsed=rollout_elapsed,
                        decision_steps=decision_steps,
                        decision_interval=decision_interval,
                        completed_episodes=len(rollout_returns),
                    ),
                )
                while next_progress_at <= now:
                    next_progress_at += progress_interval_seconds

        phase_started_at = time.perf_counter()
        with torch.no_grad():
            next_value = self.model.get_value(self._obs_tensor(self.obs)).cpu().numpy()
        phase_wall_seconds["bootstrap"] += time.perf_counter() - phase_started_at

        phase_started_at = time.perf_counter()
        advantages = np.zeros_like(reward_buf)
        last_gae = np.zeros(cfg.num_envs, dtype=np.float32)
        for step_idx in reversed(range(cfg.rollout_steps)):
            next_nonterminal = 1.0 - done_buf[step_idx]
            if step_idx == cfg.rollout_steps - 1:
                next_values = next_value
            else:
                next_values = value_buf[step_idx + 1]
            delta = (
                reward_buf[step_idx]
                + cfg.gamma * next_values * next_nonterminal
                - value_buf[step_idx]
            )
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            advantages[step_idx] = last_gae
        returns = advantages + value_buf
        phase_wall_seconds["advantage"] += time.perf_counter() - phase_started_at
        self._synchronize_device()
        rollout_wall_seconds = time.perf_counter() - rollout_started_at
        worker_diagnostics = self.worker_diagnostics()
        worker_cpu_summary = self._worker_cpu_sampler.sample(worker_diagnostics)
        progress_event = self._rollout_progress_metrics(
            stage="rollout_end",
            global_step=global_step_start + cfg.rollout_steps * cfg.num_envs,
            update_count=update_count,
            rollout_step=cfg.rollout_steps,
            rollout_elapsed=rollout_wall_seconds,
            decision_steps=decision_steps,
            decision_interval=decision_interval,
            completed_episodes=len(rollout_returns),
        )
        phase_started_at = time.perf_counter()
        rollout_tensors = {
            "obs": self._tensor(obs_buf.reshape(-1, self.obs.shape[1])),
            "actions": self._action_tensor(action_buf.reshape(-1, *self.action_shape)),
            "logprobs": self._tensor(logprob_buf.ravel()),
            "advantages": self._tensor(advantages.ravel()),
            "returns": self._tensor(returns.ravel()),
            "values": self._tensor(value_buf.ravel()),
        }
        phase_wall_seconds["tensorize"] += time.perf_counter() - phase_started_at
        progress_event.update(_phase_metrics(phase_wall_seconds))
        progress_event.update(worker_cpu_summary)
        _report_progress(
            progress_callback,
            progress_event,
        )

        return (
            rollout_tensors,
            RolloutStats(
                returns=rollout_returns,
                lengths=rollout_lengths,
                successes=rollout_successes,
                wall_seconds=rollout_wall_seconds,
                phase_wall_seconds=phase_wall_seconds,
                worker_cpu_percent_sum=worker_cpu_summary["worker_cpu_percent_sum"],
                worker_cpu_percent_mean=worker_cpu_summary["worker_cpu_percent_mean"],
                worker_cpu_percent_max=worker_cpu_summary["worker_cpu_percent_max"],
                worker_count=int(worker_cpu_summary["worker_count"]),
            ),
        )

    def _update(
        self,
        rollout: dict[str, torch.Tensor],
        *,
        metrics_callback: Callable[[dict[str, float]], None] | None,
        rollout_stats: RolloutStats,
        global_step: int,
        update_count: int,
        train_started_at: float,
    ) -> dict[str, float]:
        cfg = self.config
        batch_size = rollout["obs"].shape[0]
        batch_indices = np.arange(batch_size)
        advantages = rollout["advantages"]
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []

        for _ in range(cfg.update_epochs):
            self.rng.shuffle(batch_indices)
            for start in range(0, batch_size, cfg.minibatch_size):
                indices = batch_indices[start : start + cfg.minibatch_size]
                _, new_logprob, entropy, new_value = self.model.get_action_and_value(
                    rollout["obs"][indices],
                    rollout["actions"][indices],
                )
                log_ratio = new_logprob - rollout["logprobs"][indices]
                ratio = log_ratio.exp()
                minibatch_advantages = advantages[indices]
                policy_loss_1 = -minibatch_advantages * ratio
                policy_loss_2 = -minibatch_advantages * torch.clamp(
                    ratio,
                    1.0 - cfg.clip_coef,
                    1.0 + cfg.clip_coef,
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                value_loss = (
                    0.5 * (new_value - rollout["returns"][indices]).pow(2).mean()
                )
                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    + cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()
                    step_metrics = self._optimizer_step_metrics(
                        loss=loss,
                        policy_loss=policy_loss,
                        value_loss=value_loss,
                        entropy_loss=entropy_loss,
                        approx_kl=approx_kl,
                        clip_fraction=clip_fraction,
                        values=new_value,
                        returns=rollout["returns"][indices],
                    )
                    losses.append(step_metrics["train/loss"])
                    policy_losses.append(step_metrics["train/policy_gradient_loss"])
                    value_losses.append(step_metrics["train/value_loss"])
                    entropy_losses.append(step_metrics["train/entropy_loss"])
                    approx_kls.append(step_metrics["train/approx_kl"])
                    clip_fractions.append(step_metrics["train/clip_fraction"])
                self.optimizer_step += 1
                if metrics_callback is not None:
                    step_metrics["train/optimizer_step"] = float(self.optimizer_step)
                    metrics_callback(
                        self._make_train_metrics(
                            step_metrics,
                            rollout_stats,
                            global_step=global_step,
                            update_count=update_count,
                            train_started_at=train_started_at,
                        )
                    )

        return {
            "train/loss": float(np.mean(losses)),
            "train/policy_gradient_loss": float(np.mean(policy_losses)),
            "train/value_loss": float(np.mean(value_losses)),
            "train/entropy_loss": float(np.mean(entropy_losses)),
            "train/approx_kl": float(np.mean(approx_kls)),
            "train/clip_fraction": float(np.mean(clip_fractions)),
            "train/clip_range": float(cfg.clip_coef),
            "train/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "train/std": self._policy_std_metric(),
            "train/optimizer_step": float(self.optimizer_step),
            "train/explained_variance": _explained_variance(
                rollout["values"],
                rollout["returns"],
            ),
        }

    def _optimizer_step_metrics(
        self,
        *,
        loss: torch.Tensor,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        approx_kl: torch.Tensor,
        clip_fraction: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        return {
            "train/loss": float(loss.item()),
            "train/policy_gradient_loss": float(policy_loss.item()),
            "train/value_loss": float(value_loss.item()),
            "train/entropy_loss": float(entropy_loss.item()),
            "train/approx_kl": float(approx_kl.item()),
            "train/clip_fraction": float(clip_fraction.item()),
            "train/clip_range": float(self.config.clip_coef),
            "train/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "train/std": self._policy_std_metric(),
            "train/explained_variance": _explained_variance(values, returns),
        }

    def _make_train_metrics(
        self,
        update_metrics: dict[str, float],
        rollout_stats: RolloutStats,
        *,
        global_step: int,
        update_count: int,
        train_started_at: float,
    ) -> dict[str, float]:
        elapsed = time.perf_counter() - train_started_at
        rollout_timesteps = self.config.rollout_steps * self.config.num_envs
        decision_steps, decision_interval = self._task_timing()
        metrics = {
            **update_metrics,
            "time/fps": _rate(global_step, elapsed),
            "speed/wall_time_seconds": float(elapsed),
            "speed/total_timesteps_per_sec": _rate(global_step, elapsed),
            "speed/simulator_substeps_per_sec": _rate(
                rollout_timesteps * decision_steps,
                rollout_stats.wall_seconds,
            ),
            "speed/rollout_wall_seconds": float(rollout_stats.wall_seconds),
            "speed/simulated_seconds_per_sec": _rate(
                rollout_timesteps * decision_interval,
                rollout_stats.wall_seconds,
            ),
            "speed/rollout_timesteps_per_sec": _rate(
                rollout_timesteps,
                rollout_stats.wall_seconds,
            ),
            "speed/rollout_index": float(update_count),
            "speed/parent_cpu_percent": self._cpu_sampler.sample(),
            "speed/worker_cpu_percent_sum": _none_to_zero(
                rollout_stats.worker_cpu_percent_sum
            ),
            "speed/worker_cpu_percent_mean": _none_to_zero(
                rollout_stats.worker_cpu_percent_mean
            ),
            "speed/worker_cpu_percent_max": _none_to_zero(
                rollout_stats.worker_cpu_percent_max
            ),
            "speed/worker_count": float(rollout_stats.worker_count),
            "global_step": float(global_step),
        }
        metrics.update(
            {
                f"speed/phase_{phase}_wall_seconds": seconds
                for phase, seconds in rollout_stats.phase_wall_seconds.items()
            }
        )
        if rollout_stats.returns:
            metrics.update(
                {
                    "rollout/success_rate": float(
                        100.0 * np.mean(rollout_stats.successes)
                    ),
                    "rollout/ep_rew_mean": float(np.mean(rollout_stats.returns)),
                    "rollout/ep_len_mean": float(np.mean(rollout_stats.lengths)),
                }
            )
        return metrics

    def _evaluate_metrics(
        self,
        num_episodes: int,
        *,
        seed: int | None = None,
    ) -> dict[str, float]:
        results = self.evaluate(num_episodes=num_episodes, seed=seed)
        return evaluation_metrics(results)

    def _task_timing(self) -> tuple[float, float]:
        return self.runner.decision_steps, self.runner.decision_interval

    def _rollout_progress_metrics(
        self,
        *,
        stage: str,
        global_step: int,
        update_count: int,
        rollout_step: int,
        rollout_elapsed: float,
        decision_steps: float,
        decision_interval: float,
        completed_episodes: int,
    ) -> dict[str, Any]:
        rollout_timesteps = rollout_step * self.config.num_envs
        return {
            "stage": stage,
            "global_step": float(global_step),
            "rollout_index": float(update_count),
            "rollout_step": float(rollout_step),
            "rollout_steps": float(self.config.rollout_steps),
            "rollout_timesteps": float(rollout_timesteps),
            "rollout_wall_seconds": float(rollout_elapsed),
            "rollout_timesteps_per_sec": _rate(rollout_timesteps, rollout_elapsed),
            "simulator_substeps_per_sec": _rate(
                rollout_timesteps * decision_steps,
                rollout_elapsed,
            ),
            "simulated_seconds_per_sec": _rate(
                rollout_timesteps * decision_interval,
                rollout_elapsed,
            ),
            "episodes_completed": float(completed_episodes),
        }

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        obs = np.asarray(obs, dtype=np.float32)
        if self.config.normalize_observations:
            obs = self.obs_rms.normalize(obs)
        return self._tensor(obs)

    def _action_numpy(self, action: torch.Tensor) -> np.ndarray:
        if self.action_space == "discrete":
            return action.cpu().numpy().astype(np.int64)
        return action.cpu().numpy().astype(np.float32)

    def _action_tensor(self, action: np.ndarray) -> torch.Tensor:
        if self.action_space == "discrete":
            return torch.as_tensor(action, dtype=torch.long, device=self.device)
        return self._tensor(action)

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _policy_std_metric(self) -> float:
        if self.action_space != "continuous":
            return 0.0
        if self.model.log_std is None:
            return 0.0
        return float(self.model.log_std.exp().mean().item())

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def _make_mlp(
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.extend((nn.Linear(prev_dim, hidden_size), nn.Tanh()))
        prev_dim = hidden_size
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def evaluation_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    returns = np.asarray([result["return"] for result in results], dtype=np.float32)
    lengths = np.asarray([result["steps"] for result in results], dtype=np.float32)
    successes = np.asarray([result["success"] for result in results], dtype=np.float32)
    return {
        "eval/success_rate": float(100.0 * successes.mean()),
        "eval/mean_reward": float(returns.mean()),
        "eval/mean_ep_length": float(lengths.mean()),
    }


def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    values_np = values.detach().cpu().numpy()
    returns_np = returns.detach().cpu().numpy()
    returns_var = np.var(returns_np)
    if returns_var == 0:
        return 0.0
    return float(1.0 - np.var(returns_np - values_np) / returns_var)


def _rate(numerator: float, elapsed: float) -> float:
    if elapsed <= 0:
        return 0.0
    return float(numerator / elapsed)


def _report_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    if callback is not None:
        callback(event)


def _phase_metrics(phase_wall_seconds: dict[str, float]) -> dict[str, float]:
    return {
        f"phase_{phase}_wall_seconds": float(seconds)
        for phase, seconds in phase_wall_seconds.items()
    }


def _none_to_zero(value: float | None) -> float:
    return 0.0 if value is None else float(value)


class _ParentCpuSampler:
    def __init__(self) -> None:
        self._pid = os.getpid()
        self._ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        self._last_cpu_seconds = self._read_cpu_seconds()
        self._last_wall_seconds = time.perf_counter()

    def sample(self) -> float:
        cpu_seconds = self._read_cpu_seconds()
        wall_seconds = time.perf_counter()
        cpu_delta = cpu_seconds - self._last_cpu_seconds
        wall_delta = wall_seconds - self._last_wall_seconds
        self._last_cpu_seconds = cpu_seconds
        self._last_wall_seconds = wall_seconds
        return _rate(cpu_delta * 100.0, wall_delta)

    def _read_cpu_seconds(self) -> float:
        stat = Path(f"/proc/{self._pid}/stat").read_text(encoding="utf-8")
        fields = stat.rsplit(")", maxsplit=1)[1].split()
        return (float(fields[11]) + float(fields[12])) / self._ticks_per_second


class _WorkerCpuSampler:
    def __init__(self) -> None:
        self._last_cpu_seconds_by_pid: dict[int, float] = {}
        self._last_wall_seconds = time.perf_counter()

    def sample(self, diagnostics: list[dict[str, Any]]) -> dict[str, float | None]:
        wall_seconds = time.perf_counter()
        wall_delta = wall_seconds - self._last_wall_seconds
        worker_cpu_percent: list[float] = []
        current_cpu_seconds_by_pid: dict[int, float] = {}
        for diagnostic in diagnostics:
            pid = diagnostic.get("pid")
            cpu_seconds = diagnostic.get("cpu_seconds")
            if pid is None or cpu_seconds is None:
                continue
            pid = int(pid)
            cpu_seconds = float(cpu_seconds)
            current_cpu_seconds_by_pid[pid] = cpu_seconds
            previous_cpu_seconds = self._last_cpu_seconds_by_pid.get(pid)
            if previous_cpu_seconds is not None:
                worker_cpu_percent.append(
                    _rate((cpu_seconds - previous_cpu_seconds) * 100.0, wall_delta)
                )
        self._last_cpu_seconds_by_pid = current_cpu_seconds_by_pid
        self._last_wall_seconds = wall_seconds
        if not worker_cpu_percent:
            return {
                "worker_cpu_percent_sum": None,
                "worker_cpu_percent_mean": None,
                "worker_cpu_percent_max": None,
                "worker_count": float(len(diagnostics)),
            }
        return {
            "worker_cpu_percent_sum": float(np.sum(worker_cpu_percent)),
            "worker_cpu_percent_mean": float(np.mean(worker_cpu_percent)),
            "worker_cpu_percent_max": float(np.max(worker_cpu_percent)),
            "worker_count": float(len(diagnostics)),
        }
