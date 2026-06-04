from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, VecNormalize

from flygym_demo.odor_obstacle_navigation.runners import make_task_runner


def _finite_float32_array(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    elif hasattr(values, "numpy") and callable(values.numpy):
        values = values.numpy()
    array = np.asarray(values)
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(array)):
        array = np.nan_to_num(
            array,
            copy=True,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)
    return array


def _nonfinite_counts_by_env(values: Any, num_envs: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[0] != num_envs:
        raise ValueError(
            f"Expected first dimension {num_envs}, got shape {array.shape}."
        )
    flattened = array.reshape(num_envs, -1)
    return np.count_nonzero(~np.isfinite(flattened), axis=1).astype(np.int32)


def _mark_nonfinite_reset_infos(
    infos: list[dict[str, Any]],
    obs_nonfinite_counts: np.ndarray,
) -> None:
    for env_idx, count in enumerate(obs_nonfinite_counts):
        if count <= 0:
            continue
        infos[env_idx]["nonfinite_reset_observation_count"] = int(count)


def _mark_nonfinite_step_infos(
    infos: list[dict[str, Any]],
    env_indices: np.ndarray,
    obs_nonfinite_counts: np.ndarray,
    reward_nonfinite: np.ndarray,
) -> None:
    for env_idx in env_indices:
        info = infos[int(env_idx)]
        previous_reason = info.get("failure_reason")
        if previous_reason is not None and previous_reason != "nonfinite_env_output":
            info["previous_failure_reason"] = previous_reason
        info["failure_reason"] = "nonfinite_env_output"
        info["nonfinite_env_output"] = True
        info["nonfinite_observation_count"] = int(obs_nonfinite_counts[int(env_idx)])
        info["nonfinite_reward"] = bool(reward_nonfinite[int(env_idx)])


def resolve_minibatch_size(
    minibatch_size: int,
    *,
    rollout_steps: int,
    num_envs: int,
) -> int:
    """Pick a minibatch size that evenly divides ``rollout_steps * num_envs``."""
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive.")
    if rollout_steps <= 0 or num_envs <= 0:
        raise ValueError("rollout_steps and num_envs must be positive.")
    buffer_size = rollout_steps * num_envs
    if buffer_size % minibatch_size == 0:
        return minibatch_size
    for candidate in range(minibatch_size, 0, -1):
        if buffer_size % candidate == 0:
            return candidate
    raise ValueError(
        f"No minibatch_size <= {minibatch_size} divides rollout buffer "
        f"{buffer_size} ({rollout_steps} x {num_envs})."
    )


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
    seed: int = 0
    device: str = "cpu"
    env_device: str = "cpu"
    env_runner: str = "sync"
    mp_start_method: str = "spawn"

    def sb3_kwargs(self) -> dict[str, Any]:
        return {
            "n_steps": self.rollout_steps,
            "batch_size": self.minibatch_size,
            "n_epochs": self.update_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "learning_rate": self.learning_rate,
            "clip_range": self.clip_coef,
            "ent_coef": self.entropy_coef,
            "vf_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "policy_kwargs": {"net_arch": list(self.hidden_sizes)},
            "seed": self.seed,
            "device": self.device,
        }


class NavigationVecEnv(VecEnv):
    def __init__(
        self,
        task_factory: Callable[[], Any],
        *,
        num_envs: int,
        env_device: str = "cpu",
        env_runner: str = "sync",
        mp_start_method: str = "spawn",
    ) -> None:
        self.runner = make_task_runner(
            task_factory,
            num_envs=num_envs,
            env_device=env_device,
            env_runner=env_runner,
            mp_start_method=mp_start_method,
        )
        self._owns_runner = True
        self._actions: np.ndarray | None = None
        super().__init__(
            num_envs=self.runner.num_envs,
            observation_space=_observation_space(self.runner.observation_shape),
            action_space=_action_space(self.runner),
        )
        self.metadata = {"render_modes": []}
        self.render_mode = None

    @classmethod
    def from_runner(cls, runner) -> "NavigationVecEnv":
        vec_env = cls.__new__(cls)
        vec_env.runner = runner
        vec_env._owns_runner = False
        vec_env._actions = None
        VecEnv.__init__(
            vec_env,
            num_envs=runner.num_envs,
            observation_space=_observation_space(runner.observation_shape),
            action_space=_action_space(runner),
        )
        vec_env.metadata = {"render_modes": []}
        vec_env.render_mode = None
        return vec_env

    def reset(self) -> np.ndarray:
        seeds = list(self._seeds)
        obs, infos = self.runner.reset_at(range(self.num_envs), seeds)
        infos = [dict(info) for info in infos]
        obs_nonfinite_counts = _nonfinite_counts_by_env(obs, self.num_envs)
        obs = _finite_float32_array(obs)
        _mark_nonfinite_reset_infos(infos, obs_nonfinite_counts)
        self.reset_infos = infos
        self._reset_seeds()
        self._reset_options()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait.")
        obs, rewards, terminated, truncated, infos = self.runner.step(self._actions)
        self._actions = None
        infos = [dict(info) for info in infos]
        obs_nonfinite_counts = _nonfinite_counts_by_env(obs, self.num_envs)
        reward_nonfinite = ~np.isfinite(np.asarray(rewards, dtype=np.float32))
        nonfinite_indices = np.flatnonzero(
            (obs_nonfinite_counts > 0) | reward_nonfinite
        )
        obs = _finite_float32_array(obs)
        rewards = _finite_float32_array(rewards)
        terminated = np.asarray(terminated, dtype=bool).copy()
        truncated = np.asarray(truncated, dtype=bool).copy()
        if nonfinite_indices.size:
            _mark_nonfinite_step_infos(
                infos,
                nonfinite_indices,
                obs_nonfinite_counts,
                reward_nonfinite,
            )
            terminated[nonfinite_indices] = True
            truncated[nonfinite_indices] = False
        dones = np.logical_or(terminated, truncated)
        done_indices = np.flatnonzero(dones)
        for env_idx in done_indices:
            infos[env_idx]["terminal_observation"] = obs[env_idx].copy()
            infos[env_idx]["TimeLimit.truncated"] = bool(
                truncated[env_idx] and not terminated[env_idx]
            )
        if done_indices.size:
            reset_obs, reset_infos = self.runner.reset_at(done_indices)
            reset_infos = [dict(info) for info in reset_infos]
            reset_obs_nonfinite_counts = _nonfinite_counts_by_env(
                reset_obs,
                len(done_indices),
            )
            reset_obs = _finite_float32_array(reset_obs)
            _mark_nonfinite_reset_infos(reset_infos, reset_obs_nonfinite_counts)
            for row, env_idx in enumerate(done_indices):
                obs[env_idx] = reset_obs[row]
                self.reset_infos[env_idx] = reset_infos[row]
        return obs, rewards, dones, infos

    def close(self) -> None:
        if self._owns_runner:
            self.runner.close()

    def get_images(self):
        return [None for _ in range(self.num_envs)]

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return [
            getattr(self._target(index), attr_name, None)
            for index in self._get_indices(indices)
        ]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        for index in self._get_indices(indices):
            setattr(self._target(index), attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ) -> list[Any]:
        results = []
        for index in self._get_indices(indices):
            method = getattr(self._target(index), method_name)
            results.append(method(*method_args, **method_kwargs))
        return results

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        del wrapper_class
        return [False for _ in self._get_indices(indices)]

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = getattr(self.runner, "diagnostics", None)
        return {"workers": [] if diagnostics is None else diagnostics()}

    def _target(self, index: int):
        tasks = getattr(self.runner, "tasks", None)
        if tasks is None:
            return self.runner
        return tasks[index]


def make_navigation_vec_env(
    task_factory: Callable[[], Any],
    config: PPOConfig,
) -> NavigationVecEnv:
    return NavigationVecEnv(
        task_factory,
        num_envs=config.num_envs,
        env_device=config.env_device,
        env_runner=config.env_runner,
        mp_start_method=config.mp_start_method,
    )


def build_training_vec_env(
    task_factory: Callable[[], Any],
    config: PPOConfig,
    *,
    vecnormalize_path: Path | str | None = None,
) -> VecNormalize:
    env = make_navigation_vec_env(task_factory, config)
    env = VecMonitor(env)
    if vecnormalize_path is not None:
        env = VecNormalize.load(str(vecnormalize_path), env)
        env.training = True
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return env


def build_eval_vec_env(
    task_factory: Callable[[], Any],
    config: PPOConfig,
    *,
    num_envs: int,
) -> VecNormalize:
    eval_config = PPOConfig(
        total_timesteps=config.total_timesteps,
        num_envs=num_envs,
        rollout_steps=config.rollout_steps,
        update_epochs=config.update_epochs,
        minibatch_size=config.minibatch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        learning_rate=config.learning_rate,
        clip_coef=config.clip_coef,
        entropy_coef=config.entropy_coef,
        value_coef=config.value_coef,
        max_grad_norm=config.max_grad_norm,
        hidden_sizes=config.hidden_sizes,
        seed=config.seed,
        device=config.device,
        env_device=config.env_device,
        env_runner=config.env_runner,
        mp_start_method=config.mp_start_method,
    )
    env = make_navigation_vec_env(task_factory, eval_config)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    return env


def evaluation_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    returns = np.asarray([result["return"] for result in results], dtype=np.float32)
    lengths = np.asarray([result["steps"] for result in results], dtype=np.float32)
    successes = np.asarray([result["success"] for result in results], dtype=np.float32)
    return {
        "eval/success_rate": float(successes.mean()),
        "eval/mean_reward": float(returns.mean()),
        "eval/mean_ep_length": float(lengths.mean()),
    }


def _observation_space(observation_shape: tuple[int, ...]) -> spaces.Box:
    return spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=observation_shape,
        dtype=np.float32,
    )


def _action_space(runner) -> spaces.Box | spaces.Discrete:
    if runner.action_space == "continuous":
        return spaces.Box(
            low=np.asarray(runner.action_low, dtype=np.float32),
            high=np.asarray(runner.action_high, dtype=np.float32),
            dtype=np.float32,
        )
    if runner.action_n is None:
        raise ValueError("discrete action space requires action_n.")
    return spaces.Discrete(int(runner.action_n))
