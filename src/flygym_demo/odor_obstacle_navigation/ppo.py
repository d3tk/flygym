from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

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
        dones = np.logical_or(terminated, truncated)
        infos = [dict(info) for info in infos]
        done_indices = np.flatnonzero(dones)
        for env_idx in done_indices:
            infos[env_idx]["terminal_observation"] = obs[env_idx].copy()
            infos[env_idx]["TimeLimit.truncated"] = bool(
                truncated[env_idx] and not terminated[env_idx]
            )
        if done_indices.size:
            reset_obs, reset_infos = self.runner.reset_at(done_indices)
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


def evaluation_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    returns = np.asarray([result["return"] for result in results], dtype=np.float32)
    lengths = np.asarray([result["steps"] for result in results], dtype=np.float32)
    successes = np.asarray([result["success"] for result in results], dtype=np.float32)
    return {
        "eval/success_rate": float(100.0 * successes.mean()),
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
