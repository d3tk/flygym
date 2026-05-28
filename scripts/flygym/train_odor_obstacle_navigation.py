from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flygym_demo.odor_obstacle_navigation import (  # noqa: E402
    NavigationTaskConfig,
    NavigationTaskFactory,
    OdorObstacleNavigationTask,
    PPOConfig,
    evaluation_metrics,
    make_navigation_vec_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an SB3 PPO policy for odor obstacle navigation."
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--progress-interval-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--env-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Environment stepping device. Use cuda for the MuJoCo-Warp runner.",
    )
    parser.add_argument(
        "--env-runner",
        choices=["sync", "subproc", "warp"],
        default="sync",
    )
    parser.add_argument(
        "--mp-start-method",
        choices=["spawn", "forkserver", "fork"],
        default="spawn",
    )
    parser.add_argument("--timestep", type=float, default=1e-4)
    parser.add_argument(
        "--ccd-iterations",
        type=int,
        default=NavigationTaskConfig().ccd_iterations,
    )
    parser.add_argument("--decision-steps", type=int, default=500)
    parser.add_argument("--episode-time", type=float, default=5.0)
    parser.add_argument("--target-radius", type=float, default=2.0)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--progress-reward-weight", type=float, default=1.0)
    parser.add_argument("--time-penalty", type=float, default=0.01)
    parser.add_argument(
        "--action-space",
        choices=["continuous", "discrete"],
        default="continuous",
    )
    parser.add_argument("--vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--olfaction", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--randomize-layout", action="store_true")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/odor_obstacle_navigation"),
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY", "auto-fly"),
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "autonomous-drosophila"),
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive.")
    task_config = NavigationTaskConfig(
        timestep=args.timestep,
        ccd_iterations=args.ccd_iterations,
        decision_steps=args.decision_steps,
        episode_time=args.episode_time,
        target_radius=args.target_radius,
        success_bonus=args.success_bonus,
        progress_reward_weight=args.progress_reward_weight,
        time_penalty=args.time_penalty,
        action_space=args.action_space,
        enable_vision=args.vision,
        enable_olfaction=args.olfaction,
        randomize_layout=args.randomize_layout,
        render=False,
    )
    ppo_config = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        clip_coef=args.clip_coef,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
        env_device=args.env_device,
        env_runner=args.env_runner,
        mp_start_method=args.mp_start_method,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    run_config = _json_ready(args, task_config, ppo_config)
    with (args.run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(_training_summary(args, task_config), flush=True)
    wandb_run = _init_wandb(args, run_config)
    env = make_navigation_vec_env(NavigationTaskFactory(deepcopy(task_config)), ppo_config)
    worker_diagnostics_path = args.run_dir / "worker_diagnostics.json"
    with worker_diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(env.diagnostics(), f, indent=2)
    print(
        f"[train] worker diagnostics written to {worker_diagnostics_path}",
        flush=True,
    )

    callback = _TrainingCallback(
        task_config=task_config,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_seed=args.seed + 100_000,
        progress_interval_seconds=args.progress_interval_seconds,
        wandb_run=wandb_run,
    )
    started_at = time.perf_counter()
    try:
        model = PPO("MlpPolicy", env, **ppo_config.sb3_kwargs())
        model.learn(total_timesteps=ppo_config.total_timesteps, callback=callback)
        model.save(args.run_dir / "policy.zip")
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()
    metrics = {
        "global_step": float(ppo_config.total_timesteps),
        "time/wall_seconds": time.perf_counter() - started_at,
        **callback.last_metrics,
    }
    print(json.dumps(metrics, indent=2))


class _TrainingCallback(BaseCallback):
    def __init__(
        self,
        *,
        task_config: NavigationTaskConfig,
        eval_interval: int | None,
        eval_episodes: int,
        eval_seed: int,
        progress_interval_seconds: float,
        wandb_run,
    ) -> None:
        super().__init__()
        self.task_config = task_config
        self.eval_interval = None if eval_interval is None or eval_interval <= 0 else eval_interval
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.progress_interval_seconds = progress_interval_seconds
        self.wandb_run = wandb_run
        self.last_metrics: dict[str, float] = {}
        self._started_at = 0.0
        self._next_progress_at = 0.0
        self._next_eval_step = self.eval_interval

    def _on_training_start(self) -> None:
        self._started_at = time.perf_counter()
        self._next_progress_at = self._started_at + self.progress_interval_seconds

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if self.progress_interval_seconds > 0 and now >= self._next_progress_at:
            elapsed = now - self._started_at
            metrics = {
                "global_step": float(self.num_timesteps),
                "time/wall_seconds": elapsed,
                "time/fps": float(self.num_timesteps / elapsed) if elapsed > 0 else 0.0,
            }
            self._log(metrics)
            print(
                "[train] "
                f"global_step={self.num_timesteps} "
                f"elapsed={elapsed:.1f}s "
                f"decisions/s={metrics['time/fps']:.2f}",
                flush=True,
            )
            self._next_progress_at = now + self.progress_interval_seconds

        while self._next_eval_step is not None and self.num_timesteps >= self._next_eval_step:
            metrics = evaluation_metrics(
                _evaluate_model(
                    self.model,
                    self.task_config,
                    episodes=self.eval_episodes,
                    seed=self.eval_seed + self.num_timesteps,
                )
            )
            metrics["global_step"] = float(self.num_timesteps)
            self.last_metrics = metrics
            self._log(metrics)
            print(
                "[train] "
                f"eval global_step={self.num_timesteps} "
                f"success={metrics['eval/success_rate']:.1f}% "
                f"reward={metrics['eval/mean_reward']:.3f} "
                f"len={metrics['eval/mean_ep_length']:.1f}",
                flush=True,
            )
            self._next_eval_step += self.eval_interval
        return True

    def _log(self, metrics: dict[str, float]) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(metrics)


def _evaluate_model(
    model: PPO,
    task_config: NavigationTaskConfig,
    *,
    episodes: int,
    seed: int,
) -> list[dict[str, float]]:
    results = []
    for episode_idx in range(episodes):
        task = OdorObstacleNavigationTask(deepcopy(task_config))
        try:
            obs, info = task.reset(seed=seed + episode_idx)
            done = False
            total_reward = 0.0
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = task.step(action)
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
        finally:
            task.close()
    return results


def _training_summary(
    args: argparse.Namespace,
    task_config: NavigationTaskConfig,
) -> str:
    return (
        "[train] "
        f"run_dir={args.run_dir} "
        f"num_envs={args.num_envs} "
        f"rollout_steps={args.rollout_steps} "
        f"decision_steps={args.decision_steps} "
        f"ccd_iterations={args.ccd_iterations} "
        f"decision_interval={task_config.decision_interval:.6f}s "
        f"reward=w_progress:{task_config.progress_reward_weight:g} "
        f"time_penalty:{task_config.time_penalty:g} "
        f"success_bonus:{task_config.success_bonus:g} "
        f"action_space={args.action_space} "
        f"env_device={args.env_device} "
        f"env_runner={args.env_runner} "
        f"wandb={'off' if args.no_wandb else args.wandb_mode}"
    )


def _json_ready(
    args: argparse.Namespace,
    task_config: NavigationTaskConfig,
    ppo_config: PPOConfig,
) -> dict:
    return {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "task_config": {
            "decision_steps": task_config.decision_steps,
            "decision_interval": task_config.decision_interval,
            "episode_time": task_config.episode_time,
            "timestep": task_config.timestep,
            "target_radius": task_config.target_radius,
            "success_bonus": task_config.success_bonus,
            "progress_reward_weight": task_config.progress_reward_weight,
            "time_penalty": task_config.time_penalty,
            "action_space": task_config.action_space,
            "action_bounds": list(task_config.action_bounds),
            "action_max_norm": task_config.action_max_norm,
            "enable_vision": task_config.enable_vision,
            "enable_olfaction": task_config.enable_olfaction,
            "randomize_layout": task_config.randomize_layout,
            "odor_source": task_config.odor_source.tolist(),
            "obstacle_positions": task_config.obstacle_positions.tolist(),
        },
        "ppo_config": {
            "total_timesteps": ppo_config.total_timesteps,
            "num_envs": ppo_config.num_envs,
            "rollout_steps": ppo_config.rollout_steps,
            "update_epochs": ppo_config.update_epochs,
            "minibatch_size": ppo_config.minibatch_size,
            "gamma": ppo_config.gamma,
            "gae_lambda": ppo_config.gae_lambda,
            "learning_rate": ppo_config.learning_rate,
            "clip_coef": ppo_config.clip_coef,
            "entropy_coef": ppo_config.entropy_coef,
            "value_coef": ppo_config.value_coef,
            "max_grad_norm": ppo_config.max_grad_norm,
            "hidden_sizes": list(ppo_config.hidden_sizes),
            "seed": ppo_config.seed,
            "device": ppo_config.device,
            "env_device": ppo_config.env_device,
            "env_runner": ppo_config.env_runner,
            "mp_start_method": ppo_config.mp_start_method,
        },
    }


def _init_wandb(args: argparse.Namespace, run_config: dict):
    if args.no_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb or pass --no-wandb.") from exc

    os.environ["WANDB_ENTITY"] = args.wandb_entity
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_MODE"] = args.wandb_mode
    print(
        "[train] "
        f"initializing wandb entity={args.wandb_entity} "
        f"project={args.wandb_project} mode={args.wandb_mode}",
        flush=True,
    )
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        mode=args.wandb_mode,
        name=args.wandb_run_name,
        dir=str(args.run_dir),
        config=run_config,
    )
    wandb.define_metric("global_step")
    wandb.define_metric("eval/*", step_metric="global_step")
    wandb.define_metric("time/*", step_metric="global_step")
    return run


if __name__ == "__main__":
    main()
