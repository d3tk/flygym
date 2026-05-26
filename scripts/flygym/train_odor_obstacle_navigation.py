from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flygym_demo.odor_obstacle_navigation import (  # noqa: E402
    NavigationTaskConfig,
    NavigationTaskFactory,
    PPOConfig,
    PPOTrainer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a native PPO policy for odor obstacle navigation."
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
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
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
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

    print(
        "[train] "
        f"run_dir={args.run_dir} "
        f"num_envs={args.num_envs} "
        f"rollout_steps={args.rollout_steps} "
        f"decision_steps={args.decision_steps} "
        f"decision_interval={task_config.decision_interval:.6f}s "
        f"reward=w_progress:{task_config.progress_reward_weight:g} "
        f"time_penalty:{task_config.time_penalty:g} "
        f"success_bonus:{task_config.success_bonus:g} "
        f"action_space={args.action_space} "
        f"env_device={args.env_device} "
        f"env_runner={args.env_runner} "
        f"wandb={'off' if args.no_wandb else args.wandb_mode}",
        flush=True,
    )
    task_factory = NavigationTaskFactory(deepcopy(task_config))
    wandb_run = _init_wandb(args, run_config)
    trainer = PPOTrainer(
        task_factory,
        ppo_config,
    )
    worker_diagnostics_path = args.run_dir / "worker_diagnostics.json"
    with worker_diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(trainer.diagnostics(), f, indent=2)
    print(
        f"[train] worker diagnostics written to {worker_diagnostics_path}",
        flush=True,
    )
    try:
        metrics = trainer.train(
            metrics_callback=(
                None if wandb_run is None else lambda values: wandb_run.log(values)
            ),
            progress_callback=(
                None if args.progress_interval_seconds <= 0 else _print_progress
            ),
            progress_interval_seconds=args.progress_interval_seconds,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            eval_seed=args.seed + 100_000,
        )
        trainer.save(args.run_dir / "policy.pt")
    finally:
        trainer.close()
        if wandb_run is not None:
            wandb_run.finish()
    print(json.dumps(metrics, indent=2))


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
            "learning_rate": ppo_config.learning_rate,
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
    wandb.define_metric("train/optimizer_step")
    wandb.define_metric("train/*", step_metric="train/optimizer_step")
    wandb.define_metric("rollout/*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step")
    wandb.define_metric("time/*", step_metric="global_step")
    wandb.define_metric("speed/*", step_metric="global_step")
    return run


def _print_progress(event: dict) -> None:
    stage = event["stage"]
    if stage == "rollout_start":
        print(
            "[train] "
            f"rollout={int(event['rollout_index'])} start "
            f"global_step={int(event['global_step'])} "
            f"steps={int(event['rollout_steps'])} "
            f"num_envs={int(event['num_envs'])} "
            f"decision_interval={event['decision_interval']:.6f}s",
            flush=True,
        )
    elif stage in {"rollout", "rollout_end"}:
        phase_text = ""
        if stage == "rollout_end":
            phase_text = (
                f" env_step={event.get('phase_env_step_wall_seconds', 0.0):.1f}s"
                f" policy={event.get('phase_policy_forward_wall_seconds', 0.0):.1f}s"
                f" transfer={event.get('phase_action_transfer_wall_seconds', 0.0):.1f}s"
                f" reset={event.get('phase_reset_wall_seconds', 0.0):.1f}s"
                f" worker_cpu_sum={event.get('worker_cpu_percent_sum', 0.0) or 0.0:.1f}%"
            )
        print(
            "[train] "
            f"rollout={int(event['rollout_index'])} "
            f"{int(event['rollout_step'])}/{int(event['rollout_steps'])} "
            f"global_step={int(event['global_step'])} "
            f"elapsed={event['rollout_wall_seconds']:.1f}s "
            f"decisions/s={event['rollout_timesteps_per_sec']:.2f} "
            f"substeps/s={event['simulator_substeps_per_sec']:.0f} "
            f"sim_s/s={event['simulated_seconds_per_sec']:.3f} "
            f"episodes={int(event['episodes_completed'])}"
            f"{phase_text}",
            flush=True,
        )
    elif stage == "update_start":
        print(
            "[train] "
            f"rollout={int(event['rollout_index'])} update start "
            f"global_step={int(event['global_step'])}",
            flush=True,
        )
    elif stage == "update_end":
        print(
            "[train] "
            f"rollout={int(event['rollout_index'])} update done "
            f"elapsed={event['update_wall_seconds']:.1f}s "
            f"loss={event['train/loss']:.4g} "
            f"value_loss={event['train/value_loss']:.4g} "
            f"approx_kl={event['train/approx_kl']:.4g}",
            flush=True,
        )
    elif stage == "eval_start":
        print(
            "[train] "
            f"eval start global_step={int(event['global_step'])} "
            f"episodes={int(event['eval_episodes'])}",
            flush=True,
        )
    elif stage == "eval_end":
        print(
            "[train] "
            f"eval done elapsed={event['eval_wall_seconds']:.1f}s "
            f"success={event['eval/success_rate']:.1f}% "
            f"reward={event['eval/mean_reward']:.3f} "
            f"len={event['eval/mean_ep_length']:.1f}",
            flush=True,
        )
    else:
        print(f"[train] {stage}", flush=True)


if __name__ == "__main__":
    main()
