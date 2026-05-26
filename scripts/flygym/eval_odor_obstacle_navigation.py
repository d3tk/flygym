from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flygym_demo.odor_obstacle_navigation import (  # noqa: E402
    NavigationTaskConfig,
    OdorObstacleNavigationTask,
    PPOConfig,
    PPOTrainer,
    evaluation_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a native PPO odor obstacle navigation policy."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--decision-steps", type=int, default=500)
    parser.add_argument("--episode-time", type=float, default=5.0)
    parser.add_argument("--target-radius", type=float, default=2.0)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--progress-reward-weight", type=float, default=1.0)
    parser.add_argument("--time-penalty", type=float, default=0.01)
    parser.add_argument(
        "--action-space",
        choices=["continuous", "discrete"],
        default=None,
        help="Defaults to the checkpoint action space when available.",
    )
    parser.add_argument("--randomize-layout", action="store_true")
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true")
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
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    ppo_config = PPOConfig(**checkpoint["config"])
    ppo_config.num_envs = 1
    ppo_config.device = args.device
    ppo_config.env_runner = "sync"
    action_space = args.action_space or checkpoint.get("action_space", "continuous")
    args.action_space = action_space

    task_config = NavigationTaskConfig(
        decision_steps=args.decision_steps,
        episode_time=args.episode_time,
        target_radius=args.target_radius,
        success_bonus=args.success_bonus,
        progress_reward_weight=args.progress_reward_weight,
        time_penalty=args.time_penalty,
        action_space=action_space,
        randomize_layout=args.randomize_layout,
        render=args.video_path is not None,
    )
    trainer = PPOTrainer(
        lambda: OdorObstacleNavigationTask(deepcopy(task_config)),
        ppo_config,
    )
    trainer.load(args.checkpoint)
    try:
        if args.video_path is None:
            results = trainer.evaluate(args.episodes, seed=args.seed)
        else:
            results = _evaluate_with_video(trainer, task_config, args)
    finally:
        trainer.close()
    if args.wandb:
        _log_wandb(args, results)
    print(json.dumps(results, indent=2))


def _evaluate_with_video(
    trainer: PPOTrainer,
    task_config: NavigationTaskConfig,
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    results = []
    for episode_idx in range(args.episodes):
        task = OdorObstacleNavigationTask(deepcopy(task_config))
        obs, info = task.reset(seed=args.seed + episode_idx)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = trainer.act(obs)
            obs, reward, terminated, truncated, info = task.step(action)
            task.render()
            total_reward += reward
            steps += 1
            done = terminated or truncated
        if episode_idx == 0:
            task.save_video(args.video_path)
        results.append(
            {
                "return": float(total_reward),
                "steps": float(steps),
                "success": float(bool(info.get("success", False))),
            }
        )
        task.close()
    return results


def _log_wandb(args: argparse.Namespace, results: list[dict[str, float]]) -> None:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb or omit --wandb.") from exc

    os.environ["WANDB_ENTITY"] = args.wandb_entity
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_MODE"] = args.wandb_mode
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        mode=args.wandb_mode,
        name=args.wandb_run_name,
        config={
            "checkpoint": str(args.checkpoint),
            "episodes": args.episodes,
            "seed": args.seed,
            "decision_steps": args.decision_steps,
            "episode_time": args.episode_time,
            "target_radius": args.target_radius,
            "success_bonus": args.success_bonus,
            "progress_reward_weight": args.progress_reward_weight,
            "time_penalty": args.time_penalty,
            "action_space": args.action_space,
            "randomize_layout": args.randomize_layout,
        },
    )
    run.log(evaluation_metrics(results))
    run.finish()


if __name__ == "__main__":
    main()
