"""Train SB3 PPO for odor obstacle navigation with v1-style logging.

Smoke run (4 warp envs, 5k steps, no W&B):

    uv run --extra warp python scripts/flygym/train_odor_obstacle_navigation.py \\
        --total-timesteps 5000 --num-envs 4 --env-device cuda --env-runner warp \\
        --device cuda --no-wandb --run-dir .outputs/flygym/sb3/smoke_test
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.modules.setdefault("tensorflow", None)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import sync_envs_normalization  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flygym.utils.stderr_filter import install_ccd_iterations_warning_filter  # noqa: E402

install_ccd_iterations_warning_filter()

import wandb  # noqa: E402
from wandb.integration.sb3 import WandbCallback  # noqa: E402

from flygym_demo.odor_obstacle_navigation import (  # noqa: E402
    NavigationTaskConfig,
    NavigationTaskFactory,
    PPOConfig,
    build_eval_vec_env,
    build_training_vec_env,
    resolve_minibatch_size,
)
from flygym_demo.odor_obstacle_navigation.training_callbacks import (  # noqa: E402
    FlyGymDiagnosticsCallback,
    NavigationEvalCallback,
    TrainingSpeedCallback,
    VecNormalizeFiniteCheckCallback,
    VecNormalizeOnNewBestCallback,
    resolved_env_timing_from_config,
    vecnormalize_path_for_checkpoint,
)


_V1_OBSTACLE_SLOTS = np.array(
    [
        (7.5, 0.0),
        (12.5, 5.0),
        (17.5, -5.0),
        (10.0, -5.0),
        (15.0, 5.0),
    ],
    dtype=float,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Deprecated alias for --eval-freq.",
    )
    parser.add_argument("--n-eval-envs", type=int, default=4)
    parser.add_argument("--n-eval-episodes", type=int, default=13)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
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
        "--layout-preset",
        choices=["default", "v1"],
        default="default",
    )
    parser.add_argument("--spawn-xy-jitter", type=float, default=None)
    parser.add_argument("--spawn-yaw-jitter", type=float, default=None)
    parser.add_argument(
        "--obstacle-count-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable W&B logging (default unless --no-wandb is set).",
    )
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
    parser.add_argument(
        "--wandb-run-name",
        "--wandb-name",
        dest="wandb_run_name",
        type=str,
        default=None,
    )
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-save-code", action="store_true")
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--load-policy",
        type=Path,
        default=None,
        help="Initialize from a policy .zip with fresh VecNormalize.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from a CheckpointCallback .zip and paired vecnormalize pickle.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.load_policy is not None and args.resume_from is not None:
        raise ValueError("Use only one of --load-policy or --resume-from.")
    if args.eval_interval is not None:
        args.eval_freq = args.eval_interval
    if args.n_eval_envs <= 0:
        raise ValueError("n_eval_envs must be positive.")
    if args.n_eval_episodes <= 0:
        raise ValueError("n_eval_episodes must be positive.")

    minibatch_size = resolve_minibatch_size(
        args.minibatch_size,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
    )
    if minibatch_size != args.minibatch_size:
        print(
            "[train] adjusted minibatch_size "
            f"{args.minibatch_size} -> {minibatch_size} "
            f"(rollout buffer {args.rollout_steps * args.num_envs})",
            flush=True,
        )

    task_config = _make_task_config(args)
    ppo_config = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        clip_coef=args.clip_coef,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_sizes=tuple(args.hidden_sizes),
        seed=args.seed,
        device=args.device,
        env_device=args.env_device,
        env_runner=args.env_runner,
        mp_start_method=args.mp_start_method,
    )
    timing = resolved_env_timing_from_config(task_config)
    run_dir = args.run_dir or Path(".outputs/flygym/sb3") / (
        args.wandb_run_name or f"odor_nav_{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = run_dir

    tensorboard_dir = run_dir / "tensorboard"
    checkpoints_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best_model"
    eval_dir = run_dir / "eval"
    for path in (tensorboard_dir, checkpoints_dir, best_dir, eval_dir):
        path.mkdir(parents=True, exist_ok=True)

    run_config = _json_ready(args, task_config, ppo_config, timing)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    use_wandb = not args.no_wandb
    wandb_run = _init_wandb(args, run_dir, run_config) if use_wandb else None
    if wandb_run is not None:
        wandb.save(str(run_dir / "config.json"), base_path=str(run_dir))

    print(_training_summary(args, task_config, timing, ppo_config), flush=True)

    resume_checkpoint: Path | None = None
    resume_vecnormalize: Path | None = None
    if args.resume_from is not None:
        resume_checkpoint = args.resume_from.expanduser().resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"--resume-from not found: {resume_checkpoint}")
        resume_vecnormalize = vecnormalize_path_for_checkpoint(resume_checkpoint)
        if not resume_vecnormalize.is_file():
            raise FileNotFoundError(
                f"VecNormalize checkpoint not found for resume: {resume_vecnormalize}"
            )

    task_factory = NavigationTaskFactory(deepcopy(task_config))
    train_env = build_training_vec_env(
        task_factory,
        ppo_config,
        vecnormalize_path=resume_vecnormalize,
    )
    worker_diagnostics = _collect_worker_diagnostics(train_env, args.num_envs)
    worker_diagnostics_path = run_dir / "worker_diagnostics.json"
    with worker_diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(worker_diagnostics, handle, indent=2)
    print(f"[train] worker diagnostics written to {worker_diagnostics_path}", flush=True)
    if wandb_run is not None:
        wandb.save(str(worker_diagnostics_path), base_path=str(run_dir))

    eval_env = build_eval_vec_env(
        NavigationTaskFactory(deepcopy(task_config)),
        ppo_config,
        num_envs=args.n_eval_envs,
    )
    eval_env.training = False
    eval_env.norm_reward = False

    train_hparams = {
        **ppo_config.sb3_kwargs(),
        "policy_kwargs": {"net_arch": list(ppo_config.hidden_sizes)},
        "tensorboard_log": str(tensorboard_dir),
        "verbose": 1,
    }
    if args.resume_from is not None:
        print(
            f"Resuming from {resume_checkpoint} "
            f"(VecNormalize {resume_vecnormalize})",
            flush=True,
        )
        model = PPO.load(str(resume_checkpoint), env=train_env, **train_hparams)
    elif args.load_policy is not None:
        load_path = args.load_policy.expanduser().resolve()
        if not load_path.is_file():
            raise FileNotFoundError(f"--load-policy not found: {load_path}")
        print(f"Loading policy weights from {load_path}", flush=True)
        model = PPO.load(str(load_path), env=train_env, **train_hparams)
    else:
        model = PPO("MlpPolicy", train_env, **train_hparams)

    total_eval_episodes = args.n_eval_envs * args.n_eval_episodes
    training_speed_path = run_dir / "training_speed.csv"
    callback_items = [
        TrainingSpeedCallback(
            training_speed_path,
            timing,
            log_to_wandb=wandb_run is not None,
            resume=resume_checkpoint is not None,
        ),
        FlyGymDiagnosticsCallback(reward_mode="extrinsic"),
        VecNormalizeFiniteCheckCallback(),
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.num_envs, 1),
            save_path=str(checkpoints_dir),
            name_prefix="navigation_policy",
            save_vecnormalize=True,
        ),
        NavigationEvalCallback(
            eval_env=eval_env,
            callback_on_new_best=VecNormalizeOnNewBestCallback(run_dir, best_dir),
            best_model_save_path=str(best_dir),
            log_path=str(eval_dir),
            eval_freq=max(args.eval_freq // args.num_envs, 1),
            n_eval_episodes=total_eval_episodes,
            deterministic=True,
            render=False,
            log_to_wandb=wandb_run is not None,
        ),
    ]
    if wandb_run is not None:
        callback_items.append(WandbCallback(verbose=2))
    callbacks = CallbackList(callback_items)

    vec_run_path = run_dir / "vecnormalize.pkl"
    try:
        sync_envs_normalization(train_env, eval_env)
        model.learn(
            total_timesteps=ppo_config.total_timesteps,
            callback=callbacks,
            progress_bar=args.progress_bar,
            reset_num_timesteps=resume_checkpoint is None,
        )
        model.save(run_dir / "final_model")
    finally:
        vec_env = model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(str(vec_run_path))
        train_env.close()
        eval_env.close()
        if wandb_run is not None:
            if training_speed_path.exists():
                wandb.save(str(training_speed_path), base_path=str(run_dir))
            wandb_run.finish()

    print(f"Saved run artifacts to {run_dir}", flush=True)


def _make_task_config(args: argparse.Namespace) -> NavigationTaskConfig:
    task_kwargs = {
        "timestep": args.timestep,
        "ccd_iterations": args.ccd_iterations,
        "decision_steps": args.decision_steps,
        "episode_time": args.episode_time,
        "target_radius": args.target_radius,
        "success_bonus": args.success_bonus,
        "progress_reward_weight": args.progress_reward_weight,
        "time_penalty": args.time_penalty,
        "action_space": args.action_space,
        "enable_vision": args.vision,
        "enable_olfaction": args.olfaction,
        "randomize_layout": args.randomize_layout,
        "render": False,
    }

    if args.layout_preset == "v1":
        task_kwargs.update(
            {
                "obstacle_positions": _V1_OBSTACLE_SLOTS.copy(),
                "obstacle_count_range": (3, 5),
                "inactive_obstacle_position": np.array([1000.0, 0.0], dtype=float),
                "layout_min_target_spawn_distance": 10.0,
                "layout_min_obstacle_spawn_distance": 3.0,
                "layout_min_obstacle_target_distance": 4.0,
                "layout_min_obstacle_obstacle_distance": 4.0,
                "spawn_xy_jitter": 0.0,
                "spawn_yaw_jitter": 0.0,
            }
        )

    if args.spawn_xy_jitter is not None:
        task_kwargs["spawn_xy_jitter"] = args.spawn_xy_jitter
    if args.spawn_yaw_jitter is not None:
        task_kwargs["spawn_yaw_jitter"] = args.spawn_yaw_jitter
    if args.obstacle_count_range is not None:
        min_count, max_count = args.obstacle_count_range
        task_kwargs["obstacle_count_range"] = (min_count, max_count)

    return NavigationTaskConfig(**task_kwargs)


def _collect_worker_diagnostics(vec_env, num_envs: int) -> dict:
    base_env = vec_env.venv if hasattr(vec_env, "venv") else vec_env
    while hasattr(base_env, "venv"):
        base_env = base_env.venv
    diagnostics = {"workers": []}
    if hasattr(base_env, "diagnostics"):
        try:
            diagnostics = base_env.diagnostics()
        except Exception as exc:  # pragma: no cover
            diagnostics = {"workers": [], "error": repr(exc)}
    parent_affinity = _cpu_affinity()
    diagnostics.setdefault("parent", {})
    diagnostics["parent"].update(
        {
            "pid": os.getpid(),
            "cpu_affinity": parent_affinity,
            "cpu_affinity_count": (
                None if parent_affinity is None else len(parent_affinity)
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
        }
    )
    diagnostics["requested_envs"] = num_envs
    worker_pids = [
        int(worker["pid"])
        for worker in diagnostics.get("workers", [])
        if isinstance(worker, dict) and worker.get("pid") is not None
    ]
    diagnostics["unique_worker_pid_count"] = len(set(worker_pids))
    return diagnostics


def _cpu_affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def _training_summary(
    args: argparse.Namespace,
    task_config: NavigationTaskConfig,
    timing,
    ppo_config: PPOConfig,
) -> str:
    buffer_size = ppo_config.num_envs * ppo_config.rollout_steps
    return (
        "[train] "
        f"run_dir={args.run_dir} "
        f"num_envs={ppo_config.num_envs} "
        f"rollout_steps={ppo_config.rollout_steps} "
        f"buffer={buffer_size} "
        f"minibatch_size={ppo_config.minibatch_size} "
        f"decision_steps={task_config.decision_steps} "
        f"ccd_iterations={args.ccd_iterations} "
        f"decision_interval={task_config.decision_interval:.6f}s "
        f"reward=w_progress:{task_config.progress_reward_weight:g} "
        f"time_penalty:{task_config.time_penalty:g} "
        f"success_bonus:{task_config.success_bonus:g} "
        f"action_space={args.action_space} "
        f"randomize_layout={task_config.randomize_layout} "
        f"layout_preset={args.layout_preset} "
        f"env_device={ppo_config.env_device} "
        f"env_runner={ppo_config.env_runner} "
        f"wandb={'off' if args.no_wandb else args.wandb_mode}"
    )


def _json_ready(
    args: argparse.Namespace,
    task_config: NavigationTaskConfig,
    ppo_config: PPOConfig,
    timing,
) -> dict:
    return {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "resolved_timing": {
            "timestep": timing.timestep,
            "decision_steps": timing.decision_steps,
            "decision_dt": timing.decision_dt,
            "max_decision_steps": timing.max_decision_steps,
            "episode_time_seconds": timing.episode_time_seconds,
        },
        "rollout": {
            "buffer_size": ppo_config.num_envs * ppo_config.rollout_steps,
            "minibatches": (ppo_config.num_envs * ppo_config.rollout_steps)
            // ppo_config.minibatch_size,
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
            "spawn_xy_jitter": task_config.spawn_xy_jitter,
            "spawn_yaw_jitter": task_config.spawn_yaw_jitter,
            "obstacle_count_range": (
                None
                if task_config.obstacle_count_range is None
                else list(task_config.obstacle_count_range)
            ),
            "inactive_obstacle_position": (
                task_config.inactive_obstacle_position.tolist()
            ),
            "layout_min_target_spawn_distance": (
                task_config.layout_min_target_spawn_distance
            ),
            "layout_min_obstacle_spawn_distance": (
                task_config.layout_min_obstacle_spawn_distance
            ),
            "layout_min_obstacle_target_distance": (
                task_config.layout_min_obstacle_target_distance
            ),
            "layout_min_obstacle_obstacle_distance": (
                task_config.layout_min_obstacle_obstacle_distance
            ),
            "layout_sampling_attempts": task_config.layout_sampling_attempts,
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


def _init_wandb(args: argparse.Namespace, run_dir: Path, run_config: dict):
    if args.wandb_mode == "disabled":
        return None
    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "dir": str(run_dir),
        "name": args.wandb_run_name or run_dir.name,
        "group": args.wandb_group,
        "tags": args.wandb_tags,
        "config": run_config,
        "sync_tensorboard": True,
        "monitor_gym": False,
        "save_code": args.wandb_save_code,
        "mode": args.wandb_mode,
    }
    if args.wandb_id is not None:
        init_kwargs["id"] = args.wandb_id
        init_kwargs["resume"] = "must"
    print(
        "[train] "
        f"initializing wandb entity={args.wandb_entity} "
        f"project={args.wandb_project} mode={args.wandb_mode}",
        flush=True,
    )
    return wandb.init(**init_kwargs)


if __name__ == "__main__":
    main()
