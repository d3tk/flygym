#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402

from flygym_demo.odor_obstacle_navigation import (  # noqa: E402
    NavigationTaskConfig,
    NavigationTaskFactory,
    PPOConfig,
    PPOTrainer,
)

FEATURE_MODES = {
    "full": (True, True),
    "no-vision": (False, True),
    "no-olfaction": (True, False),
    "no-sensors": (False, False),
}


@dataclass(slots=True)
class BenchmarkResult:
    ppo_device: str
    env_device: str
    runner: str
    mp_start_method: str
    feature_mode: str
    vision: bool
    olfaction: bool
    num_envs: int
    repeat: int
    rollout_steps: int
    rollouts: int
    decision_steps: int
    construction_wall_seconds: float
    train_wall_seconds: float
    total_wall_seconds: float
    rollout_wall_seconds: float
    update_wall_seconds: float
    rollout_decisions_per_second: float
    update_decisions_per_second: float
    train_decisions_per_second: float
    total_decisions_per_second: float
    simulator_substeps_per_second: float
    simulated_seconds_per_second: float
    phase_obs_normalize_wall_seconds: float
    phase_policy_forward_wall_seconds: float
    phase_action_transfer_wall_seconds: float
    phase_env_step_wall_seconds: float
    phase_reset_wall_seconds: float
    phase_bootstrap_wall_seconds: float
    phase_advantage_wall_seconds: float
    phase_tensorize_wall_seconds: float
    worker_pid_count: int
    worker_affinity_count_min: int
    worker_affinity_count_max: int
    worker_cpu_percent_sum: float
    worker_cpu_percent_mean: float
    worker_cpu_percent_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark odor obstacle navigation training throughput."
    )
    parser.add_argument(
        "--runners",
        nargs="+",
        choices=["sync", "subproc", "warp"],
        default=["sync", "subproc"],
    )
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 20])
    parser.add_argument(
        "--mp-start-method",
        choices=["spawn", "forkserver", "fork"],
        default="spawn",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--minibatch-size", type=int, default=1200)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ppo-device", "--device", dest="ppo_device", type=str, default="cpu")
    parser.add_argument(
        "--ppo-devices",
        "--devices",
        dest="ppo_devices",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--env-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Environment stepping device. cuda uses the MuJoCo-Warp runner.",
    )
    parser.add_argument(
        "--env-devices",
        choices=["cpu", "cuda"],
        nargs="+",
        default=None,
    )
    parser.add_argument("--timestep", type=float, default=1e-4)
    parser.add_argument("--decision-steps", type=int, default=500)
    parser.add_argument("--episode-time", type=float, default=5.0)
    parser.add_argument("--target-radius", type=float, default=2.0)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--progress-reward-weight", type=float, default=10.0)
    parser.add_argument("--vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--olfaction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        choices=sorted(FEATURE_MODES),
        default=None,
        help=(
            "Feature matrix to run. If omitted, use the single --vision/--olfaction "
            "combination."
        ),
    )
    parser.add_argument("--randomize-layout", action="store_true")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/odor_obstacle_navigation/benchmarks/training_speed.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive.")
    if args.rollouts <= 0:
        raise ValueError("rollouts must be positive.")
    if args.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive.")
    if any(num_envs <= 0 for num_envs in args.num_envs):
        raise ValueError("all num-envs values must be positive.")

    ppo_devices = _requested_ppo_devices(args)
    env_devices = _requested_env_devices(args)
    feature_modes = _requested_feature_modes(args)
    _validate_ppo_devices(ppo_devices)
    _validate_env_devices(env_devices)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[BenchmarkResult] = []
    print(_system_summary(args, ppo_devices, env_devices, feature_modes), flush=True)
    for ppo_device in ppo_devices:
        for env_device in env_devices:
            runners = _requested_runners(args, env_device)
            for runner in runners:
                for feature_mode in feature_modes:
                    vision, olfaction = FEATURE_MODES[feature_mode]
                    if not _supports_feature_mode(env_device, vision, olfaction):
                        print(
                            "Skipping "
                            f"ppo_device={ppo_device} env_device={env_device} "
                            f"runner={runner} features={feature_mode}: "
                            "unsupported feature mode.",
                            flush=True,
                        )
                        continue
                    for num_envs in args.num_envs:
                        for repeat in range(args.repeats):
                            result = _run_trial(
                                args,
                                ppo_device,
                                env_device,
                                runner,
                                feature_mode,
                                vision,
                                olfaction,
                                num_envs,
                                repeat,
                            )
                            rows.append(result)
                            _print_trial(result)
                            _write_rows(args.output_csv, rows)
    _print_summary(rows, args.output_csv)


def _run_trial(
    args: argparse.Namespace,
    ppo_device: str,
    env_device: str,
    runner: str,
    feature_mode: str,
    vision: bool,
    olfaction: bool,
    num_envs: int,
    repeat: int,
) -> BenchmarkResult:
    seed = args.seed + 10_000 * repeat + num_envs
    task_config = NavigationTaskConfig(
        timestep=args.timestep,
        decision_steps=args.decision_steps,
        episode_time=args.episode_time,
        target_radius=args.target_radius,
        success_bonus=args.success_bonus,
        progress_reward_weight=args.progress_reward_weight,
        randomize_layout=args.randomize_layout,
        enable_vision=vision,
        enable_olfaction=olfaction,
        render=False,
    )
    total_timesteps = args.rollouts * args.rollout_steps * num_envs
    ppo_config = PPOConfig(
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        hidden_sizes=tuple(args.hidden_sizes),
        seed=seed,
        device=ppo_device,
        env_device=env_device,
        env_runner=runner,
        mp_start_method=args.mp_start_method,
    )
    rollout_events = []
    _synchronize_devices(ppo_device, env_device)
    trial_started_at = time.perf_counter()
    construction_started_at = time.perf_counter()
    trainer = PPOTrainer(NavigationTaskFactory(task_config), ppo_config)
    _synchronize_devices(ppo_device, env_device)
    construction_wall_seconds = time.perf_counter() - construction_started_at
    worker_summary = _worker_diagnostics_summary(trainer.diagnostics())
    try:
        _synchronize_devices(ppo_device, env_device)
        train_started_at = time.perf_counter()
        trainer.train(
            metrics_callback=None,
            progress_callback=rollout_events.append,
            progress_interval_seconds=None,
            eval_interval=None,
        )
        _synchronize_devices(ppo_device, env_device)
        train_wall_seconds = time.perf_counter() - train_started_at
    finally:
        trainer.close()
    _synchronize_devices(ppo_device, env_device)
    total_wall_seconds = time.perf_counter() - trial_started_at
    rollout_wall_seconds = sum(
        event["rollout_wall_seconds"]
        for event in rollout_events
        if event["stage"] == "rollout_end"
    )
    rollout_end_events = [
        event for event in rollout_events if event["stage"] == "rollout_end"
    ]
    rollout_decisions = sum(
        event["rollout_timesteps"]
        for event in rollout_end_events
    )
    update_wall_seconds = sum(
        event["update_wall_seconds"]
        for event in rollout_events
        if event["stage"] == "update_end"
    )
    return BenchmarkResult(
        ppo_device=ppo_device,
        env_device=env_device,
        runner=runner,
        mp_start_method=args.mp_start_method if runner == "subproc" and env_device == "cpu" else "",
        feature_mode=feature_mode,
        vision=vision,
        olfaction=olfaction,
        num_envs=num_envs,
        repeat=repeat,
        rollout_steps=args.rollout_steps,
        rollouts=args.rollouts,
        decision_steps=args.decision_steps,
        construction_wall_seconds=construction_wall_seconds,
        train_wall_seconds=train_wall_seconds,
        total_wall_seconds=total_wall_seconds,
        rollout_wall_seconds=rollout_wall_seconds,
        update_wall_seconds=update_wall_seconds,
        rollout_decisions_per_second=_rate(rollout_decisions, rollout_wall_seconds),
        update_decisions_per_second=_rate(total_timesteps, update_wall_seconds),
        train_decisions_per_second=_rate(total_timesteps, train_wall_seconds),
        total_decisions_per_second=_rate(total_timesteps, total_wall_seconds),
        simulator_substeps_per_second=_rate(
            rollout_decisions * args.decision_steps,
            rollout_wall_seconds,
        ),
        simulated_seconds_per_second=_rate(
            rollout_decisions * task_config.decision_interval,
            rollout_wall_seconds,
        ),
        phase_obs_normalize_wall_seconds=_sum_event_field(
            rollout_end_events, "phase_obs_normalize_wall_seconds"
        ),
        phase_policy_forward_wall_seconds=_sum_event_field(
            rollout_end_events, "phase_policy_forward_wall_seconds"
        ),
        phase_action_transfer_wall_seconds=_sum_event_field(
            rollout_end_events, "phase_action_transfer_wall_seconds"
        ),
        phase_env_step_wall_seconds=_sum_event_field(
            rollout_end_events, "phase_env_step_wall_seconds"
        ),
        phase_reset_wall_seconds=_sum_event_field(
            rollout_end_events, "phase_reset_wall_seconds"
        ),
        phase_bootstrap_wall_seconds=_sum_event_field(
            rollout_end_events, "phase_bootstrap_wall_seconds"
        ),
        phase_advantage_wall_seconds=_sum_event_field(
            rollout_end_events, "phase_advantage_wall_seconds"
        ),
        phase_tensorize_wall_seconds=_sum_event_field(
            rollout_end_events, "phase_tensorize_wall_seconds"
        ),
        worker_pid_count=worker_summary["worker_pid_count"],
        worker_affinity_count_min=worker_summary["worker_affinity_count_min"],
        worker_affinity_count_max=worker_summary["worker_affinity_count_max"],
        worker_cpu_percent_sum=_mean_event_field(
            rollout_end_events, "worker_cpu_percent_sum"
        ),
        worker_cpu_percent_mean=_mean_event_field(
            rollout_end_events, "worker_cpu_percent_mean"
        ),
        worker_cpu_percent_max=_mean_event_field(
            rollout_end_events, "worker_cpu_percent_max"
        ),
    )


def _system_summary(
    args: argparse.Namespace,
    ppo_devices: list[str],
    env_devices: list[str],
    feature_modes: list[str],
) -> str:
    affinity_count = None
    if hasattr(os, "sched_getaffinity"):
        affinity_count = len(os.sched_getaffinity(0))
    summary = {
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": affinity_count,
        "threads": {
            "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
            "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
            "NUMEXPR_NUM_THREADS": os.environ["NUMEXPR_NUM_THREADS"],
            "VECLIB_MAXIMUM_THREADS": os.environ["VECLIB_MAXIMUM_THREADS"],
        },
        "mujoco_gl": os.environ["MUJOCO_GL"],
        "pyopengl_platform": os.environ["PYOPENGL_PLATFORM"],
        "decision_steps": args.decision_steps,
        "rollout_steps": args.rollout_steps,
        "rollouts": args.rollouts,
        "ppo_devices": ppo_devices,
        "env_devices": env_devices,
        "feature_modes": feature_modes,
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_devices": _cuda_device_names(),
        },
        "vision": args.vision,
        "olfaction": args.olfaction,
    }
    return json.dumps(summary, indent=2)


def _print_trial(result: BenchmarkResult) -> None:
    print(
        "ppo={ppo:<7} env={env:<4} runner={runner:<7} envs={envs:<4} repeat={repeat:<2} "
        "features={features:<11} "
        "rollout_dec/s={rollout_decisions_per_second:>8.2f} "
        "update_dec/s={update_decisions_per_second:>8.2f} "
        "train_dec/s={train_decisions_per_second:>8.2f} "
        "total_dec/s={total_decisions_per_second:>8.2f} "
        "substeps/s={simulator_substeps_per_second:>10.0f} "
        "sim_s/s={simulated_seconds_per_second:>7.3f} "
        "construct={construction_wall_seconds:>6.2f}s "
        "update={update_wall_seconds:>6.2f}s "
        "train={train_wall_seconds:>6.2f}s".format(
            ppo=result.ppo_device,
            env=result.env_device,
            runner=result.runner,
            envs=result.num_envs,
            repeat=result.repeat,
            features=result.feature_mode,
            rollout_decisions_per_second=result.rollout_decisions_per_second,
            update_decisions_per_second=result.update_decisions_per_second,
            train_decisions_per_second=result.train_decisions_per_second,
            total_decisions_per_second=result.total_decisions_per_second,
            simulator_substeps_per_second=result.simulator_substeps_per_second,
            simulated_seconds_per_second=result.simulated_seconds_per_second,
            construction_wall_seconds=result.construction_wall_seconds,
            update_wall_seconds=result.update_wall_seconds,
            train_wall_seconds=result.train_wall_seconds,
        ),
        flush=True,
    )


def _write_rows(path: Path, rows: list[BenchmarkResult]) -> None:
    fieldnames = list(BenchmarkResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def _print_summary(rows: list[BenchmarkResult], path: Path) -> None:
    if not rows:
        print(f"\nNo benchmark rows were produced; no CSV was written to {path}")
        return
    groups: dict[tuple[str, str, str, str, str, int], list[BenchmarkResult]] = {}
    for row in rows:
        groups.setdefault(
            (
                row.ppo_device,
                row.env_device,
                row.runner,
                row.mp_start_method,
                row.feature_mode,
                row.num_envs,
            ),
            [],
        ).append(row)
    summary = []
    for (
        ppo_device,
        env_device,
        runner,
        mp_start_method,
        feature_mode,
        num_envs,
    ), group_rows in groups.items():
        summary.append(
            {
                "ppo_device": ppo_device,
                "env_device": env_device,
                "runner": runner,
                "mp_start_method": mp_start_method,
                "feature_mode": feature_mode,
                "num_envs": num_envs,
                "mean_rollout_decisions_per_second": statistics.mean(
                    row.rollout_decisions_per_second for row in group_rows
                ),
                "mean_simulator_substeps_per_second": statistics.mean(
                    row.simulator_substeps_per_second for row in group_rows
                ),
                "mean_update_decisions_per_second": statistics.mean(
                    row.update_decisions_per_second for row in group_rows
                ),
                "mean_train_decisions_per_second": statistics.mean(
                    row.train_decisions_per_second for row in group_rows
                ),
                "mean_total_decisions_per_second": statistics.mean(
                    row.total_decisions_per_second for row in group_rows
                ),
            }
        )
    summary.sort(
        key=lambda row: row["mean_train_decisions_per_second"],
        reverse=True,
    )
    print(f"\nWrote benchmark rows to {path}")
    print("Top configurations by train decisions/s:")
    for row in summary[:10]:
        print(
            "ppo={ppo:<7} env={env:<4} runner={runner:<7} start={start:<10} envs={envs:<4} "
            "features={features:<11} "
            "rollout_dec/s={rollout_decisions_per_second:>8.2f} "
            "update_dec/s={update_decisions_per_second:>8.2f} "
            "train_dec/s={train_decisions_per_second:>8.2f} "
            "total_dec/s={total_decisions_per_second:>8.2f} "
            "substeps/s={substeps_per_second:>10.0f} "
            .format(
                ppo=row["ppo_device"],
                env=row["env_device"],
                runner=row["runner"],
                start=row["mp_start_method"] or "-",
                envs=row["num_envs"],
                features=row["feature_mode"],
                rollout_decisions_per_second=row[
                    "mean_rollout_decisions_per_second"
                ],
                update_decisions_per_second=row[
                    "mean_update_decisions_per_second"
                ],
                train_decisions_per_second=row["mean_train_decisions_per_second"],
                total_decisions_per_second=row["mean_total_decisions_per_second"],
                substeps_per_second=row["mean_simulator_substeps_per_second"],
            )
        )


def _requested_ppo_devices(args: argparse.Namespace) -> list[str]:
    return args.ppo_devices if args.ppo_devices is not None else [args.ppo_device]


def _requested_env_devices(args: argparse.Namespace) -> list[str]:
    return args.env_devices if args.env_devices is not None else [args.env_device]


def _requested_runners(args: argparse.Namespace, env_device: str) -> list[str]:
    if env_device == "cuda":
        return ["warp"]
    return [runner for runner in args.runners if runner != "warp"]


def _requested_feature_modes(args: argparse.Namespace) -> list[str]:
    if args.feature_modes is not None:
        return list(args.feature_modes)
    for name, (vision, olfaction) in FEATURE_MODES.items():
        if vision == args.vision and olfaction == args.olfaction:
            return [name]
    raise RuntimeError("Unsupported feature combination.")


def _supports_feature_mode(env_device: str, vision: bool, olfaction: bool) -> bool:
    return True


def _validate_ppo_devices(ppo_devices: list[str]) -> None:
    for ppo_device in ppo_devices:
        torch_device = torch.device(ppo_device)
        if torch_device.type != "cuda":
            continue
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA PPO device requested but unavailable: {ppo_device}")
        if torch_device.index is not None and torch_device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA PPO device index out of range: {ppo_device}")


def _validate_env_devices(env_devices: list[str]) -> None:
    for env_device in env_devices:
        if env_device == "cpu":
            continue
        if env_device != "cuda":
            raise RuntimeError(f"Unsupported env_device: {env_device}")
        try:
            from flygym.warp.utils import check_gpu
        except ImportError as exc:
            raise RuntimeError(
                "CUDA env_device requires FlyGym's warp extra. "
                "Install/sync with --extra warp."
            ) from exc
        check_gpu()


def _synchronize_devices(ppo_device: str, env_device: str) -> None:
    torch_device = torch.device(ppo_device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    if env_device == "cuda":
        import warp as wp

        wp.synchronize()


def _cuda_device_names() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]


def _worker_diagnostics_summary(diagnostics: dict) -> dict[str, int]:
    workers = diagnostics.get("workers", [])
    pids = {
        int(worker["pid"])
        for worker in workers
        if isinstance(worker, dict) and worker.get("pid") is not None
    }
    affinity_counts = [
        int(worker["cpu_affinity_count"])
        for worker in workers
        if isinstance(worker, dict) and worker.get("cpu_affinity_count") is not None
    ]
    return {
        "worker_pid_count": len(pids),
        "worker_affinity_count_min": min(affinity_counts) if affinity_counts else 0,
        "worker_affinity_count_max": max(affinity_counts) if affinity_counts else 0,
    }


def _sum_event_field(events: list[dict], field: str) -> float:
    return float(sum(float(event.get(field, 0.0) or 0.0) for event in events))


def _mean_event_field(events: list[dict], field: str) -> float:
    values = [
        float(event[field])
        for event in events
        if event.get(field) is not None
    ]
    return float(statistics.mean(values)) if values else 0.0


def _rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
