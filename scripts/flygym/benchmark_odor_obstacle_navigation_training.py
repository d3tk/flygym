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


@dataclass(slots=True)
class BenchmarkResult:
    device: str
    runner: str
    mp_start_method: str
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark odor obstacle navigation training throughput."
    )
    parser.add_argument(
        "--runners",
        nargs="+",
        choices=["sync", "subproc"],
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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--devices", type=str, nargs="+", default=None)
    parser.add_argument("--timestep", type=float, default=1e-4)
    parser.add_argument("--decision-steps", type=int, default=500)
    parser.add_argument("--episode-time", type=float, default=5.0)
    parser.add_argument("--target-radius", type=float, default=2.0)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--progress-reward-weight", type=float, default=10.0)
    parser.add_argument("--vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--olfaction", action=argparse.BooleanOptionalAction, default=True)
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

    devices = _requested_devices(args)
    _validate_devices(devices)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[BenchmarkResult] = []
    print(_system_summary(args, devices), flush=True)
    for device in devices:
        for runner in args.runners:
            for num_envs in args.num_envs:
                for repeat in range(args.repeats):
                    result = _run_trial(args, device, runner, num_envs, repeat)
                    rows.append(result)
                    _print_trial(result)
                    _write_rows(args.output_csv, rows)
    _print_summary(rows, args.output_csv)


def _run_trial(
    args: argparse.Namespace,
    device: str,
    runner: str,
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
        enable_vision=args.vision,
        enable_olfaction=args.olfaction,
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
        device=device,
        env_runner=runner,
        mp_start_method=args.mp_start_method,
    )
    rollout_events = []
    _synchronize_device(device)
    trial_started_at = time.perf_counter()
    construction_started_at = time.perf_counter()
    trainer = PPOTrainer(NavigationTaskFactory(task_config), ppo_config)
    _synchronize_device(device)
    construction_wall_seconds = time.perf_counter() - construction_started_at
    try:
        _synchronize_device(device)
        train_started_at = time.perf_counter()
        trainer.train(
            metrics_callback=None,
            progress_callback=rollout_events.append,
            progress_interval_seconds=None,
            eval_interval=None,
        )
        _synchronize_device(device)
        train_wall_seconds = time.perf_counter() - train_started_at
    finally:
        trainer.close()
    _synchronize_device(device)
    total_wall_seconds = time.perf_counter() - trial_started_at
    rollout_wall_seconds = sum(
        event["rollout_wall_seconds"]
        for event in rollout_events
        if event["stage"] == "rollout_end"
    )
    rollout_decisions = sum(
        event["rollout_timesteps"]
        for event in rollout_events
        if event["stage"] == "rollout_end"
    )
    update_wall_seconds = sum(
        event["update_wall_seconds"]
        for event in rollout_events
        if event["stage"] == "update_end"
    )
    return BenchmarkResult(
        device=device,
        runner=runner,
        mp_start_method=args.mp_start_method if runner == "subproc" else "",
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
    )


def _system_summary(args: argparse.Namespace, devices: list[str]) -> str:
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
        "devices": devices,
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
        "device={device:<7} runner={runner:<7} envs={envs:<3} repeat={repeat:<2} "
        "rollout_dec/s={rollout_decisions_per_second:>8.2f} "
        "update_dec/s={update_decisions_per_second:>8.2f} "
        "train_dec/s={train_decisions_per_second:>8.2f} "
        "total_dec/s={total_decisions_per_second:>8.2f} "
        "substeps/s={simulator_substeps_per_second:>10.0f} "
        "sim_s/s={simulated_seconds_per_second:>7.3f} "
        "construct={construction_wall_seconds:>6.2f}s "
        "update={update_wall_seconds:>6.2f}s "
        "train={train_wall_seconds:>6.2f}s".format(
            device=result.device,
            runner=result.runner,
            envs=result.num_envs,
            repeat=result.repeat,
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
    groups: dict[tuple[str, str, str, int], list[BenchmarkResult]] = {}
    for row in rows:
        groups.setdefault(
            (row.device, row.runner, row.mp_start_method, row.num_envs), []
        ).append(row)
    summary = []
    for (device, runner, mp_start_method, num_envs), group_rows in groups.items():
        summary.append(
            {
                "device": device,
                "runner": runner,
                "mp_start_method": mp_start_method,
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
            "device={device:<7} runner={runner:<7} start={start:<10} envs={envs:<3} "
            "rollout_dec/s={rollout_decisions_per_second:>8.2f} "
            "update_dec/s={update_decisions_per_second:>8.2f} "
            "train_dec/s={train_decisions_per_second:>8.2f} "
            "total_dec/s={total_decisions_per_second:>8.2f} "
            "substeps/s={substeps_per_second:>10.0f} "
            .format(
                device=row["device"],
                runner=row["runner"],
                start=row["mp_start_method"] or "-",
                envs=row["num_envs"],
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


def _requested_devices(args: argparse.Namespace) -> list[str]:
    return args.devices if args.devices is not None else [args.device]


def _validate_devices(devices: list[str]) -> None:
    for device in devices:
        torch_device = torch.device(device)
        if torch_device.type != "cuda":
            continue
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        if torch_device.index is not None and torch_device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index out of range: {device}")


def _synchronize_device(device: str) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def _cuda_device_names() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]


def _rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
