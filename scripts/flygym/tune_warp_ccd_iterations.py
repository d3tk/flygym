#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Sequence

import numpy as np

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

CCD_WARNING_RE = re.compile(r"ccd_iterations.*needs to be increased")
OOM_RE = re.compile(r"(out of memory|Failed to allocate \d+ bytes)", re.IGNORECASE)
DEFAULT_CCD_ITERATIONS = [200, 250, 300, 350, 400]
FEATURE_MODES = {
    "full": (True, True),
    "no-vision": (False, True),
    "no-olfaction": (True, False),
    "no-sensors": (False, False),
}


@dataclass(slots=True)
class TrialResult:
    ccd_iterations: int
    num_envs: int
    repeat: int
    ccd_warning_count: int
    construction_wall_seconds: float
    rollout_wall_seconds: float
    phase_env_step_wall_seconds: float
    simulator_substeps_per_second: float
    nan_or_inf_count: int
    mean_min_obstacle_clearance: float
    mean_forward_distance: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune MuJoCo-Warp ccd_iterations with a fixed collision rollout."
    )
    parser.add_argument(
        "--ccd-iterations",
        type=int,
        nargs="+",
        default=DEFAULT_CCD_ITERATIONS,
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument(
        "--feature-mode",
        choices=sorted(FEATURE_MODES),
        default="full",
    )
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--decision-steps", type=int, default=500)
    parser.add_argument("--timestep", type=float, default=1e-4)
    parser.add_argument("--episode-time", type=float, default=None)
    parser.add_argument("--action", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/odor_obstacle_navigation/benchmarks/ccd_iterations"),
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--auto-reduce-envs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry OOM trials with fewer envs until they fit or min-num-envs is hit.",
    )
    parser.add_argument("--min-num-envs", type=int, default=1)
    parser.add_argument("--env-reduction-factor", type=float, default=0.75)
    parser.add_argument("--max-warning-count", type=int, default=0)
    parser.add_argument("--max-slowdown-percent", type=float, default=10.0)
    parser.add_argument("--max-mean-obstacle-clearance", type=float, default=0.5)
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_repeat", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_metrics-json", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args._child:
        _run_child(args)
        return
    _run_parent(args)


def count_ccd_warnings(text: str) -> int:
    return len(CCD_WARNING_RE.findall(text))


def min_obstacle_clearances(
    fly_positions: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radius: float,
) -> np.ndarray:
    fly_positions = np.asarray(fly_positions, dtype=float)
    obstacle_positions = np.asarray(obstacle_positions, dtype=float)
    if obstacle_positions.size == 0:
        return np.full(fly_positions.shape[0], np.inf, dtype=float)
    xy_delta = fly_positions[:, np.newaxis, :2] - obstacle_positions[np.newaxis, :, :]
    center_distances = np.linalg.norm(xy_delta, axis=2)
    return np.min(center_distances - float(obstacle_radius), axis=1)


def choose_recommended_ccd(
    rows: list[TrialResult],
    *,
    max_warning_count: int,
    max_slowdown_percent: float,
    max_mean_obstacle_clearance: float,
) -> int | None:
    if not rows:
        return None
    grouped = _group_rows(rows)
    baseline_ccd = 100 if 100 in grouped else min(grouped)
    baseline_speed = _finite_mean(
        row.simulator_substeps_per_second for row in grouped[baseline_ccd]
    )
    max_slowdown = max_slowdown_percent / 100.0
    for ccd_iterations in sorted(grouped):
        group = grouped[ccd_iterations]
        warning_count = sum(row.ccd_warning_count for row in group)
        nan_or_inf_count = sum(row.nan_or_inf_count for row in group)
        speed = _finite_mean(row.simulator_substeps_per_second for row in group)
        clearance = _finite_mean(row.mean_min_obstacle_clearance for row in group)
        if warning_count > max_warning_count or nan_or_inf_count:
            continue
        if not math.isfinite(clearance) or clearance > max_mean_obstacle_clearance:
            continue
        if not math.isfinite(speed) or speed <= 0:
            continue
        if baseline_speed > 0 and (1.0 - speed / baseline_speed) > max_slowdown:
            continue
        return ccd_iterations
    return None


def _run_parent(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or args.output_dir / "ccd_sweep.csv"
    rows: list[TrialResult] = []
    for ccd_iterations in args.ccd_iterations:
        for repeat in range(args.repeats):
            num_envs = args.num_envs
            attempt = 0
            while True:
                log_path = _attempt_path(
                    args.output_dir,
                    ccd_iterations=ccd_iterations,
                    num_envs=num_envs,
                    repeat=repeat,
                    attempt=attempt,
                    suffix=".log",
                )
                metrics_path = _attempt_path(
                    args.output_dir,
                    ccd_iterations=ccd_iterations,
                    num_envs=num_envs,
                    repeat=repeat,
                    attempt=attempt,
                    suffix=".json",
                )
                if metrics_path.exists():
                    metrics_path.unlink()
                with log_path.open("w", encoding="utf-8") as log_file:
                    completed = subprocess.run(
                        _child_command(
                            args,
                            ccd_iterations,
                            repeat,
                            metrics_path,
                            num_envs,
                        ),
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                warning_count = count_ccd_warnings(log_text)
                if (
                    args.auto_reduce_envs
                    and _is_oom_failure(log_text, completed.returncode)
                    and num_envs > args.min_num_envs
                ):
                    next_num_envs = _reduced_num_envs(
                        num_envs,
                        factor=args.env_reduction_factor,
                        min_num_envs=args.min_num_envs,
                    )
                    print(
                        f"ccd={ccd_iterations:<4} repeat={repeat:<2} "
                        f"num_envs={num_envs:<5} OOM; retrying num_envs={next_num_envs} "
                        f"log={log_path}",
                        flush=True,
                    )
                    num_envs = next_num_envs
                    attempt += 1
                    continue
                row = _load_trial_result(
                    metrics_path,
                    ccd_iterations,
                    num_envs,
                    repeat,
                    warning_count,
                )
                row.ccd_warning_count = warning_count
                rows.append(row)
                _write_rows(output_csv, rows)
                _print_trial(row, log_path, completed.returncode)
                break

    recommendation = choose_recommended_ccd(
        rows,
        max_warning_count=args.max_warning_count,
        max_slowdown_percent=args.max_slowdown_percent,
        max_mean_obstacle_clearance=args.max_mean_obstacle_clearance,
    )
    print(f"\nWrote CCD sweep rows to {output_csv}", flush=True)
    if recommendation is None:
        print("No ccd_iterations value satisfied the recommendation rule.", flush=True)
    else:
        print(f"Recommended ccd_iterations: {recommendation}", flush=True)


def _run_child(args: argparse.Namespace) -> None:
    result = run_stress_rollout(
        args,
        ccd_iterations=args.ccd_iterations[0],
        repeat=args._repeat,
    )
    if args._metrics_json is not None:
        with args._metrics_json.open("w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
    print(json.dumps(asdict(result), indent=2), flush=True)


def run_stress_rollout(
    args: argparse.Namespace,
    *,
    ccd_iterations: int,
    repeat: int,
) -> TrialResult:
    from flygym.warp.utils import check_gpu
    from flygym_demo.odor_obstacle_navigation.task import NavigationTaskConfig
    from flygym_demo.odor_obstacle_navigation.warp_runner import WarpNavigationTaskRunner
    import warp as wp

    check_gpu()
    enable_vision, enable_olfaction = FEATURE_MODES[args.feature_mode]
    episode_time = args.episode_time
    if episode_time is None:
        episode_time = (args.rollout_steps + 1) * args.decision_steps * args.timestep
    task_config = NavigationTaskConfig(
        timestep=args.timestep,
        ccd_iterations=ccd_iterations,
        decision_steps=args.decision_steps,
        episode_time=episode_time,
        randomize_layout=False,
        enable_vision=enable_vision,
        enable_olfaction=enable_olfaction,
        render=False,
    )

    construction_started_at = time.perf_counter()
    runner = WarpNavigationTaskRunner(task_config, num_envs=args.num_envs)
    wp.synchronize()
    construction_wall_seconds = time.perf_counter() - construction_started_at
    try:
        obs, infos = runner.reset(seed=args.seed + repeat)
        wp.synchronize()
        action = np.asarray(args.action, dtype=np.float32)
        actions = np.repeat(action[np.newaxis, :], args.num_envs, axis=0)
        fly_positions = _fly_positions_from_infos(infos)
        min_clearances = min_obstacle_clearances(
            fly_positions,
            task_config.obstacle_positions,
            task_config.obstacle_radius,
        )
        nan_or_inf_count = _count_nonfinite(obs, fly_positions)
        phase_env_step_wall_seconds = 0.0
        rollout_started_at = time.perf_counter()
        for _ in range(args.rollout_steps):
            step_started_at = time.perf_counter()
            obs, rewards, _, _, infos = runner.step(actions)
            wp.synchronize()
            phase_env_step_wall_seconds += time.perf_counter() - step_started_at
            fly_positions = _fly_positions_from_infos(infos)
            min_clearances = np.minimum(
                min_clearances,
                min_obstacle_clearances(
                    fly_positions,
                    task_config.obstacle_positions,
                    task_config.obstacle_radius,
                ),
            )
            nan_or_inf_count += _count_nonfinite(obs, rewards, fly_positions)
        rollout_wall_seconds = time.perf_counter() - rollout_started_at
    finally:
        runner.close()

    simulator_substeps = args.num_envs * args.rollout_steps * args.decision_steps
    return TrialResult(
        ccd_iterations=ccd_iterations,
        num_envs=args.num_envs,
        repeat=repeat,
        ccd_warning_count=0,
        construction_wall_seconds=construction_wall_seconds,
        rollout_wall_seconds=rollout_wall_seconds,
        phase_env_step_wall_seconds=phase_env_step_wall_seconds,
        simulator_substeps_per_second=_rate(simulator_substeps, rollout_wall_seconds),
        nan_or_inf_count=nan_or_inf_count,
        mean_min_obstacle_clearance=float(np.mean(min_clearances)),
        mean_forward_distance=float(
            np.mean(fly_positions[:, 0] - task_config.spawn_position[0])
        ),
    )


def _child_command(
    args: argparse.Namespace,
    ccd_iterations: int,
    repeat: int,
    metrics_path: Path,
    num_envs: int,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child",
        "--_repeat",
        str(repeat),
        "--_metrics-json",
        str(metrics_path),
        "--ccd-iterations",
        str(ccd_iterations),
        "--num-envs",
        str(num_envs),
        "--feature-mode",
        args.feature_mode,
        "--rollout-steps",
        str(args.rollout_steps),
        "--decision-steps",
        str(args.decision_steps),
        "--timestep",
        str(args.timestep),
        "--action",
        str(args.action[0]),
        str(args.action[1]),
        "--seed",
        str(args.seed),
    ]
    if args.episode_time is not None:
        command.extend(["--episode-time", str(args.episode_time)])
    return command


def _load_trial_result(
    metrics_path: Path,
    ccd_iterations: int,
    num_envs: int,
    repeat: int,
    warning_count: int,
) -> TrialResult:
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("num_envs", num_envs)
        return TrialResult(**data)
    return TrialResult(
        ccd_iterations=ccd_iterations,
        num_envs=num_envs,
        repeat=repeat,
        ccd_warning_count=warning_count,
        construction_wall_seconds=math.nan,
        rollout_wall_seconds=math.nan,
        phase_env_step_wall_seconds=math.nan,
        simulator_substeps_per_second=math.nan,
        nan_or_inf_count=1,
        mean_min_obstacle_clearance=math.inf,
        mean_forward_distance=math.nan,
    )


def _write_rows(path: Path, rows: list[TrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(TrialResult)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _print_trial(row: TrialResult, log_path: Path, returncode: int) -> None:
    print(
        f"ccd={row.ccd_iterations:<4} repeat={row.repeat:<2} "
        f"num_envs={row.num_envs:<5} "
        f"warnings={row.ccd_warning_count:<3} "
        f"nan_inf={row.nan_or_inf_count:<3} "
        f"substeps/s={row.simulator_substeps_per_second:>10.0f} "
        f"clearance={row.mean_min_obstacle_clearance:>7.3f} "
        f"forward={row.mean_forward_distance:>7.3f} "
        f"returncode={returncode} log={log_path}",
        flush=True,
    )


def _group_rows(rows: list[TrialResult]) -> dict[int, list[TrialResult]]:
    grouped: dict[int, list[TrialResult]] = {}
    for row in rows:
        grouped.setdefault(row.ccd_iterations, []).append(row)
    return grouped


def _attempt_path(
    output_dir: Path,
    *,
    ccd_iterations: int,
    num_envs: int,
    repeat: int,
    attempt: int,
    suffix: str,
) -> Path:
    if attempt == 0:
        return output_dir / f"ccd_{ccd_iterations}_repeat_{repeat}{suffix}"
    return output_dir / (
        f"ccd_{ccd_iterations}_envs_{num_envs}_repeat_{repeat}_attempt_{attempt}{suffix}"
    )


def _is_oom_failure(log_text: str, returncode: int) -> bool:
    return returncode != 0 and OOM_RE.search(log_text) is not None


def _reduced_num_envs(
    num_envs: int,
    *,
    factor: float,
    min_num_envs: int,
) -> int:
    reduced = max(min_num_envs, int(math.floor(num_envs * factor)))
    if reduced >= num_envs:
        reduced = num_envs - 1
    return max(min_num_envs, reduced)


def _finite_mean(values) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return math.nan
    return float(np.mean(finite_values))


def _fly_positions_from_infos(infos: list[dict]) -> np.ndarray:
    return np.stack([np.asarray(info["fly_position"], dtype=np.float32) for info in infos])


def _count_nonfinite(*arrays: np.ndarray) -> int:
    count = 0
    for array in arrays:
        values = np.asarray(array)
        count += int(values.size - np.count_nonzero(np.isfinite(values)))
    return count


def _rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _validate_args(args: argparse.Namespace) -> None:
    if any(ccd_iterations <= 0 for ccd_iterations in args.ccd_iterations):
        raise ValueError("ccd-iterations values must be positive.")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive.")
    if args.num_envs <= 0:
        raise ValueError("num-envs must be positive.")
    if args.min_num_envs <= 0:
        raise ValueError("min-num-envs must be positive.")
    if args.min_num_envs > args.num_envs:
        raise ValueError("min-num-envs cannot exceed num-envs.")
    if not 0.0 < args.env_reduction_factor < 1.0:
        raise ValueError("env-reduction-factor must be between 0 and 1.")
    if args.rollout_steps <= 0:
        raise ValueError("rollout-steps must be positive.")
    if args.decision_steps <= 0:
        raise ValueError("decision-steps must be positive.")
    if args.timestep <= 0:
        raise ValueError("timestep must be positive.")
    if args.episode_time is not None and args.episode_time <= 0:
        raise ValueError("episode-time must be positive.")
    if args.max_warning_count < 0:
        raise ValueError("max-warning-count must be non-negative.")
    if args.max_slowdown_percent < 0:
        raise ValueError("max-slowdown-percent must be non-negative.")


if __name__ == "__main__":
    main()
