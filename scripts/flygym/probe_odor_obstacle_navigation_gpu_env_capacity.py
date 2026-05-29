#!/usr/bin/env python3
"""Find the max parallel Warp env count for this GPU.

Default mode: verify ``--search-start``, estimate a high ``num_envs`` from VRAM (or
optional ``--calibration-envs`` fit), probe that count first, then binary-search the
OOM boundary. If the initial high count succeeds, env count doubles until OOM.

Each probe runs in a fresh subprocess so CUDA OOM does not poison later trials.
Memory is read from ``nvidia-smi`` (Warp + PyTorch on the device).

Example:
    bash scripts/flygym/launch_probe_gpu_env_capacity.sh
    bash scripts/flygym/launch_probe_gpu_env_capacity.sh --search-start 64 --search-step 16
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

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

FEATURE_MODES = {
    "full": (True, True),
    "no-vision": (False, True),
    "no-olfaction": (True, False),
    "no-sensors": (False, False),
}

# Hard stop for exponential search (not a normal cap on usable env count).
_SANITY_MAX_ENVS = 8192

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe GPU VRAM for odor obstacle navigation Warp envs and "
            "estimate the maximum num_envs."
        )
    )
    parser.add_argument(
        "--calibration-envs",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional num_envs for a low-end VRAM fit (e.g. 1 2 4 8 16 32 64). "
            "Default: skip calibration and only sweep upward from --search-start."
        ),
    )
    parser.add_argument(
        "--search-start",
        type=int,
        default=64,
        help="Known-good floor for num_envs; must succeed before the OOM sweep.",
    )
    parser.add_argument(
        "--search-step",
        type=int,
        default=16,
        help="Step size when bracketing upward after the initial high probe succeeds.",
    )
    parser.add_argument(
        "--reservation-fraction",
        type=float,
        default=0.90,
        help="Used only with --calibration-envs to estimate max from a linear fit.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Decision steps after reset before recording peak memory.",
    )
    parser.add_argument("--ccd-iterations", type=int, default=300)
    parser.add_argument("--decision-steps", type=int, default=500)
    parser.add_argument("--timestep", type=float, default=1e-4)
    parser.add_argument(
        "--feature-mode",
        choices=sorted(FEATURE_MODES),
        default="full",
    )
    parser.add_argument(
        "--include-ppo",
        action="store_true",
        help="Also allocate a CUDA PPO policy (matches cuda policy training).",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[128, 128],
    )
    parser.add_argument(
        "--skip-oom-sweep",
        action="store_true",
        help="Do not sweep above --search-start (only run --calibration-envs if set).",
    )
    parser.add_argument(
        "--max-search-ceiling",
        type=int,
        default=None,
        help=(
            "Optional cap on num_envs during the sweep. By default the first probe "
            "uses VRAM (or calibration fit) and search continues until OOM."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/odor_obstacle_navigation/benchmarks/gpu_env_capacity/probe.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_num-envs", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_metrics-json", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args._child:
        _run_child(args)
        return
    _run_parent(args)


def _run_parent(args: argparse.Namespace) -> None:
    if not (0.0 < args.reservation_fraction <= 1.0):
        raise ValueError("reservation-fraction must be in (0, 1].")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps must be non-negative.")
    if args.search_start <= 0:
        raise ValueError("search-start must be positive.")
    if args.search_step <= 0:
        raise ValueError("search-step must be positive.")
    if args.max_search_ceiling is not None and args.max_search_ceiling <= args.search_start:
        raise ValueError("max-search-ceiling must be greater than search-start.")

    gpu_name, _memory_used_mib, memory_total_mib = query_gpu_memory_mib()
    calibration_envs = sorted(
        {int(n) for n in (args.calibration_envs or []) if int(n) > 0}
    )

    print(f"GPU: {gpu_name}", flush=True)
    print(f"Total VRAM: {memory_total_mib} MiB", flush=True)
    print(f"Feature mode: {args.feature_mode}", flush=True)
    print(f"Include PPO on CUDA: {args.include_ppo}", flush=True)
    if calibration_envs:
        print(f"Calibration num_envs: {calibration_envs}", flush=True)
    else:
        ceiling_note = (
            f", optional_ceiling={args.max_search_ceiling}"
            if args.max_search_ceiling is not None
            else ""
        )
        print(
            f"OOM sweep: floor={args.search_start}, step={args.search_step}"
            f"{ceiling_note}",
            flush=True,
        )

    calibration_rows: list[dict[str, Any]] = []
    last_good = 0
    for num_envs in calibration_envs:
        row = _run_probe_subprocess(args, num_envs)
        calibration_rows.append(row)
        _print_probe_row(num_envs, row, memory_total_mib)
        if row["succeeded"]:
            last_good = max(last_good, num_envs)

    fit = _fit_memory_model(calibration_rows)
    target_mib = int(memory_total_mib * args.reservation_fraction)
    estimated_max = _estimate_max_envs(fit, target_mib)

    oom_sweep_rows: list[dict[str, Any]] = []
    max_verified = last_good
    max_probe_row: dict[str, Any] | None = None
    initial_high = 0
    if not args.skip_oom_sweep:
        max_verified, max_probe_row, oom_sweep_rows, initial_high = _oom_sweep_max(
            args,
            memory_total_mib=memory_total_mib,
            estimated_max=estimated_max,
        )

    report = {
        "gpu_name": gpu_name,
        "memory_total_mib": memory_total_mib,
        "reservation_fraction": args.reservation_fraction,
        "target_mib": target_mib,
        "search_start": args.search_start,
        "search_step": args.search_step,
        "initial_high_num_envs": initial_high,
        "max_search_ceiling": args.max_search_ceiling,
        "feature_mode": args.feature_mode,
        "include_ppo": args.include_ppo,
        "ccd_iterations": args.ccd_iterations,
        "decision_steps": args.decision_steps,
        "timestep": args.timestep,
        "warmup_steps": args.warmup_steps,
        "calibration": calibration_rows,
        "oom_sweep": oom_sweep_rows,
        "memory_fit": fit,
        "estimated_max_num_envs": estimated_max,
        "max_verified_num_envs": max_verified,
        "max_verified_probe": max_probe_row,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("", flush=True)
    print("--- Summary ---", flush=True)
    if fit["per_env_mib"] > 0:
        print(
            f"VRAM model: {fit['baseline_mib']:.0f} MiB baseline + "
            f"{fit['per_env_mib']:.2f} MiB per env "
            f"(R²={fit['r_squared']:.3f}, n={fit['n_samples']})",
            flush=True,
        )
        print(
            f"Estimated max num_envs at {args.reservation_fraction:.0%} VRAM "
            f"({target_mib} MiB): {estimated_max}",
            flush=True,
        )
    elif calibration_envs:
        print(
            "Could not fit per-env VRAM (need at least two successful calibration points).",
            flush=True,
        )
    print(f"Max verified num_envs (this machine): {max_verified}", flush=True)
    print(f"Wrote {args.output_json}", flush=True)


def _print_probe_row(
    num_envs: int,
    row: dict[str, Any],
    memory_total_mib: int,
) -> None:
    if row["succeeded"]:
        print(
            f"  n={num_envs:4d}: construct +{row['construct_delta_mib']} MiB, "
            f"warmup +{row['warmup_delta_mib']} MiB, "
            f"device used {row['memory_after_warmup_mib']} / {memory_total_mib} MiB",
            flush=True,
        )
    else:
        print(f"  n={num_envs:4d}: FAIL ({row.get('error', 'unknown')})", flush=True)


def _effective_search_cap(
    args: argparse.Namespace,
    *,
    memory_total_mib: int,
) -> int:
    cap = min(_SANITY_MAX_ENVS, memory_total_mib)
    if args.max_search_ceiling is not None:
        cap = min(cap, args.max_search_ceiling)
    return cap


def _derive_initial_high(
    *,
    args: argparse.Namespace,
    memory_total_mib: int,
    start_row: dict[str, Any],
    estimated_max: int,
) -> int:
    if estimated_max > args.search_start:
        return estimated_max
    mem = int(start_row["memory_after_warmup_mib"])
    baseline = int(start_row.get("memory_baseline_mib", 0))
    used_mib = max(mem - baseline, 1)
    per_env_mib = used_mib / args.search_start
    target_mib = int(memory_total_mib * args.reservation_fraction)
    return max(args.search_start + args.search_step, int(target_mib / per_env_mib))


def _oom_sweep_max(
    args: argparse.Namespace,
    *,
    memory_total_mib: int,
    estimated_max: int,
) -> tuple[int, dict[str, Any] | None, list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    search_cap = _effective_search_cap(args, memory_total_mib=memory_total_mib)

    start_row = _run_probe_subprocess(args, args.search_start)
    rows.append(start_row)
    _print_probe_row(args.search_start, start_row, memory_total_mib)
    if not start_row["succeeded"]:
        raise SystemExit(
            f"--search-start {args.search_start} failed: {start_row.get('error', 'unknown')}"
        )

    last_good = args.search_start
    best_row = start_row
    initial_high = min(
        _derive_initial_high(
            args=args,
            memory_total_mib=memory_total_mib,
            start_row=start_row,
            estimated_max=estimated_max,
        ),
        search_cap,
    )
    print(
        f"Initial high probe: n={initial_high} "
        f"(VRAM target {int(memory_total_mib * args.reservation_fraction)} MiB, "
        f"search cap {search_cap})",
        flush=True,
    )

    high_row = _run_probe_subprocess(args, initial_high)
    rows.append(high_row)
    _print_probe_row(initial_high, high_row, memory_total_mib)

    first_fail: int | None = None
    if high_row["succeeded"]:
        last_good = initial_high
        best_row = high_row
        candidate = min(last_good * 2, search_cap)
        while candidate > last_good:
            if candidate == last_good:
                break
            row = _run_probe_subprocess(args, candidate)
            rows.append(row)
            _print_probe_row(candidate, row, memory_total_mib)
            if row["succeeded"]:
                last_good = candidate
                best_row = row
                if candidate >= search_cap:
                    print(
                        f"Reached search cap {search_cap} without OOM.",
                        flush=True,
                    )
                    break
                candidate = min(last_good * 2, search_cap)
                continue
            first_fail = candidate
            break
    else:
        first_fail = initial_high

    if first_fail is not None and first_fail > last_good + 1:
        print(
            f"Binary search between last good n={last_good} and first fail n={first_fail} ...",
            flush=True,
        )
        last_good, best_row = _binary_search_max(
            args,
            low=last_good,
            high=first_fail,
        )

    print(f"Max verified num_envs: {last_good}", flush=True)
    return last_good, best_row, rows, initial_high


def _run_child(args: argparse.Namespace) -> None:
    import gc

    import numpy as np
    import warp as wp

    from flygym.warp.utils import check_gpu
    from flygym_demo.odor_obstacle_navigation.task import NavigationTaskConfig
    from flygym_demo.odor_obstacle_navigation.warp_runner import WarpNavigationTaskRunner

    num_envs = args._num_envs
    if num_envs <= 0:
        raise ValueError("_num_envs must be positive in child mode.")

    result: dict[str, Any] = {
        "num_envs": num_envs,
        "succeeded": False,
        "error": "",
        "memory_baseline_mib": 0,
        "memory_after_construct_mib": 0,
        "memory_after_warmup_mib": 0,
        "construct_delta_mib": 0,
        "warmup_delta_mib": 0,
        "torch_peak_allocated_mib": 0.0,
        "construction_wall_seconds": 0.0,
        "warmup_wall_seconds": 0.0,
    }

    runner = None
    try:
        check_gpu()
        wp.init()
        wp.synchronize()
        result["memory_baseline_mib"] = query_gpu_memory_mib()[1]

        enable_vision, enable_olfaction = FEATURE_MODES[args.feature_mode]
        task_config = NavigationTaskConfig(
            timestep=args.timestep,
            ccd_iterations=args.ccd_iterations,
            decision_steps=args.decision_steps,
            randomize_layout=False,
            enable_vision=enable_vision,
            enable_olfaction=enable_olfaction,
            render=False,
        )

        construction_started_at = time.perf_counter()
        runner = WarpNavigationTaskRunner(task_config, num_envs=num_envs)
        wp.synchronize()
        result["construction_wall_seconds"] = (
            time.perf_counter() - construction_started_at
        )
        result["memory_after_construct_mib"] = query_gpu_memory_mib()[1]
        result["construct_delta_mib"] = (
            result["memory_after_construct_mib"] - result["memory_baseline_mib"]
        )

        ppo_model = None
        if args.include_ppo:
            import torch
            from stable_baselines3 import PPO

            from flygym_demo.odor_obstacle_navigation.ppo import NavigationVecEnv

            torch.cuda.reset_peak_memory_stats()
            ppo_env = NavigationVecEnv.from_runner(runner)
            ppo_model = PPO(
                "MlpPolicy",
                ppo_env,
                n_steps=max(2, args.warmup_steps),
                batch_size=max(2, args.warmup_steps),
                policy_kwargs={"net_arch": list(args.hidden_sizes)},
                device="cuda",
            )
            wp.synchronize()
            result["torch_peak_allocated_mib"] = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            )

        warmup_started_at = time.perf_counter()
        obs, _ = runner.reset(seed=args.seed)
        del obs
        action = np.zeros((num_envs, 2), dtype=np.float32)
        for _ in range(args.warmup_steps):
            _, _, terminated, truncated, _ = runner.step(action)
            done_indices = np.flatnonzero(np.logical_or(terminated, truncated))
            if done_indices.size:
                runner.reset_at(done_indices)
        wp.synchronize()
        if ppo_model is not None:
            import torch

            obs_tensor = torch.zeros(
                num_envs,
                runner.observation_shape[0],
                device="cuda",
            )
            with torch.no_grad():
                ppo_model.policy(obs_tensor, deterministic=True)
            wp.synchronize()
            result["torch_peak_allocated_mib"] = max(
                result["torch_peak_allocated_mib"],
                torch.cuda.max_memory_allocated() / (1024 * 1024),
            )
        result["warmup_wall_seconds"] = time.perf_counter() - warmup_started_at
        result["memory_after_warmup_mib"] = query_gpu_memory_mib()[1]
        result["warmup_delta_mib"] = (
            result["memory_after_warmup_mib"] - result["memory_after_construct_mib"]
        )
        result["succeeded"] = True
    except Exception as exc:
        result["error"] = _format_error(exc)
        result["succeeded"] = False
    finally:
        if runner is not None:
            runner.close()
        gc.collect()
        try:
            wp.synchronize()
        except Exception:
            pass

    if args._metrics_json is not None:
        args._metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args._metrics_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    if not result["succeeded"]:
        sys.exit(1)


def _run_probe_subprocess(args: argparse.Namespace, num_envs: int) -> dict[str, Any]:
    metrics_path = args.output_json.parent / f"probe_n{num_envs}.json"
    command = _child_command(args, num_envs, metrics_path)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if metrics_path.is_file():
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
        row["exit_code"] = completed.returncode
        return row
    return {
        "num_envs": num_envs,
        "succeeded": False,
        "error": f"probe subprocess failed (exit {completed.returncode})",
        "exit_code": completed.returncode,
    }


def _child_command(
    args: argparse.Namespace,
    num_envs: int,
    metrics_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child",
        "--_num-envs",
        str(num_envs),
        "--_metrics-json",
        str(metrics_path),
        "--ccd-iterations",
        str(args.ccd_iterations),
        "--decision-steps",
        str(args.decision_steps),
        "--timestep",
        str(args.timestep),
        "--feature-mode",
        args.feature_mode,
        "--warmup-steps",
        str(args.warmup_steps),
        "--seed",
        str(args.seed),
    ]
    if args.include_ppo:
        command.append("--include-ppo")
        command.extend(["--hidden-sizes", *[str(x) for x in args.hidden_sizes]])
    return command


def _binary_search_max(
    args: argparse.Namespace,
    *,
    low: int,
    high: int,
) -> tuple[int, dict[str, Any] | None]:
    best_row: dict[str, Any] | None = None
    while low < high:
        mid = (low + high + 1) // 2
        row = _run_probe_subprocess(args, mid)
        if row["succeeded"]:
            low = mid
            best_row = row
            print(f"  n={mid}: ok ({row['memory_after_warmup_mib']} MiB used)", flush=True)
        else:
            high = mid - 1
            print(f"  n={mid}: failed ({row.get('error', 'unknown')})", flush=True)
    if best_row is None and low > 0:
        best_row = _run_probe_subprocess(args, low)
    return low, best_row


def _fit_memory_model(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    points = [
        (int(row["num_envs"]), float(row["memory_after_warmup_mib"]))
        for row in rows
        if row.get("succeeded")
    ]
    if len(points) < 2:
        if len(points) == 1:
            n_envs, memory_mib = points[0]
            return {
                "baseline_mib": memory_mib,
                "per_env_mib": 0.0,
                "r_squared": 0.0,
                "n_samples": 1,
                "note": "single_point_only",
                "single_num_envs": n_envs,
            }
        return {
            "baseline_mib": 0.0,
            "per_env_mib": 0.0,
            "r_squared": 0.0,
            "n_samples": 0,
        }

    xs = [float(n) for n, _ in points]
    ys = [mem for _, mem in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x <= 0:
        return {
            "baseline_mib": ys[0],
            "per_env_mib": 0.0,
            "r_squared": 0.0,
            "n_samples": len(points),
        }
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return {
        "baseline_mib": intercept,
        "per_env_mib": slope,
        "r_squared": r_squared,
        "n_samples": len(points),
    }


def _estimate_max_envs(fit: dict[str, float | int], target_mib: int) -> int:
    per_env = float(fit.get("per_env_mib", 0.0))
    baseline = float(fit.get("baseline_mib", 0.0))
    if per_env <= 0:
        single = fit.get("single_num_envs")
        if single is not None:
            return int(single)
        return 0
    available = target_mib - baseline
    if available <= 0:
        return 0
    return max(0, int(available // per_env))


def query_gpu_memory_mib(device_index: int = 0) -> tuple[str, int, int]:
    """Return ``(gpu_name, memory_used_mib, memory_total_mib)``."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            f"--id={device_index}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    line = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"Unexpected nvidia-smi output: {line!r}")
    name, used_mib, total_mib = parts
    return name, int(used_mib), int(total_mib)


def _format_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    main()
