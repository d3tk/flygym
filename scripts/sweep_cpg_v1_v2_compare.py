#!/usr/bin/env python3
"""Sweep ``compare_cpg_v1_v2_kinematics.py`` over v2 CPG / actuator settings.

Each run writes its own NPZ, kinematics report, and (optionally) side-by-side MP4
under ``--output-dir``. A ``summary.csv`` aggregates thorax Δx/Δy/Δz and
tracking RMSE from the NPZs so you can sort by translation or regress gain vs Δx.

Example (fast sweep, no MP4 encoding; needs ``uv run --extra cpg-v1-compare``)::

    uv run --extra cpg-v1-compare python scripts/sweep_cpg_v1_v2_compare.py \\
        --output-dir debug_outputs/sweep_gain \\
        --gains 'lin 40 140 6' \\
        --target-dx-v2-mm 2.5

Re-render a short list with video for visual pick::

    uv run --extra cpg-v1-compare python scripts/sweep_cpg_v1_v2_compare.py \\
        --output-dir debug_outputs/sweep_top3 \\
        --gains 60,90,120 --with-mp4
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_float_grid(spec: str) -> list[float]:
    spec = spec.strip()
    if not spec:
        return []
    parts = spec.split()
    if len(parts) >= 4 and parts[0].lower() == "lin":
        lo, hi, n = float(parts[1]), float(parts[2]), int(parts[3])
        if n < 1:
            raise ValueError("lin N must be >= 1")
        return [float(x) for x in np.linspace(lo, hi, n)]
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def _stem(g: float, f_hz: float, c: float) -> str:
    """Filesystem-safe stem (avoid ``.`` / ``-`` in filenames)."""

    def q(x: float) -> str:
        t = f"{x:.8f}".rstrip("0").rstrip(".")
        return t.replace(".", "p").replace("-", "m")

    return f"g{q(g)}_f{q(f_hz)}_c{q(c)}"


def _stats_from_npz_arrays(
    joint_cmd: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    thorax_pos: np.ndarray,
    *,
    contact_flag: np.ndarray | None,
) -> dict[str, float]:
    err = joint_cmd - joint_pos
    rmse_t = np.sqrt(np.mean(err**2, axis=1))
    vel_l2 = np.linalg.norm(joint_vel, axis=1)
    thorax = np.asarray(thorax_pos)
    out: dict[str, float] = {
        "mean_rmse_cmd_vs_pos_rad": float(rmse_t.mean()),
        "max_rmse_cmd_vs_pos_rad": float(rmse_t.max()),
        "mean_L2_joint_vel": float(vel_l2.mean()),
        "thorax_delta_x_mm": float(thorax[-1, 0] - thorax[0, 0]),
        "thorax_delta_y_mm": float(thorax[-1, 1] - thorax[0, 1]),
        "thorax_delta_z_mm": float(thorax[-1, 2] - thorax[0, 2]),
    }
    if contact_flag is not None:
        out["mean_contact_frac"] = float(np.mean(contact_flag))
    return out


def _load_row_from_npz(npz_path: Path) -> dict[str, float]:
    d = np.load(npz_path, allow_pickle=True)
    need = (
        "sim_v1_joint_cmd",
        "sim_v1_joint_pos",
        "sim_v1_joint_vel",
        "sim_v1_thorax_pos",
        "sim_v2_joint_cmd",
        "sim_v2_joint_pos",
        "sim_v2_joint_vel",
        "sim_v2_thorax_pos",
    )
    for k in need:
        if k not in d:
            raise KeyError(f"{npz_path}: missing {k} (need full MuJoCo run, not --no-video)")
    cf2 = d["sim_v2_contact_flag"] if "sim_v2_contact_flag" in d else None
    s1 = _stats_from_npz_arrays(
        d["sim_v1_joint_cmd"],
        d["sim_v1_joint_pos"],
        d["sim_v1_joint_vel"],
        d["sim_v1_thorax_pos"],
        contact_flag=None,
    )
    s2 = _stats_from_npz_arrays(
        d["sim_v2_joint_cmd"],
        d["sim_v2_joint_pos"],
        d["sim_v2_joint_vel"],
        d["sim_v2_thorax_pos"],
        contact_flag=cf2,
    )
    out: dict[str, float] = {}
    for k, v in s1.items():
        out[f"v1_{k}"] = v
    for k, v in s2.items():
        out[f"v2_{k}"] = v
    return out


def _linear_gain_for_target_dx(gains: list[float], dxs: list[float], target: float) -> float | None:
    if len(gains) < 2 or len(gains) != len(dxs):
        return None
    m, b = np.polyfit(np.asarray(gains, dtype=float), np.asarray(dxs, dtype=float), 1)
    if abs(m) < 1e-12:
        return None
    return float((target - b) / m)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--compare-script",
        type=Path,
        default=REPO_ROOT / "scripts/compare_cpg_v1_v2_kinematics.py",
        help="Path to compare_cpg_v1_v2_kinematics.py",
    )
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for all sweep artifacts")
    p.add_argument(
        "--gains",
        type=str,
        default="45,80,120",
        help="Comma-separated floats, or ``lin LO HI N`` (v2 --actuator-gain)",
    )
    p.add_argument(
        "--intrinsic-frequencies",
        type=str,
        default="12",
        help="Comma-separated or ``lin LO HI N`` (CPG Hz, both stacks)",
    )
    p.add_argument(
        "--coupling-strengths",
        type=str,
        default="10",
        help="Comma-separated or ``lin LO HI N`` (tripod coupling, both stacks)",
    )
    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pickle", type=Path, default=None)
    p.add_argument(
        "--with-mp4",
        action="store_true",
        help="Encode side-by-side MP4 each run (slow); default skips via compare --no-mp4",
    )
    p.add_argument(
        "--target-dx-v2-mm",
        type=float,
        default=None,
        help="If set, print a linear fit of v2 thorax Δx vs actuator gain and estimated gain",
    )
    p.add_argument(
        "--extra-compare-args",
        type=str,
        default="",
        help="Raw string appended to the compare subprocess (e.g. ``--spawn-z-mm 0.5``)",
    )
    args = p.parse_args()

    gains = _parse_float_grid(args.gains)
    freqs = _parse_float_grid(args.intrinsic_frequencies)
    coups = _parse_float_grid(args.coupling_strengths)
    if not gains or not freqs or not coups:
        sys.exit("gains, intrinsic-frequencies, and coupling-strengths must each expand to a non-empty list")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    compare = args.compare_script.resolve()
    if not compare.is_file():
        sys.exit(f"compare script not found: {compare}")

    summary_path = out_dir / "summary.csv"
    fieldnames = [
        "stem",
        "actuator_gain",
        "intrinsic_frequency_hz",
        "coupling_strength",
        "returncode",
        "v1_thorax_delta_x_mm",
        "v2_thorax_delta_x_mm",
        "v2_thorax_delta_y_mm",
        "v2_thorax_delta_z_mm",
        "v1_mean_rmse_cmd_vs_pos_rad",
        "v2_mean_rmse_cmd_vs_pos_rad",
        "v2_mean_L2_joint_vel",
        "v2_mean_contact_frac",
        "npz_path",
        "mp4_path",
        "report_path",
    ]
    rows: list[dict[str, object]] = []

    extra_tokens = args.extra_compare_args.split() if args.extra_compare_args.strip() else []

    for g in gains:
        for f_hz in freqs:
            for c in coups:
                stem = _stem(g, f_hz, c)
                cmd = [
                    sys.executable,
                    str(compare),
                    "--output-dir",
                    str(out_dir),
                    "--stem",
                    stem,
                    "--duration",
                    str(args.duration),
                    "--seed",
                    str(args.seed),
                    "--intrinsic-frequency",
                    str(f_hz),
                    "--coupling-strength",
                    str(c),
                    "--actuator-gain",
                    str(g),
                    *extra_tokens,
                ]
                if args.pickle is not None:
                    cmd += ["--pickle", str(args.pickle.resolve())]
                if not args.with_mp4:
                    cmd.append("--no-mp4")

                print("+", " ".join(cmd), flush=True)
                proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
                rc = int(proc.returncode)

                npz_path = out_dir / f"{stem}.npz"
                mp4_path = out_dir / f"{stem}_side_by_side.mp4"
                report_path = out_dir / f"{stem}_kinematics_compare.txt"

                row: dict[str, object] = {
                    "stem": stem,
                    "actuator_gain": g,
                    "intrinsic_frequency_hz": f_hz,
                    "coupling_strength": c,
                    "returncode": rc,
                    "npz_path": str(npz_path),
                    "mp4_path": str(mp4_path) if args.with_mp4 else "",
                    "report_path": str(report_path),
                }
                if rc == 0 and npz_path.is_file():
                    try:
                        m = _load_row_from_npz(npz_path)
                        row["v1_thorax_delta_x_mm"] = m["v1_thorax_delta_x_mm"]
                        row["v2_thorax_delta_x_mm"] = m["v2_thorax_delta_x_mm"]
                        row["v2_thorax_delta_y_mm"] = m["v2_thorax_delta_y_mm"]
                        row["v2_thorax_delta_z_mm"] = m["v2_thorax_delta_z_mm"]
                        row["v1_mean_rmse_cmd_vs_pos_rad"] = m["v1_mean_rmse_cmd_vs_pos_rad"]
                        row["v2_mean_rmse_cmd_vs_pos_rad"] = m["v2_mean_rmse_cmd_vs_pos_rad"]
                        row["v2_mean_L2_joint_vel"] = m["v2_mean_L2_joint_vel"]
                        row["v2_mean_contact_frac"] = m.get("v2_mean_contact_frac", "")
                    except Exception as exc:  # noqa: BLE001 — batch helper
                        row["v1_thorax_delta_x_mm"] = ""
                        row["v2_thorax_delta_x_mm"] = f"npz_error:{exc}"
                        row["v2_thorax_delta_y_mm"] = ""
                        row["v2_thorax_delta_z_mm"] = ""
                        row["v1_mean_rmse_cmd_vs_pos_rad"] = ""
                        row["v2_mean_rmse_cmd_vs_pos_rad"] = ""
                        row["v2_mean_L2_joint_vel"] = ""
                        row["v2_mean_contact_frac"] = ""
                else:
                    row["v1_thorax_delta_x_mm"] = ""
                    row["v2_thorax_delta_x_mm"] = ""
                    row["v2_thorax_delta_y_mm"] = ""
                    row["v2_thorax_delta_z_mm"] = ""
                    row["v1_mean_rmse_cmd_vs_pos_rad"] = ""
                    row["v2_mean_rmse_cmd_vs_pos_rad"] = ""
                    row["v2_mean_L2_joint_vel"] = ""
                    row["v2_mean_contact_frac"] = ""

                rows.append(row)

    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print("Wrote", summary_path)

    if args.target_dx_v2_mm is not None:
        ok_g: list[float] = []
        ok_dx: list[float] = []
        for row in rows:
            if row["returncode"] != 0:
                continue
            dx = row.get("v2_thorax_delta_x_mm")
            if isinstance(dx, (int, float)):
                ok_g.append(float(row["actuator_gain"]))
                ok_dx.append(float(dx))
        est = _linear_gain_for_target_dx(ok_g, ok_dx, args.target_dx_v2_mm)
        print(
            f"Linear fit v2 Δx vs gain over {len(ok_g)} successful runs: "
            f"target Δx = {args.target_dx_v2_mm} mm → estimated gain ≈ {est}"
            if est is not None
            else "Could not estimate gain (need ≥2 successful runs with numeric Δx and |slope| > 0)."
        )


if __name__ == "__main__":
    main()
