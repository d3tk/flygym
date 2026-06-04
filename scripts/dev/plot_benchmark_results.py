from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

DEFAULT_DEVICES = ["rtx3080ti", "rtx4090", "l40s", "h100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GPU simulation benchmark scaling curves.")
    parser.add_argument(
        "--basedir",
        type=Path,
        default=Path("gpu_benchmark_results_all"),
        help="Directory containing gpu_benchmark_results_<device> subfolders.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=DEFAULT_DEVICES,
        help="Device suffix subdirectories to include (e.g. l40s local).",
    )
    parser.add_argument(
        "--simplify-geom",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot rows where simplify_geom matches this value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <basedir>/scaling_test_simplifygeom<flag>.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_res_basedir = args.basedir
    devices = args.devices
    simplify_geom = args.simplify_geom

    dfs = []
    for suffix in devices:
        dir_ = benchmark_res_basedir / f"gpu_benchmark_results_{suffix}"
        gpu_info_df = pd.read_csv(dir_ / "gpu_info.csv")
        gpu_name = gpu_info_df["name"][0]
        df = pd.read_csv(dir_ / "gpu_simulation_benchmark_results.csv")
        df["device"] = suffix
        df["device_fullname"] = gpu_name
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    for device in devices:
        df = df_all[df_all["device"] == device]
        df = df[df["simplify_geom"] == simplify_geom]
        if df.empty:
            continue
        device_fullname = df["device_fullname"].iloc[0]
        ax.plot(df["n_worlds"], df["realtime_factor"], marker="o", label=device_fullname)

    ax.set_xscale("log", base=2)

    x_values = sorted(df_all["n_worlds"].unique())
    ax.set_xticks(x_values)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    ax.set_xlabel("Number of parallel worlds (log scale)")
    ax.set_ylabel("Throughput (× realtime)")
    ax.set_title("GPU simulation scaling test")
    ax.legend()

    output_path = args.output or (
        benchmark_res_basedir / f"scaling_test_simplifygeom{simplify_geom}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
