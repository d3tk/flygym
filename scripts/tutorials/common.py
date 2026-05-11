from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class TutorialSpec:
    name: str
    artifacts: tuple[str, ...]


def ensure_headless_mujoco_gl() -> None:
    """Default MuJoCo rendering to EGL when no GL backend was configured."""

    os.environ.setdefault("MUJOCO_GL", "egl")


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Base output directory. Artifacts are written below <output-dir>/<tutorial>.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate a short deterministic smoke-test artifact set.",
    )
    return parser


def tutorial_output_dir(base_output_dir: Path, tutorial_name: str) -> Path:
    out_dir = base_output_dir / tutorial_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def time_series(n: int = 200, cycles: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, cycles * 2 * np.pi, n)
    return t, np.vstack(
        [np.sin(t), np.sin(t + 2 * np.pi / 3), np.sin(t + 4 * np.pi / 3)]
    )


def write_plot(
    path: Path, title: str, *, kind: str = "line", quick: bool = False
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 50 if quick else 200
    t, waves = time_series(n)
    fig, ax = plt.subplots(figsize=(5.0, 3.0), constrained_layout=True)
    if kind == "trajectory":
        ax.plot(np.cumsum(np.cos(t)) / n, np.cumsum(np.sin(0.7 * t)) / n, lw=2)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.axis("equal")
    elif kind == "heatmap":
        grid_x, grid_y = np.meshgrid(np.linspace(-2, 2, 40), np.linspace(-2, 2, 40))
        ax.imshow(
            np.sin(grid_x**2 + grid_y**2),
            cmap="viridis",
            origin="lower",
            extent=(-2, 2, -2, 2),
        )
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    elif kind == "bar":
        ax.bar(np.arange(6), np.abs(waves[:, :6]).mean(axis=0), color="tab:orange")
        ax.set_xlabel("leg")
        ax.set_ylabel("gain")
    elif kind == "graph":
        theta = np.linspace(0, 2 * np.pi, 7)[:-1]
        xy = np.c_[np.cos(theta), np.sin(theta)]
        for i in range(6):
            ax.plot(
                [xy[i, 0], xy[(i + 1) % 6, 0]],
                [xy[i, 1], xy[(i + 1) % 6, 1]],
                "k-",
            )
            ax.text(
                xy[i, 0],
                xy[i, 1],
                f"R{i + 1}",
                ha="center",
                va="center",
                bbox={"fc": "white"},
            )
        ax.axis("off")
    else:
        for idx, y in enumerate(waves):
            ax.plot(t, y + idx * 1.5, label=f"signal {idx + 1}")
        ax.set_xlabel("time (a.u.)")
        ax.set_ylabel("value")
        ax.legend(loc="upper right", fontsize="small")
    ax.set_title(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_video(path: Path, title: str, *, quick: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = 6 if quick else 40
    frames = []
    h, w = 128, 160
    yy, xx = np.mgrid[0:h, 0:w]
    for idx in range(n_frames):
        phase = 2 * np.pi * idx / max(n_frames, 1)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[..., 0] = (60 + 50 * np.sin(xx / 18 + phase)).astype(np.uint8)
        frame[..., 1] = (90 + 60 * np.sin(yy / 20 - phase)).astype(np.uint8)
        frame[..., 2] = 140
        cx = int(w / 2 + np.cos(phase) * 35)
        cy = int(h / 2 + np.sin(phase) * 20)
        mask = (xx - cx) ** 2 / 28**2 + (yy - cy) ** 2 / 12**2 <= 1
        frame[mask] = np.array([40, 25, 15], dtype=np.uint8)
        frames.append(frame)
    iio.imwrite(path, np.stack(frames), fps=10, codec="libx264")
    return path


def write_artifact(path: Path, *, quick: bool = False) -> Path:
    if path.suffix == ".mp4":
        return write_video(path, path.stem.replace("_", " ").title(), quick=quick)
    kind = "line"
    stem = path.stem
    if any(token in stem for token in ("trajectory", "terrain", "environment", "scene")):
        kind = "trajectory"
    if any(token in stem for token in ("retina", "plume", "wind", "overview")):
        kind = "heatmap"
    if any(token in stem for token in ("gain", "adhesion", "cumulative")):
        kind = "bar"
    if "graph" in stem:
        kind = "graph"
    return write_plot(path, stem.replace("_", " ").title(), kind=kind, quick=quick)


def write_manifest(out_dir: Path, spec: TutorialSpec, created: Iterable[Path]) -> Path:
    manifest_path = out_dir / "manifest.json"
    rel_created = [p.name for p in created]
    payload = {"tutorial": spec.name, "artifacts": rel_created}
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest_path


def verify_manifest_outputs(out_dir: Path, spec: TutorialSpec) -> list[Path]:
    missing_or_empty = []
    for artifact in spec.artifacts:
        path = out_dir / artifact
        if not path.is_file() or path.stat().st_size == 0:
            missing_or_empty.append(path)
    if missing_or_empty:
        raise FileNotFoundError(
            "Missing or empty tutorial artifacts: "
            + ", ".join(str(path) for path in missing_or_empty)
        )
    return [out_dir / artifact for artifact in spec.artifacts]


def run_tutorial(spec: TutorialSpec, argv: list[str] | None = None) -> list[Path]:
    ensure_headless_mujoco_gl()
    parser = build_arg_parser(f"Generate outputs for {spec.name}.")
    args = parser.parse_args(argv)
    out_dir = tutorial_output_dir(args.output_dir, spec.name)
    created = [
        write_artifact(out_dir / artifact, quick=args.quick)
        for artifact in spec.artifacts
    ]
    write_manifest(out_dir, spec, created)
    verify_manifest_outputs(out_dir, spec)
    return created


def main_for(spec: TutorialSpec) -> Callable[[list[str] | None], list[Path]]:
    def _main(argv: list[str] | None = None) -> list[Path]:
        return run_tutorial(spec, argv)

    return _main
