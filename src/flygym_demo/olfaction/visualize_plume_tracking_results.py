from __future__ import annotations

import gc
import pickle
from pathlib import Path

import numpy as np


def load_successful_trajectories(
    data_dir: str | Path,
    *,
    dimension_scale_factor: float = 0.5,
    success_radius: float = 15,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    import h5py

    data_dir = Path(data_dir)
    with h5py.File(data_dir / "plume_dataset/plume.hdf5", "r") as f:
        inflow_pos = f["inflow_pos"][:] / dimension_scale_factor

    trajectories = {}
    for file in data_dir.glob("sim_results/plume_navigation_*_controlFalse.pkl"):
        with open(file, "rb") as f:
            data = pickle.load(f)
        trajectories[file.stem] = np.array([obs["fly"][0, :2] for obs in data["obs_hist"]])
        del data
        gc.collect()

    successful = {}
    for trial, traj in trajectories.items():
        dist_to_target = np.linalg.norm(traj - inflow_pos, axis=1)
        dist_argmin = np.argmin(dist_to_target)
        if dist_to_target[dist_argmin] < success_radius:
            successful[trial] = traj[: dist_argmin + 1]
    return trajectories, successful, inflow_pos


def plot_successful_trajectories(
    data_dir: str | Path,
    trajectories: dict[str, np.ndarray],
    inflow_pos: np.ndarray,
    *,
    dimension_scale_factor: float = 0.5,
    success_radius: float = 15,
) -> None:
    import h5py
    import matplotlib.pyplot as plt

    data_dir = Path(data_dir)
    figs_dir = data_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    with h5py.File(data_dir / "plume_dataset/plume.hdf5", "r") as f:
        mean_intensity = np.mean(f["plume"], axis=0)

    arena_height = mean_intensity.shape[0] * dimension_scale_factor
    arena_width = mean_intensity.shape[1] * dimension_scale_factor
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(
        mean_intensity,
        origin="lower",
        cmap="Blues",
        extent=[0, arena_width, 0, arena_height],
        vmin=0,
        vmax=0.3,
    )
    ax.plot([inflow_pos[0]], [inflow_pos[1]], marker="o", markersize=5, color="black")
    thetas = np.linspace(0, 2 * np.pi, 100)
    ax.plot(
        inflow_pos[0] + success_radius * np.cos(thetas),
        inflow_pos[1] + success_radius * np.sin(thetas),
        color="black",
        linestyle="--",
        lw=1,
    )
    for traj in trajectories.values():
        ax.plot(traj[:, 0], traj[:, 1], lw=1)
    ax.set_xlim(0, arena_width)
    ax.set_ylim(0, arena_height)
    fig.savefig(figs_dir / "trajectory_plot.pdf", dpi=300)


def trim_video_by_fraction(input_file: Path, output_file: Path, fraction: float) -> None:
    import cv2

    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_file}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    frames_to_keep = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * fraction)

    out = cv2.VideoWriter(str(output_file), codec, fps, (frame_width, frame_height))
    for _ in range(frames_to_keep):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("./outputs/plume_tracking/"))
    parser.add_argument("--chosen-trial")
    args = parser.parse_args()

    all_trajectories, successful_trials, inflow = load_successful_trajectories(
        args.data_dir
    )
    with open(args.data_dir / "all_trajectories.pkl", "wb") as f:
        pickle.dump(all_trajectories, f)
    plot_successful_trajectories(args.data_dir, successful_trials, inflow)

    if args.chosen_trial is not None:
        trimmed_len = successful_trials[args.chosen_trial].shape[0]
        total_len = all_trajectories[args.chosen_trial].shape[0]
        trim_video_by_fraction(
            args.data_dir / f"sim_results/{args.chosen_trial}.mp4",
            args.data_dir / f"figs/{args.chosen_trial}_trimmed.mp4",
            trimmed_len / total_len,
        )
