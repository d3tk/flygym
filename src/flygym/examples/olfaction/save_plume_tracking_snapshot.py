from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from flygym.examples.olfaction.track_plume_closed_loop import run_simulation


def save_snapshot(
    plume_dataset_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 12,
    run_time: float = 0.8,
) -> None:
    import cv2

    xx, yy = np.meshgrid(np.linspace(155, 200, 10), np.linspace(57.5, 102.5, 10))
    initial_position = np.vstack((xx.flat, yy.flat)).T[seed]

    with TemporaryDirectory() as output_dir:
        task = run_simulation(
            plume_dataset_path,
            output_dir,
            seed=seed,
            initial_position=initial_position,
            is_control=False,
            live_display=False,
            run_time=run_time,
        )
        img = next(iter(task.renderer.frames.values()))[-1]
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--plume-dataset",
        type=Path,
        default=Path("./outputs/plume_tracking/plume_dataset/plume.hdf5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./outputs/plume_tracking/figs/snapshot.png"),
    )
    args = parser.parse_args()
    save_snapshot(args.plume_dataset, args.output)
