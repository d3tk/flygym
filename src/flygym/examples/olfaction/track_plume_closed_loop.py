from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from sys import stderr

import numpy as np

from flygym.compose import OdorPlumeWorld
from flygym.examples.olfaction.common import add_world_camera, make_olfaction_fly
from flygym.examples.olfaction.plume_tracking_controller import (
    PlumeNavigationController,
)
from flygym.examples.olfaction.plume_tracking_task import PlumeNavigationTask
from flygym.utils.math import Rotation3D


def eprint(*args, **kwargs) -> None:
    print(*args, file=stderr, **kwargs)


def run_simulation(
    plume_dataset_path: str | Path,
    output_dir: str | Path,
    seed: int,
    initial_position: tuple[float, float] = (180, 80),
    live_display: bool = False,
    is_control: bool = False,
    run_time: float = 60,
) -> PlumeNavigationTask:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fly = make_olfaction_fly(draw_sensor_markers=True)
    main_camera_name = "birdeye_cam"
    arena = OdorPlumeWorld(plume_dataset_path, main_camera_name=main_camera_name)
    camera = add_world_camera(
        arena,
        main_camera_name,
        pos=(
            0.50 * arena.arena_size[0],
            0.15 * arena.arena_size[1],
            1.00 * arena.arena_size[1],
        ),
        euler=(np.deg2rad(15), 0, 0),
        fovy=60,
    )
    task = PlumeNavigationTask(
        fly=fly,
        arena=arena,
        cameras=[camera],
        spawn_position=(*initial_position, 0.8),
        spawn_rotation=Rotation3D("quat", (0, 0, 0, -1)),
    )
    if is_control:
        controller = PlumeNavigationController(
            dt=task.timestep,
            alpha=0,
            delta_lambda_sw=0,
            delta_lambda_ws=0,
            random_seed=seed,
        )
    else:
        controller = PlumeNavigationController(task.timestep, random_seed=seed)

    if live_display:
        import cv2

    encounter_threshold = 0.001
    obs_hist = []
    reward = 0.0

    for step_idx in range(int(run_time / task.timestep)):
        if step_idx % int(1 / task.timestep) == 0:
            sec = step_idx * task.timestep
            eprint(f"{datetime.now()} - seed {seed}: {sec:.1f} / {run_time:.1f} sec")

        obs = task.get_observation()
        _, dn_drive, _ = controller.decide_state(
            encounter_flag=obs["odor_intensity"].max() > encounter_threshold,
            fly_heading=obs["fly_orientation"],
        )
        obs, reward, terminated, truncated, _ = task.step(dn_drive)
        if terminated or truncated:
            break
        rendered = task.render()[0]
        if live_display and rendered is not None:
            cv2.imshow("rendered_img", rendered[:, :, ::-1])
            cv2.waitKey(1)
        obs_hist.append(obs)

    filename_stem = f"plume_navigation_seed{seed}_control{is_control}"
    task.renderer.save_video(output_dir / (filename_stem + ".mp4"))
    with open(output_dir / (filename_stem + ".pkl"), "wb") as f:
        pickle.dump({"obs_hist": obs_hist, "reward": reward}, f)
    return task


def process_trial(
    plume_dataset_path: str | Path,
    output_dir: str | Path,
    seed: int,
    initial_position: tuple[float, float],
    is_control: bool,
) -> None:
    run_simulation(
        plume_dataset_path,
        output_dir,
        seed,
        initial_position,
        is_control=is_control,
        live_display=False,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser
    from shutil import copy

    from joblib import Parallel, delayed

    parser = ArgumentParser()
    parser.add_argument(
        "--plume-dataset",
        type=Path,
        default=Path("./outputs/plume_tracking/plume_dataset/plume.hdf5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/plume_tracking/sim_results/"),
    )
    parser.add_argument("--n-jobs", type=int, default=-2)
    args = parser.parse_args()

    plume_dataset_shm_path = Path("/dev/shm/") / args.plume_dataset.name
    copy(args.plume_dataset, plume_dataset_shm_path)
    xx, yy = np.meshgrid(np.linspace(155, 200, 10), np.linspace(57.5, 102.5, 10))
    points = np.vstack((xx.flat, yy.flat)).T
    configs = [
        (plume_dataset_shm_path, args.output_dir, seed, initial_position, False)
        for seed, initial_position in enumerate(points)
    ]
    Parallel(n_jobs=args.n_jobs)(delayed(process_trial)(*config) for config in configs)
    plume_dataset_shm_path.unlink()
