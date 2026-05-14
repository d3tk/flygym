from __future__ import annotations

from pathlib import Path

import numpy as np


def _flow():
    from phi.torch import flow

    return flow


def step(
    velocity_prev,
    smoke_prev,
    noise: np.ndarray,
    noise_magnitude: tuple[float, float] = (0.1, 2),
    dt: float = 1.0,
    inflow=None,
):
    flow = _flow()
    smoke_next = flow.advect.mac_cormack(smoke_prev, velocity_prev, dt=dt) + inflow
    external_force = smoke_next * noise * noise_magnitude @ velocity_prev
    velocity_tentative = (
        flow.advect.semi_lagrangian(velocity_prev, velocity_prev, dt=dt)
        + external_force
    )
    velocity_next, _ = flow.fluid.make_incompressible(velocity_tentative)
    return velocity_next, smoke_next


def converging_brownian_step(
    value_curr: np.ndarray,
    center: np.ndarray,
    gaussian_scale: float | tuple[float, float] = 1.0,
    convergence: float = 0.5,
) -> np.ndarray:
    gaussian_center = (center - value_curr) * convergence
    value_diff = np.random.normal(
        loc=gaussian_center, scale=gaussian_scale, size=value_curr.shape
    )
    return value_curr + value_diff


def get_simulation_parameters(simulation_time: int):
    dt = 0.7
    arena_size = (120, 80)
    inflow_pos = (5, 40)
    inflow_radius = 1
    inflow_scaler = 0.2
    velocity_grid_size = 0.5
    smoke_grid_size = 0.25
    simulation_steps = int(simulation_time / dt)
    return (
        dt,
        arena_size,
        inflow_pos,
        inflow_radius,
        inflow_scaler,
        velocity_grid_size,
        smoke_grid_size,
        simulation_steps,
    )


def generate_simulation_inputs(
    simulation_steps: int,
    arena_size: tuple[int, int],
    inflow_pos: tuple[int, int],
    inflow_radius: float,
    inflow_scaler: float,
    velocity_grid_size: float,
    smoke_grid_size: float,
):
    flow = _flow()

    curr_wind = np.zeros(2)
    wind_hist = [curr_wind.copy()]
    for _ in range(simulation_steps - 1):
        curr_wind = converging_brownian_step(curr_wind, (0, 0), (0.2, 0.2), 1.0)
        wind_hist.append(curr_wind.copy())

    velocity = flow.StaggeredGrid(
        values=(1.0, 0.0),
        extrapolation=flow.extrapolation.BOUNDARY,
        x=int(arena_size[0] / velocity_grid_size),
        y=int(arena_size[1] / velocity_grid_size),
        bounds=flow.Box(x=arena_size[0], y=arena_size[1]),
    )
    smoke = flow.CenteredGrid(
        values=0.0,
        extrapolation=flow.extrapolation.BOUNDARY,
        x=int(arena_size[0] / smoke_grid_size),
        y=int(arena_size[1] / smoke_grid_size),
        bounds=flow.Box(x=arena_size[0], y=arena_size[1]),
    )
    inflow = inflow_scaler * flow.field.resample(
        flow.Sphere(x=inflow_pos[0], y=inflow_pos[1], radius=inflow_radius),
        to=smoke,
        soft=True,
    )
    return wind_hist, velocity, smoke, inflow


def run_simulation(
    wind_hist,
    velocity,
    smoke,
    inflow,
    dt: float,
    arena_size: tuple[float, float],
    plot: bool = False,
):
    if plot:
        import matplotlib.pyplot as plt

    smoke_hist = []
    for wind in wind_hist:
        velocity, smoke = step(velocity, smoke, wind, dt=dt, inflow=inflow)
        smoke_vals = smoke.values.numpy("y,x")
        smoke_hist.append(smoke_vals)

        if plot:
            plt.imshow(
                smoke_vals,
                cmap="gray_r",
                origin="lower",
                vmin=0,
                vmax=0.7,
                extent=(0, arena_size[0], 0, arena_size[1]),
            )
            plt.gca().invert_yaxis()
            plt.draw()
            plt.pause(0.01)
            plt.clf()
    return smoke_hist


def save_simulation_outputs(
    wind_hist,
    smoke_hist,
    arena_size: tuple[int, int],
    output_dir: Path,
    inflow_pos: tuple[int, int],
    inflow_radius: float,
    inflow_scaler: float,
) -> None:
    import h5py
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    output_dir.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), tight_layout=True)
    ax.plot(wind_hist, label=("x", "y"))
    ax.legend()
    ax.set_xlabel("Time [AU]")
    ax.set_ylabel("Wind [AU]")
    ax.set_title('Brownian "wind"')
    fig.savefig(output_dir / "brownian_wind.png")

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), tight_layout=True)
    img = ax.imshow(
        smoke_hist[0],
        cmap="gray_r",
        origin="lower",
        vmin=0,
        vmax=0.7,
        extent=(0, arena_size[0], 0, arena_size[1]),
    )
    ax.invert_yaxis()

    def update(i):
        img.set_data(smoke_hist[i])

    animation = FuncAnimation(fig, update, frames=len(smoke_hist), repeat=False)
    animation.save(output_dir / "plume.mp4", fps=100, dpi=300, bitrate=500)

    with h5py.File(output_dir / "plume.hdf5", "w") as f:
        f.create_dataset(
            "plume", data=np.stack(smoke_hist).astype(np.float16), compression="gzip"
        )
        f["inflow_pos"] = inflow_pos
        f["inflow_radius"] = [inflow_radius]
        f["inflow_scaler"] = [inflow_scaler]

    with h5py.File(output_dir / "plume_short.hdf5", "w") as f:
        f.create_dataset(
            "plume",
            data=np.stack(smoke_hist[5000:5600:10]).astype(np.float16),
            compression="gzip",
        )
        f["inflow_pos"] = inflow_pos
        f["inflow_radius"] = [inflow_radius]
        f["inflow_scaler"] = [inflow_scaler]


if __name__ == "__main__":
    np.random.seed(0)
    output = Path("./outputs/plume_tracking/plume_dataset/")
    simulation_time = 13000
    params = get_simulation_parameters(simulation_time)
    (
        dt,
        arena_size,
        inflow_pos,
        inflow_radius,
        inflow_scaler,
        velocity_grid_size,
        smoke_grid_size,
        simulation_steps,
    ) = params
    wind_hist, velocity, smoke, inflow = generate_simulation_inputs(
        simulation_steps,
        arena_size,
        inflow_pos,
        inflow_radius,
        inflow_scaler,
        velocity_grid_size,
        smoke_grid_size,
    )
    smoke_hist = run_simulation(wind_hist, velocity, smoke, inflow, dt, arena_size)
    save_simulation_outputs(
        wind_hist,
        smoke_hist,
        arena_size,
        output,
        inflow_pos,
        inflow_radius,
        inflow_scaler,
    )
