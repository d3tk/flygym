"""Couple FlyGym eye readouts to a pretrained FlyVis network.

Importing this module imports FlyVis. FlyVis is not a dependency of FlyGym,
and ``import flygym`` does not load it. The verified combination is FlyVis
1.2.0 on Python 3.12.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

import flyvis
from flyvis.datasets.rendering import BoxEye
from flyvis.network import NetworkView
from flyvis.network.network import IntegrationWarning, Network
from flyvis.utils.activity_utils import LayerActivity
from flyvis.utils.chkpt_utils import get_from_state_dict
from flyvis.utils.tensor_utils import AutoDeref

from flygym.vision.retina import Retina


def _cell_type_name(cell_type: Any) -> str:
    if isinstance(cell_type, bytes | np.bytes_):
        return cell_type.decode()
    return str(cell_type)


class RealTimeVisionNetwork(Network):
    """FlyVis network that integrates one visual frame at a time.

    ``flyvis.Network`` consumes a whole stimulus at once. Closed-loop FlyGym
    steps need one frame per call, with the state kept between calls. The
    integrator is FlyVis's own ``_next_state``.
    """

    def setup_step_by_step_simulation(
        self,
        dt: float,
        initial_state: str | AutoDeref | None = "auto",
        as_states: bool = False,
        num_samples: int = 1,
    ) -> None:
        """Prepare a frame-by-frame simulation.

        Args:
            dt: Integration step of the visual network, in seconds. This is
                coarser than the physics step.
            initial_state: State to start from. ``"auto"`` is one second of
                gray input via ``steady_state``. ``None`` uses the dynamics'
                own initial state.
            as_states: When True, ``forward_one_step`` returns the FlyVis
                state. When False, it returns node activity.
            num_samples: Batch size. Use 2 for the two eyes.
        """
        if dt > 1 / 50:
            warnings.warn(
                f"dt={dt} is very large for integration. "
                "Choose a smaller dt (<= 1/50 to avoid this warning).",
                IntegrationWarning,
                stacklevel=2,
            )

        is_training = self.training
        self.training = False
        params_requiring_grad = {}
        for name, params in self.named_parameters():
            params_requiring_grad[name] = params.requires_grad
            params.requires_grad = False

        self.clamp()
        params = self._param_api()

        if initial_state is None:
            initial_state = self._initial_state(params=params, batch_size=num_samples)
        elif initial_state == "auto":
            initial_state = self.steady_state(t_pre=1.0, dt=dt, batch_size=num_samples)
        self._current_step_by_step_sim_state = initial_state

        layer_index = {}
        for cell_type, index in self.connectome.nodes.layer_index.items():
            values = index[:]
            layer_index[cell_type] = values
            layer_index[_cell_type_name(cell_type)] = values
        input_node_index = np.array(
            [
                layer_index[_cell_type_name(cell_type)]
                for cell_type in self.connectome.input_cell_types[:]
            ]
        )

        self._step_by_step_sim_params = {
            "dt": dt,
            "is_training": is_training,
            "params_requiring_grad": params_requiring_grad,
            "as_states": as_states,
            "sim_params": params,
            "num_nodes": len(self.connectome.nodes.type),
            "num_samples": num_samples,
            "input_node_index": input_node_index,
        }

    def cleanup_step_by_step_simulation(self) -> None:
        """Drop the step-by-step state and restore training flags."""
        self._check_step_by_step_simulation_setup()
        self.training = self._step_by_step_sim_params["is_training"]
        for name, params in self.named_parameters():
            params.requires_grad = self._step_by_step_sim_params[
                "params_requiring_grad"
            ][name]
        del self._step_by_step_sim_params
        del self._current_step_by_step_sim_state

    def forward_one_step(self, curr_visual_input: Tensor) -> AutoDeref | Tensor:
        """Integrate one frame.

        Args:
            curr_visual_input: Luminance per ommatidium, shape
                ``(num_samples, num_ommatidia)``. The same value is written
                onto every photoreceptor type.

        Returns:
            Node activity of shape ``(num_samples, n_cells)``, or the FlyVis
            state when ``as_states`` was set.
        """
        self._check_step_by_step_simulation_setup()
        stored = self._step_by_step_sim_params
        stimulus_buffer = torch.zeros(
            (stored["num_samples"], stored["num_nodes"]),
            device=flyvis.device,
        )
        curr_visual_input = curr_visual_input.to(stimulus_buffer.device)
        input_node_index = stored["input_node_index"]
        for i in range(stored["num_samples"]):
            stimulus_buffer[i, input_node_index] += curr_visual_input[i]

        self._current_step_by_step_sim_state = self._next_state(
            params=stored["sim_params"],
            state=self._current_step_by_step_sim_state,
            x_t=stimulus_buffer,
            dt=stored["dt"],
        )
        if stored["as_states"]:
            return self._current_step_by_step_sim_state
        return self._current_step_by_step_sim_state.nodes.activity

    def _check_step_by_step_simulation_setup(self) -> None:
        if not hasattr(self, "_step_by_step_sim_params"):
            raise RuntimeError(
                "Call setup_step_by_step_simulation() before forward_one_step() "
                "or cleanup_step_by_step_simulation()."
            )


class RealTimeVisionNetworkView(NetworkView):
    """Load a ``RealTimeVisionNetwork`` from a FlyVis network directory."""

    def __init__(self, network_dir: str | Path, **kwargs: Any) -> None:
        kwargs["network_class"] = RealTimeVisionNetwork
        super().__init__(network_dir, **kwargs)

    def init_network(
        self,
        checkpoint: str = "best_chkpt",
        network: RealTimeVisionNetwork | None = None,
    ) -> RealTimeVisionNetwork:
        """Construct the network and load ``best_chkpt``.

        Args:
            checkpoint: File name inside the network directory. The released
                pretrained model stores the trained weights in ``best_chkpt``.
                This does not use FlyVis's ``"best"`` selector, which picks a
                numbered checkpoint from the validation-loss index.
            network: Existing network to load into. A new one is built from
                the directory config when this is omitted.

        Returns:
            The network with the checkpoint loaded, in eval mode.
        """
        if network is None:
            config = self.dir.config.network.to_dict()
            network = RealTimeVisionNetwork(
                connectome=config["connectome"],
                dynamics=config["dynamics"],
                node_config=config["node_config"],
                edge_config=config["edge_config"],
            )
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(self.dir.path) / checkpoint
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"FlyVis checkpoint {checkpoint_path} does not exist."
            )
        state = get_from_state_dict(checkpoint_path, "network")
        if not state:
            raise RuntimeError(f"{checkpoint_path} has no network state.")
        network.load_state_dict(state)
        network.eval()
        self._initialized["network"] = True
        return network


class RetinaMapper:
    """Permute ommatidium values between FlyGym and FlyVis order.

    Both libraries use a hexagonal eye with 721 receptors (FlyVis extent 15).
    They number those receptors differently. The map is the v1 corner
    alignment: the top receptor of the leftmost column and the bottom
    receptor of the rightmost column are lined up, and the other receptors
    follow by the same row and column scale. Coordinates are truncated
    toward zero when they are used as image indices, matching that
    implementation.

    A value that survives ``flygym_to_flyvis`` and ``flyvis_to_flygym`` is
    back where it started. That shows the map is a permutation. It does not
    by itself show that a FlyGym ommatidium and a FlyVis column see the same
    direction in the world.
    """

    def __init__(
        self, retina: Retina | None = None, boxeye: BoxEye | None = None
    ) -> None:
        if retina is None:
            retina = Retina()
        if boxeye is None:
            boxeye = BoxEye(extent=15)

        # Top of the leftmost column is *a*. Bottom of the rightmost is *b*.
        receptor_centers = boxeye.receptor_centers.detach().cpu().numpy()
        columns = np.unique(receptor_centers[:, 1])
        col_min = columns[0]
        col_max = columns[-1]
        leftmost = receptor_centers[receptor_centers[:, 1] == col_min]
        rightmost = receptor_centers[receptor_centers[:, 1] == col_max]
        flyvis_a = np.array([leftmost[:, 0].min(), col_min])
        flyvis_b = np.array([rightmost[:, 0].max(), col_max])

        flygym_a = _ommatidium_center(retina, 1)
        flygym_b = _ommatidium_center(retina, int(retina.ommatidia_id_map.max()))

        row_k = (flygym_b[0] - flygym_a[0]) / (flyvis_b[0] - flyvis_a[0])
        row_b = flygym_a[0] - row_k * flyvis_a[0]
        col_k = (flygym_b[1] - flygym_a[1]) / (flyvis_b[1] - flyvis_a[1])
        col_b = flygym_a[1] - col_k * flyvis_a[1]

        rows = row_k * receptor_centers[:, 0] + row_b
        cols = col_k * receptor_centers[:, 1] + col_b
        # Truncate toward zero, which is what the v1 float-index lookup did.
        row_index = rows.astype(np.intp)
        col_index = cols.astype(np.intp)
        self._idx_flyvis_to_flygym = retina.ommatidia_id_map[row_index, col_index] - 1
        self._idx_flygym_to_flyvis = np.argsort(self._idx_flyvis_to_flygym)
        self.n_ommatidia = int(self._idx_flyvis_to_flygym.shape[0])

    def flygym_to_flyvis(self, flygym_stimulus: np.ndarray) -> np.ndarray:
        """Reorder the last axis from FlyGym ommatidium ids to FlyVis order."""
        return flygym_stimulus[..., self._idx_flyvis_to_flygym]

    def flyvis_to_flygym(self, flyvis_stimulus: np.ndarray) -> np.ndarray:
        """Reorder the last axis from FlyVis receptor order to FlyGym ids."""
        return flyvis_stimulus[..., self._idx_flygym_to_flyvis]


def _ommatidium_center(retina: Retina, ommatidium_id: int) -> np.ndarray:
    rows, cols = np.where(retina.ommatidia_id_map == ommatidium_id)
    if rows.size == 0:
        raise ValueError(f"Ommatidium id {ommatidium_id} is not in the retina map.")
    return np.array([rows.mean(), cols.mean()])


class FlyVisVision:
    """Run the pretrained FlyVis network on FlyGym ommatidium readouts.

    This replaces v1 ``RealisticVisionFly``. The caller owns the physics
    step and passes ``Simulation.get_ommatidia_readouts`` when a new visual
    frame is due.

    Call ``fly.add_vision()`` before building the simulation. After
    ``sim.reset()``, call ``sim.forward()`` before the first readout: the
    eye cameras stay at the origin until the derived positions are
    recomputed.

    ``docs/flyvis_verification.md`` records the checkpoint and flash-response
    check. That check feeds FlyVis its own stimulus. The lattice remap used
    here is covered by the round-trip test, which shows the permutation is
    invertible.

    Example:
        vision = FlyVisVision()  # dt defaults to 1/500, the v1 refresh rate
        activities = vision.reset(sim.get_ommatidia_readouts(fly.name))
        # activities["arr"] is (2, n_cells); activities["layers"] is LayerActivity
        if vision.due(sim.time):
            activities = vision.step(sim.get_ommatidia_readouts(fly.name))
        vision.close()

    Args:
        network_dir: FlyVis network directory. The default is
            ``flow/0000/000`` under ``flyvis.results_dir``.
        dt: Visual integration step in seconds. ``due`` uses this as the
            gap between frames. The default, 1/500, is v1's
            ``vision_refresh_rate``.
    """

    def __init__(
        self,
        network_dir: str | Path | None = None,
        dt: float = 1 / 500,
    ) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}.")
        if network_dir is None:
            network_dir = flyvis.results_dir / "flow/0000/000"
        view = RealTimeVisionNetworkView(network_dir)
        self.vision_network = view.init_network(checkpoint="best_chkpt")
        self.retina_mapper = RetinaMapper()
        self.dt = float(dt)
        self._initialized = False
        self._last_update: float | None = None
        self._seen_time: float | None = None

    def reset(self, readouts: np.ndarray) -> dict[str, Any]:
        """Fade in on this frame, then integrate it once.

        Args:
            readouts: ``(2, n_ommatidia, 2)`` yellow/pale readouts for the
                left and right eyes, as returned by
                ``Simulation.get_ommatidia_readouts``.

        Returns:
            ``{"arr": array, "layers": LayerActivity}``. ``arr`` has shape
            ``(2, n_cells)`` in left, right order.
        """
        if self._initialized:
            self.vision_network.cleanup_step_by_step_simulation()
            self._initialized = False
        visual_input = self._to_flyvis(readouts)
        initial_state = self.vision_network.fade_in_state(
            t_fade_in=1.0,
            dt=self.dt,
            initial_frames=visual_input.unsqueeze(1),
        )
        self.vision_network.setup_step_by_step_simulation(
            dt=self.dt,
            initial_state=initial_state,
            as_states=False,
            num_samples=2,
        )
        self._initialized = True
        self._last_update = None
        self._seen_time = None
        return self._activities_from(visual_input)

    def due(self, sim_time: float) -> bool:
        """Return whether the visual network should integrate this time.

        The first call anchors the clock and returns False, because
        ``reset`` already integrated that frame. Later calls return True
        once ``sim_time`` is at least ``dt`` past the last integrated frame.
        Pass ``Simulation.time``. Gate ``step`` with this so the network is
        not integrated on every physics step.
        """
        if not self._initialized:
            raise RuntimeError("Call reset() before due().")
        self._seen_time = float(sim_time)
        if self._last_update is None:
            self._last_update = self._seen_time
            return False
        return self._seen_time + 1e-6 >= self._last_update + self.dt

    def step(self, readouts: np.ndarray) -> dict[str, Any]:
        """Integrate one visual frame.

        Args:
            readouts: Same shape as ``reset``.

        Returns:
            The same dictionary shape as ``reset``.
        """
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")
        activities = self._activities_from(self._to_flyvis(readouts))
        if self._seen_time is None:
            base = 0.0 if self._last_update is None else self._last_update
            self._last_update = base + self.dt
        else:
            self._last_update = self._seen_time
        return activities

    def close(self) -> None:
        """Release the step-by-step state."""
        if self._initialized:
            self.vision_network.cleanup_step_by_step_simulation()
            self._initialized = False
        self._last_update = None
        self._seen_time = None

    def _to_flyvis(self, readouts: np.ndarray) -> Tensor:
        readouts = np.asarray(readouts)
        n_ommatidia = self.retina_mapper.n_ommatidia
        if readouts.shape != (2, n_ommatidia, 2):
            raise ValueError(
                f"Expected readouts of shape (2, {n_ommatidia}, 2), "
                f"got {readouts.shape}."
            )
        # Each ommatidium fills either the yellow or the pale channel.
        luminance = readouts.max(axis=-1)
        mapped = self.retina_mapper.flygym_to_flyvis(luminance)
        return torch.as_tensor(mapped, dtype=torch.float32, device=flyvis.device)

    def _activities_from(self, visual_input: Tensor) -> dict[str, Any]:
        activity = self.vision_network.forward_one_step(visual_input)
        array = activity.detach().cpu().numpy()
        layers = LayerActivity(
            array,
            self.vision_network.connectome,
            keepref=True,
            use_central=False,
        )
        return {"arr": array, "layers": layers}
