from abc import ABC, abstractmethod
from collections.abc import Callable
from collections import defaultdict
from os import PathLike
from typing import Any, override

import mujoco as mj
import dm_control.mjcf as mjcf
import numpy as np

from flygym.anatomy import ContactBodiesPreset, BodySegment, LEG_LINKS
from flygym.compose.base import BaseCompositionElement
from flygym.compose.fly import Fly
from flygym.compose.physics import ContactParams
from flygym.utils.math import Rotation3D, Vec3
from flygym.utils.exceptions import FlyGymInternalError

__all__ = [
    "BaseWorld",
    "FlatGroundWorld",
    "GappedTerrainWorld",
    "BlocksTerrainWorld",
    "MixedTerrainWorld",
    "OdorPlumeWorld",
    "OdorWorld",
    "TetheredWorld",
]


_STATE_DIM_BY_JOINT_TYPE = {"free": 7, "ball": 4, "hinge": 1, "slide": 1}
_DEFAULT_COLOR_CYCLE_RGB = np.array(
    [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
    ],
    dtype=float,
)


class BaseWorld(BaseCompositionElement, ABC):
    """Base class for worlds that contain environmental features that the fly can
    interact with (e.g., ground) and define how flies are attached to the world (e.g.,
    free-floating or tethered). A world can contain multiple flies that can interact
    with one another.

    Concrete subclasses typically override `__init__` to set up environmental features
    (e.g., ground plane) and `_attach_fly_mjcf` to define how flies are attached. See
    method documentation below for details.

    Attributes:
        name:
            Name of the world.
        fly_lookup:
            A dictionary mapping fly names to `Fly` objects in the world.
        mjcf_root:
            The root element of the world's MJCF model (fly MJCF models are attached to
            this root).
        world_dof_neutral_states:
            A dictionary mapping names of DoFs managed by the world (e.g., free joints
            by which flies are attached to the world) to their neutral state values.
            The neutral state is 1D for slide and hinge joints, 4D for ball joints
            (quaternion), and 7D for free joints (position + orientation).
    """

    def __init__(self, name: str) -> None:
        """Initialize the world and its underlying MJCF model.

        Concrete subclasses should call this first (i.e., `super().__init__(name)`) as
        it sets up a few essential attributes.
        """
        self._mjcf_root = mjcf.RootElement(model=name)
        self._fly_lookup: dict[str, Fly] = {}
        self.ground_geoms: list = []
        self.world_dof_neutral_states = {}
        self._neutral_keyframe = self.mjcf_root.keyframe.add(
            "key", name="neutral", time=0
        )
        self._add_skybox()

    @override
    @property
    def mjcf_root(self) -> mjcf.RootElement:
        return self._mjcf_root

    @property
    def fly_lookup(self) -> dict[str, Fly]:
        """Lookup for `Fly` objects in the world, keyed by fly name."""
        return self._fly_lookup

    @abstractmethod
    def _attach_fly_mjcf(
        self,
        fly: Fly,
        spawn_position: Vec3,
        spawn_rotation: Rotation3D,
        *args,
        **kwargs,
    ) -> mjcf.Element:
        """Attach the fly's MJCF root to the world MJCF model.

        Concrete subclasses should implement this method instead of overriding
        `add_fly()` directly. The `add_fly()` method handles registering the fly under
        `fly_lookup` and updating neutral states; this method is responsible only for
        connecting the fly's MJCF model to the world's MJCF model.

        Use `dm_control.mjcf`'s `attach()` method to attach the fly's MJCF model. See
        `FlatGroundWorld` for an example. More details can be found in the
        [`dm_control.mjcf` documentation](https://github.com/google-deepmind/dm_control/tree/main/dm_control/mjcf#attaching-models).

        Returns:
            The free joint element created by the attachment.
        """
        pass

    def _add_skybox(self):
        self.mjcf_root.asset.add(
            "texture",
            name="skybox",
            type="skybox",
            builtin="gradient",
            rgb1=(1, 1, 1),
            rgb2=(1, 1, 1),
            width=10,
            height=10,
        )

    def add_fly(
        self,
        fly: Fly,
        spawn_position: Vec3,
        spawn_rotation: Rotation3D,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Attach a fly to the world at the specified pose.

        The fly's MJCF model is merged into the world and registered under
        `fly_lookup`. Extra keyword arguments are forwarded to the subclass
        `_attach_fly_mjcf` implementation (see the specific world subclass for
        available options).

        Args:
            fly: The fly to add.
            spawn_position: Initial ``(x, y, z)`` position in mm.
            spawn_rotation: Initial orientation as a `Rotation3D` in quaternion format.
            *args: Forwarded to `_attach_fly_mjcf`.
            **kwargs: Forwarded to `_attach_fly_mjcf`.

        Raises:
            ValueError: If a fly with the same name already exists in the world.
            ValueError: If ``spawn_rotation`` is not in quaternion format.
        """
        # Register fly in the fly lookup
        if fly.name in self._fly_lookup:
            raise ValueError(f"Fly with name '{fly.name}' already exists in the world.")
        self._fly_lookup[fly.name] = fly

        # Remove neutral keyframes that are already generated by the fly. Neutral states
        # are globally managed at the world level. A single neutral keyframe will be
        # managed by the world from now on.
        neutral_keyframe = fly.mjcf_root.keyframe.find("key", "neutral")
        if neutral_keyframe is not None:
            neutral_keyframe.remove()

        # Attach the fly's MJCF root to the world MJCF model with a free joint.
        # This is an abstract method that must be implemented by concrete world classes.
        freejoint = self._attach_fly_mjcf(
            fly, spawn_position, spawn_rotation, *args, **kwargs
        )

        # Set neutral state for the freejoint attaching the fly to the world
        # (freejoint state is in [x, y, z, qw, qx, qy, qz] format)
        if spawn_rotation.format != "quat":
            raise ValueError(
                "Freejoint neutral rotation can only be specified in quaternion format "
                f"for now. Got {spawn_rotation}."
            )
        neutral_state = [*spawn_position, *spawn_rotation.values]
        self.world_dof_neutral_states[freejoint.full_identifier] = neutral_state

        self._rebuild_neutral_keyframe()

    @property
    def odor_dimensions(self) -> int:
        """Dimension of the odor signal provided by this world."""
        return 0

    def get_olfaction(
        self,
        sensor_positions: np.ndarray,
        *,
        time: float | None = None,
    ) -> np.ndarray:
        """Get odor intensity at olfactory sensor positions."""
        return np.zeros((self.odor_dimensions, sensor_positions.shape[0]))

    def _rebuild_neutral_keyframe(self):
        mj_model, _ = self.compile()
        neutral_qpos = np.zeros(mj_model.nq)
        neutral_ctrl = np.zeros(mj_model.nu)

        # Step 1: set neutral qpos for DoFs created by the world
        # dm_control.mjcf has trouble finding freejoints by name with
        # .find("joint", freejoint_name), but they do show up in the list of all joints
        # obtained with .find_all("joint"). So we build a mapping manually in order to
        # set the neutral pose for freejoints corresponding to fly spawns.
        all_world_joints = {
            j.full_identifier: j for j in self.mjcf_root.find_all("joint")
        }
        for joint_name, neutral_state in self.world_dof_neutral_states.items():
            joint_element = all_world_joints.get(joint_name)
            if joint_element is None:
                raise RuntimeError(
                    f"Joint '{joint_name}' not found when rebuilding neutral keyframe."
                )
            joint_type = (
                "free" if joint_element.tag == "freejoint" else joint_element.type
            )
            internal_jointid = mj.mj_name2id(
                mj_model, mj.mjtObj.mjOBJ_JOINT, joint_element.full_identifier
            )
            dofadr_start = mj_model.jnt_dofadr[internal_jointid]
            dofadr_end = dofadr_start + _STATE_DIM_BY_JOINT_TYPE[joint_type]
            neutral_qpos[dofadr_start:dofadr_end] = neutral_state

        # Step 2: handle joints and actuators belonging to flies attached to the world
        for fly_name, fly in self.fly_lookup.items():
            # Copy neutral joint angles from fly
            qpos_filled_by_fly = fly._get_neutral_qpos(mj_model)
            indices_to_fill = qpos_filled_by_fly.nonzero()
            has_conflict = np.any(~np.isclose(neutral_qpos[indices_to_fill], 0))
            if has_conflict:
                raise FlyGymInternalError(
                    f"Conflict in neutral joint angles: fly '{fly_name}' is trying "
                    "to set neutral qpos values for DoFs that already have their "
                    "neutral qpos set."
                )
            neutral_qpos[indices_to_fill] = qpos_filled_by_fly[indices_to_fill]

            # Copy neutral actuator inputs from fly
            ctrl_filled_by_fly = fly._get_neutral_ctrl(mj_model)
            indices_to_fill = ctrl_filled_by_fly.nonzero()
            has_conflict = np.any(~np.isclose(neutral_ctrl[indices_to_fill], 0))
            if has_conflict:
                raise FlyGymInternalError(
                    f"Conflict in neutral actuator inputs: fly '{fly_name}' is trying "
                    "to set neutral ctrl values for actuators that already have their "
                    "neutral ctrl set."
                )
            neutral_ctrl[indices_to_fill] = ctrl_filled_by_fly[indices_to_fill]

        self._neutral_keyframe.qpos = neutral_qpos
        self._neutral_keyframe.ctrl = neutral_ctrl


class FlatGroundWorld(BaseWorld):
    """World with a flat infinite ground plane. Flies are free to move.

    When calling `add_fly`, the following extra keyword arguments are accepted:

    - ``bodysegs_with_ground_contact``: Body segments that collide with the ground.
      Accepts a `ContactBodiesPreset`, a preset string, or a collection of
      `BodySegment` objects. Default: ``ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD``.
    - ``ground_contact_params``: `ContactParams` for friction and contact physics.
      Default: ``ContactParams()``.
    - ``add_ground_contact_sensors``: If True, add contact force sensors for each leg.
      Default: ``True``.

    Args:
        name: Name of the world.
        half_size: Half-size of the ground plane in mm.
    """

    @override
    def __init__(
        self, name: str = "flat_ground_world", *, half_size: float = 1000
    ) -> None:
        super().__init__(name=name)

        checker_texture = self.mjcf_root.asset.add(
            "texture",
            name="checker",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.3, 0.3, 0.3),
            rgb2=(0.4, 0.4, 0.4),
        )
        grid_material = self.mjcf_root.asset.add(
            "material",
            name="grid",
            texture=checker_texture,
            texrepeat=(250, 250),
            reflectance=0.2,
        )
        self.ground_geom = self.mjcf_root.worldbody.add(
            "geom",
            type="plane",
            name="ground_plane",
            material=grid_material,
            pos=(0, 0, 0),
            size=(half_size, half_size, 1),
            contype=0,
            conaffinity=0,
        )
        self.ground_geoms = [self.ground_geom]
        self.legpos_to_groundcontactsensors_by_fly = None

    @override
    def _attach_fly_mjcf(
        self,
        fly: Fly,
        spawn_position: Vec3,
        spawn_rotation: Rotation3D,
        *,
        bodysegs_with_ground_contact: (
            list[BodySegment] | ContactBodiesPreset | str
        ) = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
        ground_contact_params: ContactParams = ContactParams(),
        add_ground_contact_sensors: bool = True,
    ) -> mjcf.Element:
        spawn_site = self.mjcf_root.worldbody.add(
            "site", name=fly.name, pos=spawn_position, **spawn_rotation.as_kwargs()
        )
        freejoint = spawn_site.attach(fly.mjcf_root).add("freejoint", name=fly.name)

        if isinstance(bodysegs_with_ground_contact, ContactBodiesPreset | str):
            preset = ContactBodiesPreset(bodysegs_with_ground_contact)
            bodysegs_with_ground_contact = preset.to_body_segments_list()

        self._set_ground_contact(
            fly, bodysegs_with_ground_contact, ground_contact_params
        )
        if add_ground_contact_sensors:
            self._add_ground_contact_sensors(fly, bodysegs_with_ground_contact)
        return freejoint

    def _set_ground_contact(
        self,
        fly: Fly,
        bodysegs_with_ground_contact: list[BodySegment],
        ground_contact_params: ContactParams,
    ) -> None:
        for body_segment in bodysegs_with_ground_contact:
            body_geom = fly.mjcf_root.find("geom", f"{body_segment.name}")
            for ground_geom in self.ground_geoms:
                self.mjcf_root.contact.add(
                    "pair",
                    geom1=body_geom,
                    geom2=ground_geom,
                    name=f"{body_segment.name}-{ground_geom.name}-ground",
                    friction=ground_contact_params.get_friction_tuple(),
                    solref=ground_contact_params.get_solref_tuple(),
                    solimp=ground_contact_params.get_solimp_tuple(),
                    margin=ground_contact_params.margin,
                )

    def _add_ground_contact_sensors(
        self, fly: Fly, bodysegs_with_ground_contact: list[BodySegment]
    ) -> None:
        if len(self.ground_geoms) != 1:
            self.legpos_to_groundcontactsensors_by_fly = None
            return

        self.legpos_to_groundcontactsensors_by_fly = defaultdict(dict)
        contact_geoms_by_leg = defaultdict(list)
        for bodyseg in bodysegs_with_ground_contact:
            if bodyseg.is_leg():
                contact_geoms_by_leg[bodyseg.pos].append(bodyseg)
        for leg, contact_geoms in contact_geoms_by_leg.items():
            subtree_rootseg = _sort_legsegs_prox2dist(contact_geoms)[0]
            subtree_rootseg_body = fly.bodyseg_to_mjcfbody[subtree_rootseg]
            sensor = self.mjcf_root.sensor.add(
                "contact",
                subtree1=subtree_rootseg_body,
                geom2=self.ground_geoms[0],
                num=1,
                reduce="netforce",
                data="found force torque pos normal tangent",
                name=f"ground_contact_{leg}_leg",
            )
            self.legpos_to_groundcontactsensors_by_fly[fly.name][leg] = sensor


def _inverse_square(distance: np.ndarray) -> np.ndarray:
    return distance**-2


class OdorWorld(FlatGroundWorld):
    """Flat world with static odor sources."""

    def __init__(
        self,
        name: str = "odor_world",
        *,
        half_size: float = 1000,
        odor_source: np.ndarray | None = None,
        peak_odor_intensity: np.ndarray | None = None,
        diffuse_func: Callable[[np.ndarray], np.ndarray] = _inverse_square,
        marker_colors: np.ndarray | None = None,
        marker_size: float = 0.25,
    ) -> None:
        super().__init__(name=name, half_size=half_size)

        if odor_source is None:
            odor_source = np.array([[10, 0, 0]], dtype=float)
        else:
            odor_source = np.asarray(odor_source, dtype=float)
        if peak_odor_intensity is None:
            peak_odor_intensity = np.array([[1]], dtype=float)
        else:
            peak_odor_intensity = np.asarray(peak_odor_intensity, dtype=float)

        if odor_source.ndim != 2 or odor_source.shape[1] != 3:
            raise ValueError("odor_source must have shape (n_sources, 3).")
        if peak_odor_intensity.ndim != 2:
            raise ValueError(
                "peak_odor_intensity must have shape "
                "(n_sources, odor_dimensions)."
            )
        if odor_source.shape[0] != peak_odor_intensity.shape[0]:
            raise ValueError(
                "Number of odor source locations and peak intensities must match."
            )

        self.odor_source = odor_source
        self.peak_odor_intensity = peak_odor_intensity
        self.diffuse_func = diffuse_func
        self.num_odor_sources = odor_source.shape[0]

        if marker_colors is None:
            color_idx = np.arange(self.num_odor_sources) % len(_DEFAULT_COLOR_CYCLE_RGB)
            marker_colors = np.column_stack(
                (_DEFAULT_COLOR_CYCLE_RGB[color_idx] / 255, np.ones(self.num_odor_sources))
            )
        marker_colors = np.asarray(marker_colors, dtype=float)
        if marker_colors.shape != (self.num_odor_sources, 4):
            raise ValueError("marker_colors must have shape (n_sources, 4).")

        for idx, (pos, rgba) in enumerate(zip(self.odor_source, marker_colors)):
            marker_body = self.mjcf_root.worldbody.add(
                "body",
                name=f"odor_source_marker_{idx}",
                pos=pos,
                mocap=True,
            )
            marker_body.add(
                "geom",
                type="capsule",
                size=(marker_size, marker_size),
                rgba=rgba,
                contype=0,
                conaffinity=0,
            )

    @override
    def get_olfaction(
        self,
        sensor_positions: np.ndarray,
        *,
        time: float | None = None,
    ) -> np.ndarray:
        n_sensors = sensor_positions.shape[0]
        odor_sources = np.repeat(
            np.repeat(
                self.odor_source[:, np.newaxis, np.newaxis, :],
                self.odor_dimensions,
                axis=1,
            ),
            n_sensors,
            axis=2,
        )
        peak_intensity = np.repeat(
            self.peak_odor_intensity[:, :, np.newaxis],
            n_sensors,
            axis=2,
        )
        sensor_positions = sensor_positions[np.newaxis, np.newaxis, :, :]
        distance = np.linalg.norm(sensor_positions - odor_sources, axis=3)
        return (peak_intensity * self.diffuse_func(distance)).sum(axis=0)

    @property
    @override
    def odor_dimensions(self) -> int:
        return self.peak_odor_intensity.shape[1]


class OdorPlumeWorld(FlatGroundWorld):
    """Flat world backed by a time-varying HDF5 odor plume."""

    def __init__(
        self,
        plume_data_path: str | PathLike,
        main_camera_name: str = "",
        *,
        name: str = "odor_plume_world",
        dimension_scale_factor: float = 0.5,
        plume_simulation_fps: float = 200,
        intensity_scale_factor: float = 1.0,
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        num_sensors: int = 4,
    ) -> None:
        import h5py

        self.plume_dataset = h5py.File(plume_data_path, "r")
        self.plume_grid = self.plume_dataset["plume"]
        self.arena_size = (
            np.array(self.plume_grid.shape[1:][::-1]) * dimension_scale_factor
        )
        self.dimension_scale_factor = dimension_scale_factor
        self.plume_simulation_fps = plume_simulation_fps
        self.intensity_scale_factor = intensity_scale_factor
        self.num_sensors = num_sensors
        self.main_camera_name = main_camera_name
        self.plume_update_interval = 1 / plume_simulation_fps

        super().__init__(name=name, half_size=float(np.max(self.arena_size)) * 2)
        self.ground_geom.friction = friction

    @override
    def get_olfaction(
        self,
        sensor_positions: np.ndarray,
        *,
        time: float | None = None,
    ) -> np.ndarray:
        if sensor_positions.shape[0] != self.num_sensors:
            raise ValueError(
                f"Expected {self.num_sensors} olfactory sensors, "
                f"got {sensor_positions.shape[0]}."
            )
        if time is None:
            time = 0.0
        frame_num = int(time * self.plume_simulation_fps)
        intensities = np.zeros((self.odor_dimensions, self.num_sensors))

        for i_sensor, (x_mm, y_mm, _) in enumerate(sensor_positions):
            x_idx = int(x_mm / self.dimension_scale_factor)
            y_idx = int(y_mm / self.dimension_scale_factor)
            if (
                x_idx < 0
                or y_idx < 0
                or x_idx >= self.plume_grid.shape[2]
                or y_idx >= self.plume_grid.shape[1]
            ):
                intensities[0, i_sensor] = np.nan
            else:
                intensities[0, i_sensor] = self.plume_grid[frame_num, y_idx, x_idx]

        return intensities * self.intensity_scale_factor

    @property
    @override
    def odor_dimensions(self) -> int:
        return 1

    def close(self) -> None:
        self.plume_dataset.close()


class _ComplexTerrainWorld(FlatGroundWorld):
    def __init__(self, name: str) -> None:
        BaseWorld.__init__(self, name=name)
        self.ground_geoms = []
        self.legpos_to_groundcontactsensors_by_fly = None

    def _add_ground_box(
        self,
        name: str,
        size: tuple[float, float, float],
        pos: tuple[float, float, float],
        *,
        rgba: tuple[float, float, float, float],
    ) -> mjcf.Element:
        geom = self.mjcf_root.worldbody.add(
            "geom",
            type="box",
            name=name,
            size=size,
            pos=pos,
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )
        self.ground_geoms.append(geom)
        return geom

    def _add_ground_plane(
        self,
        name: str,
        size: tuple[float, float, float],
        pos: tuple[float, float, float],
        *,
        rgba: tuple[float, float, float, float],
    ) -> mjcf.Element:
        geom = self.mjcf_root.worldbody.add(
            "geom",
            type="plane",
            name=name,
            size=size,
            pos=pos,
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )
        self.ground_geoms.append(geom)
        return geom


class GappedTerrainWorld(_ComplexTerrainWorld):
    """World with alternating floor blocks and transverse gaps."""

    def __init__(
        self,
        name: str = "gapped_terrain_world",
        *,
        x_range: tuple[float, float] = (-10, 25),
        y_range: tuple[float, float] = (-20, 20),
        gap_width: float = 0.3,
        block_width: float = 1.0,
        gap_depth: float = 2.0,
        ground_alpha: float = 1.0,
    ) -> None:
        super().__init__(name=name)
        y_halfwidth = (y_range[1] - y_range[0]) / 2
        block_centers = np.arange(
            x_range[0] + block_width / 2,
            x_range[1],
            block_width + gap_width,
        )
        for x_pos in block_centers:
            self._add_ground_box(
                name=f"ground_block_x{_format_name_number(x_pos)}",
                size=(block_width / 2, y_halfwidth, gap_depth / 2),
                pos=(x_pos, 0, -gap_depth / 2),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )
        self._add_ground_plane(
            name="ground_base",
            size=((x_range[1] - x_range[0]) / 2, y_halfwidth, 1),
            pos=(np.mean(x_range), 0, -gap_depth),
            rgba=(0.3, 0.3, 0.3, ground_alpha),
        )


class BlocksTerrainWorld(_ComplexTerrainWorld):
    """World tiled by blocks with alternating heights."""

    def __init__(
        self,
        name: str = "blocks_terrain_world",
        *,
        x_range: tuple[float, float] = (-10, 25),
        y_range: tuple[float, float] = (-20, 20),
        block_size: float = 1.3,
        height_range: tuple[float, float] = (0.35, 0.35),
        ground_alpha: float = 1.0,
        rand_seed: int = 0,
    ) -> None:
        super().__init__(name=name)
        rand_state = np.random.RandomState(rand_seed)
        x_centers = np.arange(x_range[0] + block_size / 2, x_range[1], block_size)
        y_centers = np.arange(y_range[0] + block_size / 2, y_range[1], block_size)
        for i, x_pos in enumerate(x_centers):
            for j, y_pos in enumerate(y_centers):
                if (i % 2 == 1) != (j % 2 == 1):
                    height = 0.1
                else:
                    height = 0.1 + rand_state.uniform(*height_range)
                self._add_ground_box(
                    name=(
                        "ground_block_"
                        f"x{_format_name_number(x_pos)}_y{_format_name_number(y_pos)}"
                    ),
                    size=(0.55 * block_size, 0.55 * block_size, height / 2),
                    pos=(x_pos, y_pos, height / 2),
                    rgba=(0.3, 0.3, 0.3, ground_alpha),
                )


class MixedTerrainWorld(_ComplexTerrainWorld):
    """World with repeated blocks, gaps, and flat floor sections."""

    def __init__(
        self,
        name: str = "mixed_terrain_world",
        *,
        gap_width: float = 0.3,
        gapped_block_width: float = 1.0,
        gap_depth: float = 2.0,
        block_size: float = 1.3,
        height_range: tuple[float, float] = (0.35, 0.35),
        ground_alpha: float = 1.0,
        rand_seed: int = 0,
    ) -> None:
        super().__init__(name=name)
        y_range = (-20, 20)
        y_halfwidth = (y_range[1] - y_range[0]) / 2
        rand_state = np.random.RandomState(rand_seed)
        height_expected_value = np.mean(height_range)

        for range_idx, x_range in enumerate([(-4, 5), (5, 14), (14, 23)]):
            x_centers = np.arange(
                x_range[0] + block_size / 2,
                x_range[0] + block_size * 3,
                block_size,
            )
            y_centers = np.arange(y_range[0] + block_size / 2, y_range[1], block_size)
            for i, x_pos in enumerate(x_centers):
                for j, y_pos in enumerate(y_centers):
                    if (i % 2 == 1) != (j % 2 == 1):
                        height = 0.1
                    else:
                        height = 0.1 + rand_state.uniform(*height_range)
                    self._add_ground_box(
                        name=(
                            f"ground_mixed_block{range_idx}_"
                            f"x{_format_name_number(x_pos)}_"
                            f"y{_format_name_number(y_pos)}"
                        ),
                        size=(
                            0.55 * block_size,
                            0.55 * block_size,
                            height / 2 + block_size / 2,
                        ),
                        pos=(
                            x_pos,
                            y_pos,
                            height / 2 - block_size / 2 - height_expected_value - 0.1,
                        ),
                        rgba=(0.3, 0.3, 0.3, ground_alpha),
                    )

            curr_x_pos = x_range[0] + block_size * 3
            self._add_ground_box(
                name=f"ground_gap_pre{range_idx}",
                size=(gapped_block_width / 4, y_halfwidth, gap_depth / 2),
                pos=(curr_x_pos + gapped_block_width / 4, 0, -gap_depth / 2),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )
            curr_x_pos += gapped_block_width / 2 + gap_width
            self._add_ground_box(
                name=f"ground_gap_post{range_idx}",
                size=(gapped_block_width / 2, y_halfwidth, gap_depth / 2),
                pos=(curr_x_pos + gapped_block_width / 2, 0, -gap_depth / 2),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )
            curr_x_pos += gapped_block_width + gap_width
            remaining_space = x_range[1] - curr_x_pos
            self._add_ground_box(
                name=f"ground_flat{range_idx}",
                size=(remaining_space / 2, y_halfwidth, gap_depth / 2),
                pos=(curr_x_pos + remaining_space / 2, 0, -gap_depth / 2),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )
            self._add_ground_plane(
                name=f"ground_base{range_idx}",
                size=((x_range[1] - x_range[0]) / 2, y_halfwidth, 1),
                pos=(np.mean(x_range), 0, -gap_depth / 2),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )


class TetheredWorld(BaseWorld):
    """World where the fly body is fixed in space via a weld constraint.

    The fly's appendages (legs, wings, etc.) can still move. Useful for motor control
    experiments without locomotion.

    Args:
        name: Name of the world.
    """

    @override
    def __init__(self, name: str = "tethered_world") -> None:
        super().__init__(name=name)
        # don't add ground plane
        self.legpos_to_groundcontactsensors_by_fly = None

    @override
    def _attach_fly_mjcf(
        self, fly, spawn_position: Vec3, spawn_rotation: Rotation3D
    ) -> mjcf.Element:
        spawn_site = self.mjcf_root.worldbody.add(
            "site", name=fly.name, pos=spawn_position, **spawn_rotation.as_kwargs()
        )
        freejoint = spawn_site.attach(fly.mjcf_root).add("freejoint", name=fly.name)
        self.mjcf_root.equality.add(
            "weld",
            body2="world",  # worldbody is called "world" in equality constraints
            body1=fly.mjcf_root.find("body", fly.root_segment.name).full_identifier,
            relpose=(*spawn_position, *spawn_rotation.values),
            solref=(2e-4, 1.0),
            solimp=(0.98, 0.99, 1e-5, 0.5, 3),
        )
        return freejoint


def _sort_legsegs_prox2dist(segments: list[BodySegment]) -> list[BodySegment]:
    bodyseg_linkpos_tuples = [(seg, LEG_LINKS.index(seg.link)) for seg in segments]
    bodyseg_linkpos_tuples.sort(key=lambda x: x[1])
    return [t[0] for t in bodyseg_linkpos_tuples]


def _format_name_number(value: float) -> str:
    return f"{value:.3f}".replace("-", "m").replace(".", "p")
