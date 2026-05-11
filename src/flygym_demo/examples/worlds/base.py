from __future__ import annotations

from collections import defaultdict

import dm_control.mjcf as mjcf
import numpy as np

from flygym.anatomy import BodySegment, ContactBodiesPreset, LEG_LINKS
from flygym.compose import BaseWorld, ContactParams, Fly
from flygym.utils.math import Rotation3D, Vec3


class DemoWorld(BaseWorld):
    """Base world for tutorial arenas with one or more contact geoms."""

    def __init__(self, name: str = "demo_world") -> None:
        super().__init__(name=name)
        self.ground_geoms: list[mjcf.Element] = []
        self.legpos_to_groundcontactsensors_by_fly = None
        self._init_lights()

    def _init_lights(self) -> None:
        self.mjcf_root.worldbody.add(
            "light",
            name="light_top",
            mode="trackcom",
            directional=True,
            castshadow=False,
            active=True,
            pos=(0, 0, 80),
            dir=(0, 0, -1),
        )

    def _attach_fly_mjcf(
        self,
        fly: Fly,
        spawn_position: Vec3,
        spawn_rotation: Rotation3D,
        *,
        bodysegs_with_ground_contact: list[BodySegment] | ContactBodiesPreset | str = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
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

        self._set_ground_contact(fly, bodysegs_with_ground_contact, ground_contact_params)
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
            for i, ground_geom in enumerate(self.ground_geoms):
                self.mjcf_root.contact.add(
                    "pair",
                    geom1=body_geom,
                    geom2=ground_geom,
                    name=f"{body_segment.name}-{ground_geom.name}-{i}",
                    friction=ground_contact_params.get_friction_tuple(),
                    solref=ground_contact_params.get_solref_tuple(),
                    solimp=ground_contact_params.get_solimp_tuple(),
                    margin=ground_contact_params.margin,
                )

    def _add_ground_contact_sensors(
        self, fly: Fly, bodysegs_with_ground_contact: list[BodySegment]
    ) -> None:
        if not self.ground_geoms:
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

    def reset(self, sim) -> None:
        return None

    def step(self, sim, dt: float) -> None:
        return None


def _sort_legsegs_prox2dist(segments: list[BodySegment]) -> list[BodySegment]:
    bodyseg_linkpos_tuples = [(seg, LEG_LINKS.index(seg.link)) for seg in segments]
    bodyseg_linkpos_tuples.sort(key=lambda x: x[1])
    return [t[0] for t in bodyseg_linkpos_tuples]


def add_checker_plane(root, name: str, half_size: tuple[float, float], z: float = 0.0):
    tex = root.asset.add(
        "texture",
        name=f"{name}_checker",
        type="2d",
        builtin="checker",
        width=300,
        height=300,
        rgb1=(0.3, 0.3, 0.3),
        rgb2=(0.4, 0.4, 0.4),
    )
    mat = root.asset.add(
        "material",
        name=f"{name}_grid",
        texture=tex,
        texrepeat=(60, 60),
        reflectance=0.1,
    )
    return root.worldbody.add(
        "geom",
        type="plane",
        name=name,
        material=mat,
        pos=(0, 0, z),
        size=(*half_size, 1),
        contype=0,
        conaffinity=0,
    )
