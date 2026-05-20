"""Olfaction demo helpers used by the tutorial examples."""

from flygym.compose import OdorPlumeWorld, OdorWorld
from flygym_demo.olfaction.plume_tracking_controller import (
    PlumeNavigationController,
    WalkingState,
)
from flygym_demo.olfaction.plume_tracking_task import PlumeNavigationTask

OdorPlumeArena = OdorPlumeWorld

__all__ = [
    "OdorWorld",
    "OdorPlumeWorld",
    "OdorPlumeArena",
    "PlumeNavigationTask",
    "PlumeNavigationController",
    "WalkingState",
]
