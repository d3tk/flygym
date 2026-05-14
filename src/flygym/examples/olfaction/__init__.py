"""Olfaction examples and plume-navigation helpers."""

from flygym.compose import OdorPlumeWorld, OdorWorld
from flygym.examples.olfaction.plume_tracking_controller import (
    PlumeNavigationController,
    WalkingState,
)
from flygym.examples.olfaction.plume_tracking_task import PlumeNavigationTask

OdorPlumeArena = OdorPlumeWorld

__all__ = [
    "OdorWorld",
    "OdorPlumeWorld",
    "OdorPlumeArena",
    "PlumeNavigationTask",
    "PlumeNavigationController",
    "WalkingState",
]
