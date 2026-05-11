from __future__ import annotations

import numpy as np


def get_leg_mask(legs: list[str], included_legs: list[str] | None = None) -> np.ndarray:
    if included_legs is None:
        included_legs = legs
    included = {leg.lower() for leg in included_legs}
    return np.array([leg.lower() in included for leg in legs], dtype=bool)


def extract_variables(records: list[dict], contact_force_thr: float = 0.5) -> dict[str, np.ndarray]:
    """Extract compact proprioceptive variables from closed-loop records."""

    joint_angles = np.asarray([r["joint_angles"] for r in records])
    body_positions = np.asarray([r["body_positions"][0] for r in records])
    contact_force = np.asarray(
        [
            r.get("metadata", {}).get("contact_force", np.zeros(6))
            for r in records
        ]
    )
    contact_mask = contact_force > contact_force_thr
    return {
        "joint_angles": joint_angles,
        "body_position": body_positions,
        "contact_mask": contact_mask,
    }
