"""Tests for the v1/v2 CPG comparison debugger helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_compare_module():
    script = Path(__file__).resolve().parents[2] / "scripts/compare_cpg_v1_v2_kinematics.py"
    spec = importlib.util.spec_from_file_location("compare_cpg_v1_v2_kinematics", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parity_summary_uses_stage_specific_checks():
    mod = _load_compare_module()
    openloop = {"d_phase": 0.0, "d_mag": 0.0, "reorder_err": 0.0}
    v1 = {
        "thorax_delta_x_mm": 10.0,
        "mean_ee_rel_x_span_mm": 2.0,
        "mean_ee_rel_y_span_mm": 1.0,
        "mean_ee_rel_z_span_mm": 0.5,
        "mean_rmse_cmd_vs_pos_rad": 0.1,
    }
    v2 = {
        "thorax_delta_x_mm": 0.0,
        "mean_ee_rel_x_span_mm": 2.1,
        "mean_ee_rel_y_span_mm": 1.1,
        "mean_ee_rel_z_span_mm": 0.55,
        "mean_rmse_cmd_vs_pos_rad": 0.2,
        "tarsus5_contact_during_swing_frac": 1.0,
    }

    fk = mod._parity_summary("fk", openloop, v1, v2)
    assert "fk_endpoint_x_span_within_20pct" in fk["checks"]
    assert "thorax_delta_x_within_20pct" not in fk["checks"]
    assert "swing_tarsus5_contact_below_25pct" not in fk["checks"]

    tracking = mod._parity_summary("tracking", openloop, v1, v2)
    assert tracking["checks"]["tracking_states_finite"] is True
    assert "thorax_delta_x_within_20pct" not in tracking["checks"]

    full = mod._parity_summary("full", openloop, v1, v2)
    assert full["checks"]["thorax_delta_x_within_20pct"] is False
    assert full["checks"]["swing_tarsus5_contact_below_25pct"] is False


def test_sim_tracking_stats_reports_per_leg_endpoint_and_contact_metrics():
    mod = _load_compare_module()
    n_steps = 4
    n_legs = 6
    thorax = np.zeros((n_steps, 3), dtype=float)
    thorax[:, 0] = np.arange(n_steps, dtype=float)
    thorax[:, 2] = 1.0
    ee_rel = np.zeros((n_steps, n_legs, 3), dtype=float)
    ee_rel[:, :, 0] = np.arange(n_steps, dtype=float)[:, None]
    ee_rel[:, :, 2] = np.linspace(0.0, 0.3, n_steps)[:, None]
    ee = thorax[:, None, :] + ee_rel
    adhesion = np.array(
        [
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=float,
    )
    contact_flag = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=float,
    )
    tarsus_force = np.zeros((n_steps, n_legs, 3), dtype=float)
    tarsus_force[contact_flag > 0, 2] = 1.0
    logs = {
        "joint_cmd": np.zeros((n_steps, 42), dtype=float),
        "joint_pos": np.zeros((n_steps, 42), dtype=float),
        "joint_vel": np.zeros((n_steps, 42), dtype=float),
        "thorax_pos": thorax,
        "thorax_rot_euler": np.zeros((n_steps, 3), dtype=float),
        "ee_pos": ee,
        "adhesion_cmd": adhesion,
        "contact_flag": contact_flag,
        "contact_force_norm": contact_flag,
        "tarsus5_contact_force": tarsus_force,
    }

    stats = mod._sim_tracking_stats(logs, dt=0.1)

    assert stats["thorax_delta_x_mm"] == pytest.approx(3.0)
    assert stats["thorax_mean_forward_velocity_mm_s"] == pytest.approx(10.0)
    assert stats["mean_ee_rel_x_span_mm"] == pytest.approx(3.0)
    assert len(stats["ee_rel_x_span_mm_by_leg"]) == n_legs
    assert len(stats["contact_during_swing_frac_by_leg"]) == n_legs
    assert len(stats["tarsus5_contact_during_swing_frac_by_leg"]) == n_legs
    assert stats["tarsus5_contact_frac_by_leg"][0] == pytest.approx(0.5)
