"""Calibration tests using a synthetic-but-physically-controlled session.

Builds a fake :class:`SpaSessionData` (two corners of different radius, joined
by straight accel/brake zones) whose speed trace is generated FROM known
``mu``/aero/accel/decel constants via the same physics relations the
calibration fits against, then checks that ``calibrate_vehicle_params``
recovers those known constants.
"""

from __future__ import annotations

import numpy as np
import pytest

from racing_line_pinn.calibration import MASS_KG, calibrate_vehicle_params
from racing_line_pinn.physics import (
    calculate_aero_drag,
    calculate_speed_dependent_normal_load,
    pacejka_combined_friction_limit,
)
from racing_line_pinn.real_data import DriverLapData, SpaSessionData

MU_TRUE = 1.6
AERO_LIFT_AREA_TRUE = 3.0
AERO_DRAG_AREA_TRUE = 1.2
MAX_ACCEL_TRUE = 10.0
MAX_DECEL_TRUE = 45.0
DS = 0.5


def _apex_speed(radius: float) -> float:
    """Self-consistent corner speed for the known mu/aero params (fixed-point iteration)."""
    speed = 30.0
    for _ in range(100):
        normal_force = calculate_speed_dependent_normal_load(MASS_KG, speed, 1.0, AERO_LIFT_AREA_TRUE)
        limit = pacejka_combined_friction_limit(MU_TRUE, normal_force, MASS_KG, long_accel=0.0)
        speed = (limit * radius) ** 0.5
    return speed


def _accel_zone(v_start: float, v_target: float) -> tuple[np.ndarray, np.ndarray]:
    """Euler-integrate dv/dt = max_accel - drag_decel(v) in time, returning (s, speed) on a DS grid."""
    dt = 0.005
    t, s, v = 0.0, 0.0, v_start
    s_samples = [0.0]
    v_samples = [v_start]
    while v < v_target and t < 120.0:
        drag_decel = calculate_aero_drag(v, 1.0, AERO_DRAG_AREA_TRUE) / MASS_KG
        a = MAX_ACCEL_TRUE - drag_decel
        v += a * dt
        s += v * dt
        t += dt
        s_samples.append(s)
        v_samples.append(v)

    s_grid = np.arange(0.0, s_samples[-1], DS)
    speed_grid = np.interp(s_grid, s_samples, v_samples)
    return s_grid, speed_grid


def _brake_zone(v_start: float, v_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Analytic constant-deceleration kinematics: v(s)^2 = v_start^2 - 2*max_decel*s."""
    length = (v_start ** 2 - v_end ** 2) / (2.0 * MAX_DECEL_TRUE)
    s_grid = np.arange(0.0, length, DS)
    speed_grid = np.sqrt(np.maximum(v_start ** 2 - 2.0 * MAX_DECEL_TRUE * s_grid, v_end ** 2))
    return s_grid, speed_grid


def _build_synthetic_session() -> tuple[SpaSessionData, float]:
    radius1, radius2 = 40.0, 90.0
    v_apex1 = _apex_speed(radius1)
    v_apex2 = _apex_speed(radius2)
    kappa1, kappa2 = 1.0 / radius1, 1.0 / radius2

    apex_len = 20.0
    s_apex1 = np.arange(0.0, apex_len, DS)
    s_zoneA, v_zoneA = _accel_zone(v_apex1, 80.0)
    s_zoneB, v_zoneB = _brake_zone(80.0, v_apex2)
    s_apex2 = np.arange(0.0, apex_len, DS)
    s_zoneC, v_zoneC = _accel_zone(v_apex2, v_apex2 + 25.0)

    segments_s = [s_apex1, s_zoneA, s_zoneB, s_apex2, s_zoneC]
    segments_v = [
        np.full_like(s_apex1, v_apex1),
        v_zoneA,
        v_zoneB,
        np.full_like(s_apex2, v_apex2),
        v_zoneC,
    ]
    segments_kappa = [
        np.full_like(s_apex1, kappa1),
        np.zeros_like(s_zoneA),
        np.zeros_like(s_zoneB),
        np.full_like(s_apex2, kappa2),
        np.zeros_like(s_zoneC),
    ]

    offsets = np.cumsum([0.0] + [seg[-1] + DS for seg in segments_s[:-1]])
    s = np.concatenate([seg + off for seg, off in zip(segments_s, offsets)])
    speed = np.concatenate(segments_v)
    curvature = np.concatenate(segments_kappa)

    total_length = s[-1]
    # Percentile just above the exact zero-curvature fraction, so both nonzero
    # corner clusters are selected while every straight (curvature == 0) sample is not.
    zero_fraction = float(np.mean(curvature == 0.0))
    apex_percentile = min(zero_fraction * 100.0 + 0.5, 99.9)

    driver = DriverLapData(
        driver="TST",
        s=s,
        speed=speed,
        x=s,
        y=np.zeros_like(s),
        throttle=np.zeros_like(s),
        brake=np.zeros_like(s),
        lap_time=float(total_length / np.mean(speed)),
    )
    session = SpaSessionData(
        year=2023,
        session_name="R",
        centerline_x=s,
        centerline_y=np.zeros_like(s),
        s=s,
        curvature=curvature,
        track_width=np.full_like(s, 12.0),
        drivers={"TST": driver},
        reference_driver="TST",
    )
    return session, apex_percentile


def test_calibration_recovers_known_mu_and_lift_area():
    session, apex_percentile = _build_synthetic_session()
    params = calibrate_vehicle_params(session, apex_percentile=apex_percentile)

    assert params.mu == pytest.approx(MU_TRUE, rel=0.1)
    assert params.aero_lift_area == pytest.approx(AERO_LIFT_AREA_TRUE, rel=0.3)


def test_calibration_recovers_plausible_accel_decel_and_drag():
    session, apex_percentile = _build_synthetic_session()
    params = calibrate_vehicle_params(session, apex_percentile=apex_percentile)

    assert params.max_accel == pytest.approx(MAX_ACCEL_TRUE, rel=0.2)
    assert params.max_decel == pytest.approx(MAX_DECEL_TRUE, rel=0.2)
    assert params.aero_drag_area == pytest.approx(AERO_DRAG_AREA_TRUE, rel=0.4)
    assert params.mass == MASS_KG
