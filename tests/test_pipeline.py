"""Pipeline-shape tests: common-s grid resampling invariant + full calibrate->train->evaluate chain.

These exercise the same shape/alignment
invariants that `load_spa_race_session` guarantees (every driver resampled onto one
shared centerline `s` grid) using the synthetic-but-physically-controlled fixture
from `test_calibration.py`, plus a full run of the non-network part of the CLI
pipeline (calibrate -> train -> leaderboard) end-to-end.
"""

from __future__ import annotations

import numpy as np
import racing_line_pinn.real_data as real_data

from racing_line_pinn.evaluation import build_leaderboard
from racing_line_pinn.real_data import DriverLapData, _resample_on_grid
from racing_line_pinn.training import train_model

from test_calibration import _build_synthetic_session


def test_openf1_fallback_builds_aligned_session(monkeypatch):
    from datetime import datetime, timedelta, timezone

    sample_count = 48
    base_time = datetime(2023, 7, 30, 14, tzinfo=timezone.utc)
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    location_rows = [
        {
            "date": (base_time + timedelta(seconds=2 * index)).isoformat(),
            "x": 10000.0 * np.cos(angle),
            "y": 10000.0 * np.sin(angle),
        }
        for index, angle in enumerate(angles)
    ]
    car_rows = [
        {
            "date": row["date"],
            "speed": 220,
            "throttle": 80,
            "brake": 0,
        }
        for row in location_rows
    ]
    endpoints = {
        "sessions": [{"session_key": 9141, "circuit_short_name": "Spa-Francorchamps"}],
        "drivers": [
            {"driver_number": 1, "name_acronym": "VER"},
            {"driver_number": 44, "name_acronym": "HAM"},
            {"driver_number": 81, "name_acronym": "PIA"},
        ],
        "laps": [
            {
                "driver_number": 1,
                "date_start": base_time.isoformat(),
                "lap_duration": 110.0,
                "is_pit_out_lap": False,
            },
            {
                "driver_number": 1,
                "date_start": (base_time + timedelta(minutes=3)).isoformat(),
                "lap_duration": 109.0,
                "is_pit_out_lap": False,
            },
            {
                "driver_number": 44,
                "date_start": base_time.isoformat(),
                "lap_duration": 108.0,
                "is_pit_out_lap": False,
            },
            {
                "driver_number": 44,
                "date_start": base_time.isoformat(),
                "lap_duration": 90.0,
                "is_pit_out_lap": True,
            },
            {
                "driver_number": 81,
                "date_start": base_time.isoformat(),
                "lap_duration": None,
                "is_pit_out_lap": False,
            },
        ],
    }
    monkeypatch.setattr(real_data, "_openf1_get", lambda endpoint, params: endpoints[endpoint])
    monkeypatch.setattr(
        real_data,
        "_fetch_openf1_lap_telemetry",
        lambda _session_key, _laps: {
            "1": (location_rows, car_rows),
            "44": (location_rows, car_rows),
            "81": (location_rows[:24], car_rows[:24]),
        },
    )

    session = real_data._load_openf1_session(2023, "R")

    assert session is not None
    assert session.reference_driver == "HAM"
    assert set(session.drivers) == {"VER", "HAM", "PIA"}
    assert session.drivers["VER"].lap_time == 109.0
    assert session.drivers["HAM"].lap_time == 108.0
    assert session.drivers["VER"].speed.shape == session.s.shape
    assert np.allclose(session.drivers["VER"].speed, 220 / 3.6)
    assert not session.drivers["PIA"].lap_complete
    assert session.drivers["PIA"].lap_time is None
    assert session.drivers["PIA"].s.size < session.s.size
    assert {row.driver for row in build_leaderboard(session.drivers, session.s, np.full_like(session.s, 60))} == {
        "VER",
        "HAM",
    }


def test_public_session_loader_falls_back_to_openf1(monkeypatch):
    expected_session = object()
    monkeypatch.setattr(real_data, "_load_openf1_session", lambda _year, _name: expected_session)

    assert real_data.load_spa_race_session() is expected_session


def test_resample_on_grid_preserves_range_and_shape():
    s_source = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    values = np.array([10.0, 12.0, 11.0, 15.0, 13.0])
    s_target = np.linspace(0.0, 8.0, 17)

    resampled = _resample_on_grid(s_target, s_source, values)

    assert resampled.shape == s_target.shape
    assert resampled.min() >= values.min()
    assert resampled.max() <= values.max()


def test_all_drivers_share_the_session_s_grid():
    session, _apex_percentile = _build_synthetic_session()
    reference = session.drivers["TST"]

    second_driver = DriverLapData(
        driver="TS2",
        s=session.s,
        speed=reference.speed * 0.98,
        x=reference.x,
        y=reference.y,
        throttle=reference.throttle,
        brake=reference.brake,
        lap_time=reference.lap_time * 1.02,
    )
    session.drivers["TS2"] = second_driver

    for lap in session.drivers.values():
        assert lap.s.shape == session.s.shape
        assert np.array_equal(lap.s, session.s)


def test_full_pipeline_shapes_end_to_end():
    from racing_line_pinn.calibration import calibrate_vehicle_params

    session, apex_percentile = _build_synthetic_session()
    vehicle = calibrate_vehicle_params(session, apex_percentile=apex_percentile)

    _model, predicted_speed = train_model(session.s, session.curvature, vehicle, epochs=5, hidden_dim=8)
    predicted_speed_np = predicted_speed.numpy().reshape(-1)

    assert predicted_speed_np.shape == session.s.shape
    assert np.all(np.isfinite(predicted_speed_np))

    leaderboard = build_leaderboard(session.drivers, session.s, predicted_speed_np)
    assert len(leaderboard) == len(session.drivers)
    for comparison in leaderboard:
        assert comparison.s.shape == session.s.shape
        assert np.isfinite(comparison.total_delta_t)
