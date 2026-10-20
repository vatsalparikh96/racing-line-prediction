from __future__ import annotations

import numpy as np

from racing_line_pinn.visualization import build_animated_lap

from test_calibration import _build_synthetic_session


def test_build_animated_lap_without_ghost():
    session, _apex_percentile = _build_synthetic_session()
    fig = build_animated_lap(session, "TST", frame_dt=1.0)

    assert len(fig.frames) > 0
    # centerline + driver line + driver marker + 3 gauges = 6 static traces (no ghost).
    assert len(fig.data) == 6
    assert fig.layout.updatemenus
    assert fig.layout.sliders


def test_build_animated_lap_with_model_ghost():
    session, _apex_percentile = _build_synthetic_session()
    model_speed = np.full_like(session.curvature, np.mean(session.drivers["TST"].speed))
    fig = build_animated_lap(session, "TST", model_speed=model_speed, frame_dt=1.0)

    # +1 trace for the ghost marker.
    assert len(fig.data) == 7
    first_frame = fig.frames[0]
    assert len(first_frame.data) == 5  # driver marker + ghost marker + 3 gauges
