import math

from racing_line_pinn.physics import (
    calculate_lateral_grip_limit,
    compute_corner_speed_limit,
    compute_racing_line_target_speed,
)
from racing_line_pinn.compare import calculate_speed_metrics, synthesize_driver_trajectory
from racing_line_pinn.session_analysis import compare_session_to_model, estimate_curvature_from_xy


def test_lateral_grip_limit_is_positive_and_increases_with_downforce():
    base = calculate_lateral_grip_limit(mu=1.2, mass=800.0, downforce=2000.0)
    loaded = calculate_lateral_grip_limit(mu=1.2, mass=800.0, downforce=5000.0)
    assert base > 0
    assert loaded > base


def test_corner_speed_limit_scales_with_radius():
    fast_corner = compute_corner_speed_limit(radius=22.0, mu=1.2, downforce=2500.0, mass=800.0)
    slow_corner = compute_corner_speed_limit(radius=10.0, mu=1.2, downforce=2500.0, mass=800.0)
    assert fast_corner > slow_corner
    assert fast_corner > 0


def test_target_speed_uses_curvature_and_lateral_limit():
    speed = compute_racing_line_target_speed(
        curvature=0.05,
        radius=20.0,
        mu=1.3,
        mass=750.0,
        downforce=2800.0,
    )
    assert speed > 0
    assert math.isfinite(speed)


def test_speed_metrics_are_non_negative_and_synthesize_trajectory_has_expected_shape():
    actual = [30.0, 32.0, 28.0, 35.0]
    predicted = [31.0, 31.5, 29.0, 34.0]
    metrics = calculate_speed_metrics(actual, predicted)
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0
    assert metrics["mean_error"] >= -100.0

    trajectory = synthesize_driver_trajectory(8)
    assert len(trajectory["x"]) == 8
    assert len(trajectory["y"]) == 8


def test_curve_estimation_and_session_compare_are_finite():
    x = [-1.0, -0.5, 0.0, 0.5, 1.0]
    y = [0.0, 0.5, 0.0, -0.5, 0.0]
    curvature = estimate_curvature_from_xy(x, y)
    assert curvature.size > 0
    assert all(value == value for value in curvature)

    session = {
        "x": x,
        "y": y,
        "speed": [20.0, 24.0, 28.0, 24.0, 20.0],
    }
    result = compare_session_to_model(session, mu=1.4, downforce=3000.0, mass=800.0)
    assert result["actual_speed"]
    assert result["predicted_speed"]
    assert result["metrics"]["rmse"] >= 0.0
