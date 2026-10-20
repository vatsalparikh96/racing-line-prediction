import math

from racing_line_pinn.physics import (
    calculate_lateral_grip_limit,
    compute_corner_speed_limit,
    compute_racing_line_target_speed,
)


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

