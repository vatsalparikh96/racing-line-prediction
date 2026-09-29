"""Vehicle dynamics helpers for a simplified racing line model."""

from __future__ import annotations

import math


def calculate_lateral_grip_limit(mu: float, mass: float, downforce: float) -> float:
    """Return the maximum lateral acceleration available in m/s^2.

    Based on a simplified load transfer model where downforce increases the normal
    force and therefore the tire's lateral grip capability.
    """
    if mu <= 0 or mass <= 0:
        raise ValueError("mu and mass must be positive.")

    normal_force = mass * 9.81 + downforce
    return mu * normal_force / mass


def compute_corner_speed_limit(
    radius: float,
    mu: float,
    downforce: float,
    mass: float,
) -> float:
    """Compute the stable corner speed from a circular track approximation."""
    if radius <= 0:
        raise ValueError("radius must be positive.")

    lateral_accel = calculate_lateral_grip_limit(mu, mass, downforce)
    return math.sqrt(max(lateral_accel * radius, 0.0))


def compute_racing_line_target_speed(
    curvature: float,
    radius: float,
    mu: float,
    mass: float,
    downforce: float,
) -> float:
    """Estimate the target speed along a curved lane under downforce.

    The speed is limited by the lateral grip and also sensitive to curvature.
    """
    if curvature == 0:
        return 1e9

    corner_limit = compute_corner_speed_limit(radius, mu, downforce, mass)
    curvature_factor = 1.0 / max(abs(curvature), 1e-6)
    return max(min(corner_limit, curvature_factor * 35.0), 0.0)
