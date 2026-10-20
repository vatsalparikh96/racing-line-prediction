"""Vehicle dynamics helpers for a simplified racing line model."""

from __future__ import annotations

import math

AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m^3


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


def calculate_aero_downforce(speed: float, c_lift: float, frontal_area: float, air_density: float = AIR_DENSITY_SEA_LEVEL) -> float:
    """Speed-dependent aerodynamic downforce: F_z = 1/2 * rho * C_L * A * v^2."""
    if frontal_area <= 0:
        raise ValueError("frontal_area must be positive.")
    return 0.5 * air_density * c_lift * frontal_area * speed ** 2


def calculate_aero_drag(speed: float, c_drag: float, frontal_area: float, air_density: float = AIR_DENSITY_SEA_LEVEL) -> float:
    """Speed-dependent aerodynamic drag: F_d = 1/2 * rho * C_D * A * v^2."""
    if frontal_area <= 0:
        raise ValueError("frontal_area must be positive.")
    return 0.5 * air_density * c_drag * frontal_area * speed ** 2


def calculate_speed_dependent_normal_load(
    mass: float,
    speed: float,
    c_lift: float,
    frontal_area: float,
    air_density: float = AIR_DENSITY_SEA_LEVEL,
) -> float:
    """Total normal load F_z = mg + aero downforce, which grows with v^2."""
    if mass <= 0:
        raise ValueError("mass must be positive.")
    return mass * 9.81 + calculate_aero_downforce(speed, c_lift, frontal_area, air_density)


def pacejka_combined_friction_limit(mu: float, normal_force: float, mass: float, long_accel: float = 0.0) -> float:
    """Max lateral acceleration under a Pacejka-style combined friction ellipse.

    Uses the elliptical friction-circle constraint
    ``(a_lat / a_max)^2 + (a_long / a_max)^2 <= 1``, so braking or accelerating
    (``long_accel``) reduces the lateral grip budget left over for cornering.
    """
    if mu <= 0 or mass <= 0:
        raise ValueError("mu and mass must be positive.")

    a_max = mu * normal_force / mass
    if a_max <= 0:
        return 0.0
    remaining = max(1.0 - (long_accel / a_max) ** 2, 0.0)
    return a_max * math.sqrt(remaining)


def compute_two_track_weight_transfer(mass: float, lateral_accel: float, cg_height: float, track_width: float) -> float:
    """Simplified two-track lateral weight transfer: delta F_z = m * a_lat * h / t."""
    if track_width <= 0:
        raise ValueError("track_width must be positive.")
    return mass * lateral_accel * cg_height / track_width


def distribute_wheel_loads(normal_force: float, weight_transfer: float) -> dict[str, float]:
    """Split total normal load into four corner loads under lateral weight transfer.

    Assumes an even static front/rear split and an even front/rear share of the
    lateral transfer (a simplified two-track model, not a full suspension model).
    """
    per_wheel_static = normal_force / 4.0
    per_axle_transfer = weight_transfer / 2.0
    outer = per_wheel_static + per_axle_transfer / 2.0
    inner = per_wheel_static - per_axle_transfer / 2.0
    return {
        "front_outer": outer,
        "front_inner": inner,
        "rear_outer": outer,
        "rear_inner": inner,
    }


def compute_target_speed_with_aero(
    curvature: float,
    radius: float,
    mu: float,
    mass: float,
    c_lift: float,
    c_drag: float = 0.9,
    frontal_area: float = 1.5,
    air_density: float = AIR_DENSITY_SEA_LEVEL,
    cg_height: float = 0.3,
    track_width: float = 1.6,
    iterations: int = 4,
) -> dict:
    """Estimate cornering target speed including speed-dependent aero downforce.

    Since downforce (and therefore available grip) grows with v^2, this solves
    for a self-consistent speed by iterating: recompute normal load, the
    Pacejka combined-ellipse lateral limit, and weight transfer at the previous
    speed estimate until the value converges. Returns the resulting speed
    alongside the supporting load/grip breakdown.
    """
    if radius <= 0:
        raise ValueError("radius must be positive.")
    if mass <= 0:
        raise ValueError("mass must be positive.")

    curvature_cap = 1.0 / max(abs(curvature), 1e-6) * 35.0
    speed = math.sqrt(max(mu * 9.81 * radius, 0.0))
    normal_force = mass * 9.81
    lateral_accel = 0.0

    for _ in range(max(iterations, 1)):
        normal_force = calculate_speed_dependent_normal_load(mass, speed, c_lift, frontal_area, air_density)
        lateral_limit = pacejka_combined_friction_limit(mu, normal_force, mass, long_accel=0.0)
        candidate = math.sqrt(max(lateral_limit * radius, 0.0))
        speed = max(min(candidate, curvature_cap), 0.0)
        lateral_accel = speed ** 2 / radius

    weight_transfer = compute_two_track_weight_transfer(mass, lateral_accel, cg_height, track_width)
    wheel_loads = distribute_wheel_loads(normal_force, weight_transfer)
    drag_force = calculate_aero_drag(speed, c_drag, frontal_area, air_density)

    return {
        "speed": speed,
        "normal_force": normal_force,
        "lateral_accel": lateral_accel,
        "weight_transfer": weight_transfer,
        "wheel_loads": wheel_loads,
        "drag_force": drag_force,
    }
