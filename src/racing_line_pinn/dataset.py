"""Physics-coupled training-sample generation for the racing-line model.

Replaces per-driver data mixing with a single physically self-consistent
speed profile: a forward/backward minimum-lap-time solve (corner grip cap,
traction-limited acceleration, max-braking deceleration) computed purely from
track curvature and the Phase-2 calibrated :class:`VehicleParams` -- the
fastest profile achievable by ONE physically-consistent car, not a
cross-driver composite.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .calibration import VehicleParams
from .physics import calculate_aero_drag, calculate_speed_dependent_normal_load

GRAVITY = 9.81


def _corner_cap_speed(curvature: np.ndarray, vehicle: VehicleParams, iterations: int = 30) -> np.ndarray:
    """Self-consistent grip-limited speed at every point (no braking/throttle, long_accel=0).

    Downforce grows with speed, so the grip limit itself depends on speed;
    this iterates to a fixed point rather than solving in closed form.
    """
    radius = 1.0 / np.maximum(np.abs(curvature), 1e-6)
    speed = np.sqrt(vehicle.mu * GRAVITY * radius)
    for _ in range(iterations):
        normal_force = calculate_speed_dependent_normal_load(vehicle.mass, speed, 1.0, vehicle.aero_lift_area)
        lateral_limit = vehicle.mu * normal_force / vehicle.mass
        speed = np.sqrt(np.maximum(lateral_limit * radius, 0.0))
    return speed


def _forward_pass(
    s: np.ndarray, corner_cap: np.ndarray, vehicle: VehicleParams, start_speed: float | None = None
) -> np.ndarray:
    """Propagate speed forward, limited by traction-limited accel minus aero drag."""
    ds = np.diff(s)
    speed = np.empty_like(corner_cap)
    speed[0] = min(start_speed, corner_cap[0]) if start_speed is not None else corner_cap[0]
    for i in range(1, speed.size):
        v_prev = float(speed[i - 1])
        drag_decel = float(calculate_aero_drag(v_prev, 1.0, vehicle.aero_drag_area)) / vehicle.mass
        net_accel = max(vehicle.max_accel - drag_decel, 0.0)
        v_candidate = math.sqrt(v_prev ** 2 + 2.0 * net_accel * ds[i - 1])
        speed[i] = min(v_candidate, corner_cap[i])
    return speed


def _backward_pass(
    s: np.ndarray, corner_cap: np.ndarray, max_decel: float, start_speed: float | None = None
) -> np.ndarray:
    """Propagate speed backward from each point, limited by max braking deceleration."""
    ds = np.diff(s)
    speed = np.empty_like(corner_cap)
    speed[-1] = min(start_speed, corner_cap[-1]) if start_speed is not None else corner_cap[-1]
    for i in range(speed.size - 2, -1, -1):
        v_next = float(speed[i + 1])
        v_candidate = math.sqrt(v_next ** 2 + 2.0 * max_decel * ds[i])
        speed[i] = min(v_candidate, corner_cap[i])
    return speed


def build_training_samples(
    s: np.ndarray, curvature: np.ndarray, vehicle: VehicleParams
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the physics-coupled target speed profile for one session's curvature(s).

    Runs the corner-cap -> forward-accel -> backward-brake solve, then takes
    the minimum at each point -- the fastest profile achievable by one
    physically-consistent car. Each pass runs twice (second pass seeded from
    the first pass's far end) to approximate the closed-loop steady state of
    a repeating lap, since the track is a circuit, not an isolated segment.

    Returns ``(curvature, target_speed, corner_cap)`` tensors, all aligned to
    ``s``'s order and length.
    """
    corner_cap = _corner_cap_speed(curvature, vehicle)

    v_fwd = _forward_pass(s, corner_cap, vehicle)
    v_fwd = _forward_pass(s, corner_cap, vehicle, start_speed=float(v_fwd[-1]))

    v_bwd = _backward_pass(s, corner_cap, vehicle.max_decel)
    v_bwd = _backward_pass(s, corner_cap, vehicle.max_decel, start_speed=float(v_bwd[0]))

    target_speed = np.minimum(np.minimum(corner_cap, v_fwd), v_bwd)

    return (
        torch.tensor(curvature, dtype=torch.float32),
        torch.tensor(target_speed, dtype=torch.float32),
        torch.tensor(corner_cap, dtype=torch.float32),
    )
