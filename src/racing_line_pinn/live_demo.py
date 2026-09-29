"""Utilities for the interactive racing-line demo."""

from __future__ import annotations

import numpy as np

from .physics import compute_racing_line_target_speed


TRACK_LIBRARY = {
    "Spa-Francorchamps": {
        "x_scale": 1.5,
        "y_scale": 1.0,
        "turn_bias": 1.0,
    },
    "Silverstone": {
        "x_scale": 1.4,
        "y_scale": 0.9,
        "turn_bias": 0.75,
    },
    "Monza": {
        "x_scale": 1.7,
        "y_scale": 0.7,
        "turn_bias": 0.55,
    },
    "Suzuka": {
        "x_scale": 1.45,
        "y_scale": 0.95,
        "turn_bias": 0.9,
    },
}

CAR_PRESETS = {
    "Ferrari SF-24": {"mu": 1.35, "downforce": 3200.0, "mass": 800.0},
    "Red Bull RB20": {"mu": 1.42, "downforce": 3600.0, "mass": 790.0},
    "Mercedes W15": {"mu": 1.38, "downforce": 3400.0, "mass": 810.0},
    "McLaren MCL60": {"mu": 1.33, "downforce": 3000.0, "mass": 820.0},
    "Custom": {"mu": 1.35, "downforce": 3000.0, "mass": 800.0},
}


def _track_parametric_profile(track_name: str, n_points: int = 400, amplitude: float = 1.0):
    """Create a synthetic but circuit-like centerline for a selected track."""
    t = np.linspace(0.0, 1.0, n_points)
    profile = TRACK_LIBRARY.get(track_name, TRACK_LIBRARY["Spa-Francorchamps"])
    x_scale = profile["x_scale"]
    y_scale = profile["y_scale"]
    turn_bias = profile["turn_bias"]

    if track_name == "Spa-Francorchamps":
        x = x_scale * (np.cos(2.0 * np.pi * t) + 0.24 * np.sin(6.0 * np.pi * t))
        y = y_scale * (np.sin(2.0 * np.pi * t) + 0.18 * np.cos(4.0 * np.pi * t))
    elif track_name == "Silverstone":
        x = x_scale * np.cos(2.0 * np.pi * t)
        y = y_scale * (np.sin(2.0 * np.pi * t) + 0.12 * np.sin(4.0 * np.pi * t))
    elif track_name == "Monza":
        x = x_scale * np.cos(2.0 * np.pi * t) + 0.35 * np.sin(2.0 * np.pi * t)
        y = y_scale * np.sin(2.0 * np.pi * t)
    else:
        x = x_scale * (np.cos(2.0 * np.pi * t) + 0.2 * np.sin(4.0 * np.pi * t))
        y = y_scale * (np.sin(2.0 * np.pi * t) + 0.15 * np.sin(6.0 * np.pi * t))

    x *= amplitude
    y *= amplitude * (0.75 + turn_bias * 0.55)
    return x, y


def _estimate_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimate the local geometric curvature from x/y coordinates."""
    dx = np.diff(x)
    dy = np.diff(y)
    theta = np.arctan2(dy, dx)
    dtheta = np.diff(np.unwrap(theta))
    ds = np.hypot(dx[:-1], dy[:-1])
    curvature = np.abs(dtheta) / np.maximum(ds, 1e-6)
    curvature = np.pad(curvature, (1, 1), mode="edge")
    return curvature


def simulate_session(
    track_name: str = "Spa-Francorchamps",
    mu: float = 1.35,
    downforce: float = 3000.0,
    mass: float = 800.0,
    amplitude: float = 1.0,
):
    """Create actual and ideal lap traces under a selected circuit and setup."""
    x, y = _track_parametric_profile(track_name, n_points=420, amplitude=amplitude)
    curvature = _estimate_curvature(x, y)

    ideal_x = x.copy()
    ideal_y = y.copy()
    actual_x = x + 0.08 * np.sin(2.0 * np.pi * np.linspace(0.0, 1.0, len(x)))
    actual_y = y + 0.06 * np.cos(2.0 * np.pi * np.linspace(0.0, 1.0, len(y)))

    predicted_speed = []
    for c in curvature:
        radius = 1.0 / max(abs(float(c)), 1e-3)
        predicted_speed.append(
            compute_racing_line_target_speed(
                curvature=float(c),
                radius=float(radius),
                mu=float(mu),
                mass=float(mass),
                downforce=float(downforce),
            )
        )

    actual_speed = np.asarray(predicted_speed, dtype=float) * (0.82 + 0.22 * np.sin(np.linspace(0.0, 2.0 * np.pi, len(predicted_speed))))
    apex_index = int(np.argmax(np.abs(curvature)))

    return {
        "track_name": track_name,
        "x": ideal_x,
        "y": ideal_y,
        "actual_x": actual_x,
        "actual_y": actual_y,
        "predicted_speed": np.asarray(predicted_speed, dtype=float),
        "actual_speed": actual_speed,
        "curvature": curvature,
        "apex_index": apex_index,
        "apex_x": ideal_x[apex_index],
        "apex_y": ideal_y[apex_index],
    }
