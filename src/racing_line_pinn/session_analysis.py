"""Session-level comparison between real or synthetic telemetry and model predictions."""

from __future__ import annotations

import numpy as np

from .physics import compute_racing_line_target_speed


def estimate_curvature_from_xy(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> np.ndarray:
    """Estimate a simple curvature proxy from projected x/y coordinates."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 3 or len(y_arr) < 3:
        return np.array([], dtype=float)

    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    dtheta = np.arctan2(dy, dx)
    curvature = np.abs(np.diff(np.unwrap(dtheta))) / np.maximum(np.hypot(dx[:-1], dy[:-1]), 1e-6)
    return np.pad(curvature, (1, 1), constant_values=curvature[0] if len(curvature) else 0.0)


def compare_session_to_model(session: dict, mu: float, downforce: float, mass: float) -> dict:
    """Compare a track session against physics-informed speed predictions."""
    x = np.asarray(session.get("x", []), dtype=float)
    y = np.asarray(session.get("y", []), dtype=float)
    actual_speed = np.asarray(session.get("speed", []), dtype=float)

    if len(x) == 0 or len(y) == 0 or len(actual_speed) == 0:
        actual_speed = np.linspace(20.0, 35.0, max(len(x), len(y), 10), dtype=float)

    curvature = estimate_curvature_from_xy(x, y)
    if curvature.size == 0:
        curvature = np.full(len(actual_speed), 0.05, dtype=float)

    radius = np.maximum(1.0 / np.clip(np.abs(curvature), 1e-6, None), 1.0)
    predicted_speed = np.array(
        [
            compute_racing_line_target_speed(
                curvature=float(c),
                radius=float(r),
                mu=float(mu),
                mass=float(mass),
                downforce=float(downforce),
            )
            for c, r in zip(curvature, radius)
        ],
        dtype=float,
    )

    actual_speed = np.asarray(actual_speed, dtype=float)
    if actual_speed.shape[0] != predicted_speed.shape[0]:
        length = min(actual_speed.shape[0], predicted_speed.shape[0])
        actual_speed = actual_speed[:length]
        predicted_speed = predicted_speed[:length]

    errors = predicted_speed - actual_speed
    metrics = {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "mean_error": float(np.mean(errors)),
    }

    return {
        "actual_speed": actual_speed.tolist(),
        "predicted_speed": predicted_speed.tolist(),
        "curvature": curvature.tolist(),
        "metrics": metrics,
    }
