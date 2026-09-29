"""Evaluation utilities for comparing predicted and actual racing-line performance."""

from __future__ import annotations

import math

import numpy as np


def calculate_speed_metrics(actual_speed: list[float] | np.ndarray, predicted_speed: list[float] | np.ndarray) -> dict[str, float]:
    """Compute summary metrics for comparing actual and predicted speed traces."""
    actual = np.asarray(actual_speed, dtype=float)
    predicted = np.asarray(predicted_speed, dtype=float)
    errors = predicted - actual
    rmse = math.sqrt(float(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    mean_error = float(np.mean(errors))
    return {
        "rmse": rmse,
        "mae": mae,
        "mean_error": mean_error,
    }


def synthesize_driver_trajectory(n_points: int = 32):
    """Create a synthetic driver trajectory around a corner with apex-like behavior."""
    t = np.linspace(0.0, 1.0, n_points)
    x = np.linspace(-1.0, 1.0, n_points)
    y = 0.35 * np.sin(np.pi * t) - 0.1 * np.cos(2.0 * np.pi * t)
    return {"x": x, "y": y}
