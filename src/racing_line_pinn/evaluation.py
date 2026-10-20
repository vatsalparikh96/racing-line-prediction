"""Consolidated evaluation utilities: per-driver vs. model speed comparison.

Merges the old ``compare.py`` / ``session_analysis.py`` / ``reporting.py`` metric
logic into one module. Unlike the old dict-based session format (which had to
estimate its own curvature/arc-length from raw x/y), every driver and the
model now already share one common centerline arc-length ``s`` grid (see
``real_data.py``), so comparison here is a direct point-by-point operation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .real_data import DriverLapData


def calculate_speed_metrics(
    actual_speed: list[float] | np.ndarray, predicted_speed: list[float] | np.ndarray
) -> dict[str, float]:
    """Compute summary metrics for comparing actual and predicted speed traces."""
    actual = np.asarray(actual_speed, dtype=float)
    predicted = np.asarray(predicted_speed, dtype=float)
    errors = predicted - actual
    return {
        "rmse": math.sqrt(float(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "mean_error": float(np.mean(errors)),
    }


def compute_delta_traces(
    s: np.ndarray, actual_speed: np.ndarray, predicted_speed: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute distance-aligned speed delta and cumulative time-loss curves.

    Delta v(s) = predicted_speed - actual_speed (positive means the model
    expects more speed than the driver actually carried). Delta t(s) is the
    cumulative integral of (ds / actual_speed - ds / predicted_speed): a
    running total, in seconds, of time the actual driver lost (positive) or
    gained (negative) relative to the model at each point along the lap.
    """
    s_arr = np.asarray(s, dtype=float)
    actual = np.clip(np.asarray(actual_speed, dtype=float), 1e-3, None)
    predicted = np.clip(np.asarray(predicted_speed, dtype=float), 1e-3, None)
    ds = np.diff(s_arr, prepend=s_arr[0])
    delta_t = np.cumsum(ds / actual - ds / predicted)
    return {"delta_v": predicted - actual, "delta_t": delta_t}


@dataclass
class DriverComparison:
    """Per-driver comparison result against the shared model speed curve."""

    driver: str
    s: np.ndarray
    delta_v: np.ndarray
    delta_t: np.ndarray
    rmse: float
    mae: float
    mean_error: float
    total_delta_t: float


def compare_driver_to_model(
    driver: str,
    driver_s: np.ndarray,
    driver_speed: np.ndarray,
    model_s: np.ndarray,
    model_speed: np.ndarray,
) -> DriverComparison:
    """Compare one driver's real speed trace to the model's predicted speed.

    Both sides are expected to already share the common centerline ``s`` grid
    (see ``real_data.py``); the model trace is re-interpolated onto
    ``driver_s`` only if the grids differ (defensive, not expected in normal
    use since every driver is resampled onto the same centerline).
    """
    driver_s_arr = np.asarray(driver_s, dtype=float)
    driver_speed_arr = np.asarray(driver_speed, dtype=float)
    model_s_arr = np.asarray(model_s, dtype=float)
    model_speed_arr = np.asarray(model_speed, dtype=float)

    if model_s_arr.shape != driver_s_arr.shape or not np.allclose(model_s_arr, driver_s_arr):
        model_speed_arr = np.interp(driver_s_arr, model_s_arr, model_speed_arr)

    metrics = calculate_speed_metrics(driver_speed_arr, model_speed_arr)
    deltas = compute_delta_traces(driver_s_arr, driver_speed_arr, model_speed_arr)

    return DriverComparison(
        driver=driver,
        s=driver_s_arr,
        delta_v=deltas["delta_v"],
        delta_t=deltas["delta_t"],
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        mean_error=metrics["mean_error"],
        total_delta_t=float(deltas["delta_t"][-1]) if deltas["delta_t"].size else 0.0,
    )


def build_leaderboard(
    drivers: dict[str, DriverLapData], model_s: np.ndarray, model_speed: np.ndarray
) -> list[DriverComparison]:
    """Compare every driver against the model, sorted fastest-vs-model first."""
    comparisons = [
        compare_driver_to_model(code, lap.s, lap.speed, model_s, model_speed)
        for code, lap in drivers.items()
        if lap.lap_complete
    ]
    comparisons.sort(key=lambda comparison: comparison.total_delta_t)
    return comparisons
