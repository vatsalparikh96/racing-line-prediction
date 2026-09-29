"""Reporting utilities for summarizing lap-performance comparison results."""

from __future__ import annotations

from dataclasses import dataclass

from .compare import calculate_speed_metrics


@dataclass
class ComparisonReport:
    rmse: float
    mae: float
    mean_error: float
    driver_speed: list[float]
    predicted_speed: list[float]


def build_report(actual_speed: list[float], predicted_speed: list[float]) -> ComparisonReport:
    """Create a structured report for model-vs-driver speed comparison."""
    metrics = calculate_speed_metrics(actual_speed, predicted_speed)
    return ComparisonReport(
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        mean_error=metrics["mean_error"],
        driver_speed=list(actual_speed),
        predicted_speed=list(predicted_speed),
    )
