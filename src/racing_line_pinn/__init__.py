"""Physics-informed racing line prediction package."""

from .compare import calculate_speed_metrics, synthesize_driver_trajectory
from .dataset import generate_synthetic_track_samples
from .physics import (
    calculate_lateral_grip_limit,
    compute_corner_speed_limit,
    compute_racing_line_target_speed,
)
from .real_data import TelemetrySession, load_fastf1_session
from .reporting import ComparisonReport, build_report
from .session_analysis import compare_session_to_model, estimate_curvature_from_xy
from .training import predict_speed, train_model

__all__ = [
    "calculate_lateral_grip_limit",
    "compute_corner_speed_limit",
    "compute_racing_line_target_speed",
    "generate_synthetic_track_samples",
    "train_model",
    "predict_speed",
    "calculate_speed_metrics",
    "synthesize_driver_trajectory",
    "TelemetrySession",
    "load_fastf1_session",
    "ComparisonReport",
    "build_report",
    "estimate_curvature_from_xy",
    "compare_session_to_model",
]
