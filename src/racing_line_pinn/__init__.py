"""Physics-informed racing line prediction package."""

from .calibration import VehicleParams, calibrate_vehicle_params, load_vehicle_params, save_vehicle_params
from .dataset import build_training_samples
from .evaluation import (
    DriverComparison,
    build_leaderboard,
    calculate_speed_metrics,
    compare_driver_to_model,
    compute_delta_traces,
)
from .frenet import cartesian_to_frenet, estimate_track_width, normalize_and_rotate
from .physics import (
    calculate_aero_downforce,
    calculate_aero_drag,
    calculate_lateral_grip_limit,
    calculate_speed_dependent_normal_load,
    compute_corner_speed_limit,
    compute_racing_line_target_speed,
    compute_target_speed_with_aero,
    compute_two_track_weight_transfer,
    distribute_wheel_loads,
    pacejka_combined_friction_limit,
)
from .real_data import DriverLapData, SpaSessionData, load_spa_race_session
from .training import predict_speed, train_model
from .visualization import build_animated_lap

__all__ = [
    "calculate_lateral_grip_limit",
    "compute_corner_speed_limit",
    "compute_racing_line_target_speed",
    "calculate_aero_downforce",
    "calculate_aero_drag",
    "calculate_speed_dependent_normal_load",
    "pacejka_combined_friction_limit",
    "compute_two_track_weight_transfer",
    "distribute_wheel_loads",
    "compute_target_speed_with_aero",
    "cartesian_to_frenet",
    "estimate_track_width",
    "normalize_and_rotate",
    "build_training_samples",
    "train_model",
    "predict_speed",
    "calculate_speed_metrics",
    "compute_delta_traces",
    "compare_driver_to_model",
    "build_leaderboard",
    "DriverComparison",
    "DriverLapData",
    "SpaSessionData",
    "load_spa_race_session",
    "VehicleParams",
    "calibrate_vehicle_params",
    "save_vehicle_params",
    "load_vehicle_params",
    "build_animated_lap",
]
