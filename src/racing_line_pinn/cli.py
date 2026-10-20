"""Command-line entrypoints for the racing-line project."""

from __future__ import annotations

import argparse

import numpy as np

from .calibration import calibrate_vehicle_params, load_vehicle_params, save_vehicle_params
from .evaluation import build_leaderboard
from .real_data import load_spa_race_session
from .training import train_model


def _lap_time_from_speed(s: np.ndarray, speed: np.ndarray) -> float:
    """Trapezoidal-integrate 1/v ds over a speed trace to estimate elapsed time."""
    inv_speed = 1.0 / np.clip(speed, 1.0, None)
    return float(np.trapezoid(inv_speed, s))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the physics-informed Spa ideal-line model and print a driver leaderboard."
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer size.")
    parser.add_argument("--year", type=int, default=2023, help="Season year.")
    parser.add_argument("--session", type=str, default="R", help="Session identifier (e.g. R for Race).")
    parser.add_argument(
        "--recalibrate", action="store_true", help="Recompute vehicle params even if a cached JSON exists."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    session = load_spa_race_session(year=args.year, session_name=args.session)
    if session is None:
        print("Could not load Spa session data from OpenF1 (network unavailable or no session data).")
        return

    vehicle = None if args.recalibrate else load_vehicle_params()
    if vehicle is None:
        vehicle = calibrate_vehicle_params(session)
        save_vehicle_params(vehicle)
    print(
        f"Vehicle params: mu={vehicle.mu:.3f}, aero_lift_area={vehicle.aero_lift_area:.3f}, "
        f"aero_drag_area={vehicle.aero_drag_area:.3f}, max_accel={vehicle.max_accel:.2f}, "
        f"max_decel={vehicle.max_decel:.2f}"
    )

    model, predicted_speed = train_model(
        session.s,
        session.curvature,
        vehicle,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
    )
    model_lap_time = _lap_time_from_speed(session.s, predicted_speed.numpy())
    print(f"\nModel ideal lap time: {model_lap_time:.3f}s ({type(model).__name__})")

    leaderboard = build_leaderboard(session.drivers, session.s, predicted_speed.numpy())

    print(f"\n{'Driver':<8}{'LapTime':>10}{'Delta vs model':>18}{'RMSE':>10}")
    for comparison in leaderboard:
        lap_time = session.drivers[comparison.driver].lap_time
        print(
            f"{comparison.driver:<8}{lap_time:>10.3f}{comparison.total_delta_t:>+18.3f}{comparison.rmse:>10.2f}"
        )


if __name__ == "__main__":
    main()

