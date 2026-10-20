import numpy as np

from racing_line_pinn.evaluation import build_leaderboard, calculate_speed_metrics, compare_driver_to_model
from racing_line_pinn.real_data import DriverLapData


def _make_lap(driver: str, s: np.ndarray, speed: np.ndarray) -> DriverLapData:
    return DriverLapData(
        driver=driver,
        s=s,
        speed=speed,
        x=s,
        y=np.zeros_like(s),
        throttle=np.full_like(s, 100.0),
        brake=np.zeros_like(s),
        lap_time=float(np.trapezoid(1.0 / np.clip(speed, 1.0, None), s)),
    )


def test_speed_metrics_are_non_negative():
    actual = [30.0, 32.0, 28.0, 35.0]
    predicted = [31.0, 31.5, 29.0, 34.0]
    metrics = calculate_speed_metrics(actual, predicted)
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0


def test_compare_driver_to_model_zero_delta_when_identical():
    s = np.linspace(0.0, 100.0, 11)
    speed = np.full_like(s, 50.0)
    comparison = compare_driver_to_model("D1", s, speed, s, speed)
    assert comparison.rmse == 0.0
    assert comparison.total_delta_t == 0.0


def test_compare_driver_to_model_detects_slower_driver():
    s = np.linspace(0.0, 100.0, 11)
    model_speed = np.full_like(s, 50.0)
    driver_speed = np.full_like(s, 40.0)  # slower everywhere than the model
    comparison = compare_driver_to_model("D1", s, driver_speed, s, model_speed)
    assert comparison.total_delta_t > 0.0  # driver lost time relative to the model


def test_build_leaderboard_sorts_fastest_relative_to_model_first():
    s = np.linspace(0.0, 100.0, 11)
    model_speed = np.full_like(s, 50.0)
    drivers = {
        "SLOW": _make_lap("SLOW", s, np.full_like(s, 30.0)),
        "FAST": _make_lap("FAST", s, np.full_like(s, 48.0)),
    }
    leaderboard = build_leaderboard(drivers, s, model_speed)
    assert [comparison.driver for comparison in leaderboard] == ["FAST", "SLOW"]
