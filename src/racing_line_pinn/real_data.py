"""OpenF1 telemetry ingestion for Spa-Francorchamps sessions.

Loads timed laps and telemetry from OpenF1, builds one smoothed reference
centerline, and aligns every driver's speed/throttle/brake/position to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
import time

import numpy as np
import requests

from .frenet import (
    cartesian_to_frenet,
    compute_curvature,
    compute_track_reference,
    estimate_track_width,
    normalize_and_rotate,
    smooth_path,
)

CIRCUIT = "Spa-Francorchamps"
DEFAULT_YEAR = 2023
DEFAULT_SESSION = "R"
OPENF1_API = "https://api.openf1.org/v1"
OPENF1_REQUEST_INTERVAL = 0.4
OPENF1_PARTIAL_LAP_WINDOW = 180.0
OPENF1_WINDOW_MERGE_GAP = 300.0
_OPENF1_REQUEST_LOCK = Lock()
_OPENF1_LAST_REQUEST = 0.0
OPENF1_SESSION_NAMES = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "Q": "Qualifying",
    "R": "Race",
    "S": "Sprint",
    "SQ": "Sprint Shootout",
}


@dataclass
class DriverLapData:
    """One driver's fastest lap or available partial trace."""

    driver: str
    s: np.ndarray
    speed: np.ndarray
    x: np.ndarray
    y: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray
    lap_time: float | None
    lap_complete: bool = True


@dataclass
class SpaSessionData:
    """Session centerline and full or partial driver traces in its coordinate frame."""

    year: int
    session_name: str
    centerline_x: np.ndarray
    centerline_y: np.ndarray
    s: np.ndarray
    curvature: np.ndarray
    track_width: np.ndarray
    drivers: dict[str, DriverLapData]
    reference_driver: str


def _resample_on_grid(s_target: np.ndarray, s_source: np.ndarray, values: np.ndarray) -> np.ndarray:
    order = np.argsort(s_source)
    s_sorted = s_source[order]
    values_sorted = values[order]
    unique_s, unique_idx = np.unique(s_sorted, return_index=True)
    unique_values = values_sorted[unique_idx]
    return np.interp(s_target, unique_s, unique_values, left=unique_values[0], right=unique_values[-1])


def _openf1_get(endpoint: str, params: dict[str, object]) -> list[dict]:
    global _OPENF1_LAST_REQUEST

    for attempt in range(4):
        try:
            with _OPENF1_REQUEST_LOCK:
                wait = OPENF1_REQUEST_INTERVAL - (time.monotonic() - _OPENF1_LAST_REQUEST)
                if wait > 0:
                    time.sleep(wait)
                _OPENF1_LAST_REQUEST = time.monotonic()
            response = requests.get(f"{OPENF1_API}/{endpoint}", params=params, timeout=30)
            if response.status_code == 404:
                return []
            response.raise_for_status()
            rows = response.json()
            return rows if isinstance(rows, list) else []
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(0.5 * (2 ** attempt))
    return []


def _openf1_timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _openf1_lap_window(lap: dict) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(lap["date_start"].replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    duration = lap.get("lap_duration")
    if duration is None:
        duration = OPENF1_PARTIAL_LAP_WINDOW
    end = start + timedelta(seconds=float(duration))
    return start - timedelta(seconds=1), end + timedelta(seconds=1)


def _fetch_openf1_lap_telemetry(
    session_key: int,
    laps: dict[str, dict],
) -> dict[str, tuple[list[dict], list[dict]]]:
    windows = {number: _openf1_lap_window(lap) for number, lap in laps.items()}
    sorted_windows = sorted(windows.values())
    merged_windows: list[tuple[datetime, datetime]] = []
    for start, end in sorted_windows:
        if merged_windows and (start - merged_windows[-1][1]).total_seconds() <= OPENF1_WINDOW_MERGE_GAP:
            merged_windows[-1] = (merged_windows[-1][0], max(merged_windows[-1][1], end))
        else:
            merged_windows.append((start, end))

    telemetry: dict[str, dict[str, list[dict]]] = {
        number: {"location": [], "car_data": []} for number in laps
    }
    for start, end in merged_windows:
        params: dict[str, object] = {
            "session_key": session_key,
            "date>": start.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="milliseconds"),
            "date<": end.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="milliseconds"),
        }
        for endpoint in ("location", "car_data"):
            for row in _openf1_get(endpoint, params):
                number = str(row.get("driver_number"))
                if number not in telemetry:
                    continue
                try:
                    sample_time = _openf1_timestamp(row["date"])
                except (KeyError, TypeError, ValueError):
                    continue
                window_start, window_end = windows[number]
                if window_start.timestamp() <= sample_time <= window_end.timestamp():
                    telemetry[number][endpoint].append(row)

    return {
        number: (rows["location"], rows["car_data"])
        for number, rows in telemetry.items()
    }


def _load_openf1_session(year: int, session_name: str) -> SpaSessionData | None:
    openf1_session_name = OPENF1_SESSION_NAMES.get(session_name.upper(), session_name)
    try:
        sessions = _openf1_get(
            "sessions",
            {"year": year, "country_name": "Belgium", "session_name": openf1_session_name},
        )
        spa_sessions = [row for row in sessions if row.get("circuit_short_name") == CIRCUIT]
        if not spa_sessions:
            return None
        session_key = int(spa_sessions[0]["session_key"])
        driver_rows = _openf1_get("drivers", {"session_key": session_key})
        lap_rows = _openf1_get("laps", {"session_key": session_key})
    except (KeyError, TypeError, ValueError, requests.RequestException):
        return None

    driver_names = {
        str(row.get("driver_number")): str(row.get("name_acronym") or row.get("driver_number"))
        for row in driver_rows
        if row.get("driver_number") is not None
    }
    fastest_laps: dict[str, dict] = {}
    for lap in lap_rows:
        driver_number = lap.get("driver_number")
        duration = lap.get("lap_duration")
        if driver_number is None or lap.get("is_pit_out_lap") or not lap.get("date_start") or duration is None:
            continue
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(duration) or duration <= 0:
            continue
        key = str(driver_number)
        if key not in fastest_laps or duration < float(fastest_laps[key]["lap_duration"]):
            fastest_laps[key] = lap

    for driver in driver_rows:
        number = driver.get("driver_number")
        if number is None or str(number) in fastest_laps:
            continue
        partial_laps = [
            lap for lap in lap_rows
            if str(lap.get("driver_number")) == str(number)
            and lap.get("date_start")
            and lap.get("lap_duration") is None
            and not lap.get("is_pit_out_lap")
        ]
        if partial_laps:
            fastest_laps[str(number)] = min(partial_laps, key=lambda lap: lap.get("lap_number", 0))

    if not fastest_laps:
        return None

    raw_laps = _fetch_openf1_lap_telemetry(session_key, fastest_laps)
    missing_numbers = {
        number
        for number in driver_names
        if number not in raw_laps or not raw_laps[number][0] or not raw_laps[number][1]
    }
    if missing_numbers:
        missing_codes = sorted(driver_names[number] for number in missing_numbers)
        raise RuntimeError(
            "OpenF1 returned an incomplete driver roster; missing telemetry for: "
            + ", ".join(missing_codes)
            + ". Retry after the API rate limit clears."
        )
    if not raw_laps:
        return None

    complete_numbers = [
        number for number in raw_laps if fastest_laps[number].get("lap_duration") is not None
    ]
    reference_number = (
        min(complete_numbers, key=lambda number: float(fastest_laps[number]["lap_duration"]))
        if complete_numbers
        else max(raw_laps, key=lambda number: len(raw_laps[number][0]))
    )
    reference_location = sorted(raw_laps[reference_number][0], key=lambda row: row["date"])
    reference_x = np.asarray([row["x"] for row in reference_location], dtype=float)
    reference_y = np.asarray([row["y"] for row in reference_location], dtype=float)
    valid_reference = np.isfinite(reference_x) & np.isfinite(reference_y)
    reference_x, reference_y = reference_x[valid_reference], reference_y[valid_reference]
    if reference_x.size < 3:
        return None

    reference_frame = compute_track_reference(reference_x, reference_y)
    centerline_x, centerline_y = normalize_and_rotate(reference_x, reference_y, reference=reference_frame)
    centerline_x, centerline_y = smooth_path(centerline_x, centerline_y)
    s_center, d_center = cartesian_to_frenet(centerline_x, centerline_y)
    curvature = compute_curvature(centerline_x, centerline_y)
    track_width = estimate_track_width(d_center)

    drivers: dict[str, DriverLapData] = {}
    for number, (location_rows, car_rows) in raw_laps.items():
        location_rows = sorted(location_rows, key=lambda row: row["date"])
        car_rows = sorted(car_rows, key=lambda row: row["date"])
        location_times = np.asarray([_openf1_timestamp(row["date"]) for row in location_rows])
        raw_x = np.asarray([row["x"] for row in location_rows], dtype=float)
        raw_y = np.asarray([row["y"] for row in location_rows], dtype=float)
        valid = np.isfinite(location_times) & np.isfinite(raw_x) & np.isfinite(raw_y)
        location_times, raw_x, raw_y = location_times[valid], raw_x[valid], raw_y[valid]
        if raw_x.size < 3:
            raise RuntimeError(f"OpenF1 returned too few position samples for {driver_names.get(number, number)}.")

        car_times = np.asarray([_openf1_timestamp(row["date"]) for row in car_rows])
        order = np.argsort(car_times)
        car_times = car_times[order]
        car_rows = [car_rows[index] for index in order]
        if car_times.size < 2:
            raise RuntimeError(f"OpenF1 returned too few car samples for {driver_names.get(number, number)}.")

        def channel(name: str) -> np.ndarray:
            return np.asarray([float(row.get(name) or 0.0) for row in car_rows], dtype=float)

        lap_complete = fastest_laps[number].get("lap_duration") is not None
        if not lap_complete:
            moving_indices = np.flatnonzero(channel("speed") > 5.0)
            if moving_indices.size == 0:
                continue
            start_time = _openf1_timestamp(fastest_laps[number]["date_start"])
            stop_time = car_times[moving_indices[-1]]
            location_keep = (location_times >= start_time) & (location_times <= stop_time)
            car_keep = (car_times >= start_time) & (car_times <= stop_time)
            location_times, raw_x, raw_y = (
                location_times[location_keep],
                raw_x[location_keep],
                raw_y[location_keep],
            )
            car_times = car_times[car_keep]
            car_rows = [row for index, row in enumerate(car_rows) if car_keep[index]]
            if raw_x.size < 3 or len(car_rows) < 2:
                raise RuntimeError(f"OpenF1 returned too little partial telemetry for {driver_names.get(number, number)}.")

        speed = np.interp(location_times, car_times, channel("speed")) / 3.6
        throttle = np.clip(np.interp(location_times, car_times, channel("throttle")), 0.0, 100.0)
        brake_values = channel("brake")
        if brake_values.size and np.nanmax(brake_values) <= 1.0:
            brake_values *= 100.0
        brake = np.clip(np.interp(location_times, car_times, brake_values), 0.0, 100.0)

        x_m, y_m = normalize_and_rotate(raw_x, raw_y, reference=reference_frame)
        s_driver, _ = cartesian_to_frenet(x_m, y_m, path_x=centerline_x, path_y=centerline_y)
        if lap_complete:
            lap_s = s_center
        else:
            lap_s = s_center[(s_center >= np.min(s_driver)) & (s_center <= np.max(s_driver))]
            if lap_s.size < 3:
                raise RuntimeError(f"Could not align telemetry for {driver_names.get(number, number)} to the track.")
        speed_on_grid = _resample_on_grid(lap_s, s_driver, speed)
        throttle_on_grid = _resample_on_grid(lap_s, s_driver, throttle)
        brake_on_grid = _resample_on_grid(lap_s, s_driver, brake)
        x_on_grid = _resample_on_grid(lap_s, s_driver, x_m)
        y_on_grid = _resample_on_grid(lap_s, s_driver, y_m)

        code = driver_names.get(number, number)
        drivers[code] = DriverLapData(
            driver=code,
            s=lap_s,
            speed=speed_on_grid,
            x=x_on_grid,
            y=y_on_grid,
            throttle=throttle_on_grid,
            brake=brake_on_grid,
            lap_time=float(fastest_laps[number]["lap_duration"]) if lap_complete else None,
            lap_complete=lap_complete,
        )

    if not drivers:
        return None

    reference_driver = driver_names.get(reference_number, reference_number)
    return SpaSessionData(
        year=year,
        session_name=session_name,
        centerline_x=centerline_x,
        centerline_y=centerline_y,
        s=s_center,
        curvature=curvature,
        track_width=track_width,
        drivers=drivers,
        reference_driver=reference_driver,
    )


def load_spa_race_session(year: int = DEFAULT_YEAR, session_name: str = DEFAULT_SESSION) -> SpaSessionData | None:
    """Load Spa telemetry from OpenF1."""
    return _load_openf1_session(year, session_name)

