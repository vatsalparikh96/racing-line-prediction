"""FastF1 integration scaffold for real telemetry ingestion.

This module intentionally validates the architecture: it provides a clean interface
for real-session retrieval, but gracefully degrades to synthetic data when FastF1
is unavailable or no session is specified.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TelemetrySession:
    circuit: str
    year: int
    driver: str
    session_name: str
    data: dict


def load_fastf1_session(circuit: str = "Spa-Francorchamps", year: int = 2023, driver: str = "VER", session_name: str = "R"):
    """Attempt to load a real FastF1 session when the dependency is available.

    Returns a lightweight session object, but if FastF1 is missing or the query fails,
    it falls back to a synthetic placeholder. This keeps the project runnable offline.
    """
    try:
        import fastf1  # type: ignore
    except Exception:
        return TelemetrySession(
            circuit=circuit,
            year=year,
            driver=driver,
            session_name=session_name,
            data={"x": [], "y": [], "speed": []},
        )

    try:
        session = fastf1.get_session(year, circuit, session_name)
        session.load()
        laps = session.laps.pick_driver(driver)
        telemetry = laps.pick_fastest().get_car_data().dropna()
        return TelemetrySession(
            circuit=circuit,
            year=year,
            driver=driver,
            session_name=session_name,
            data={
                "x": telemetry.get("X", []),
                "y": telemetry.get("Y", []),
                "speed": telemetry.get("Speed", []),
            },
        )
    except Exception:
        return TelemetrySession(
            circuit=circuit,
            year=year,
            driver=driver,
            session_name=session_name,
            data={"x": [], "y": [], "speed": []},
        )
