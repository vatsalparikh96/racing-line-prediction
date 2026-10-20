"""Vehicle parameter calibration from pooled real multi-driver telemetry.

Fits tyre friction (``mu``) and lumped aero coefficient-areas against the
physics relations in :mod:`racing_line_pinn.physics`, using every driver's
resampled speed/curvature samples from a single :class:`SpaSessionData`
(see :mod:`racing_line_pinn.real_data`). The result is a constant
:class:`VehicleParams` reused everywhere downstream -- there are no
user-tunable sliders.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from .physics import (
    calculate_aero_drag,
    calculate_speed_dependent_normal_load,
    pacejka_combined_friction_limit,
)
from .real_data import SpaSessionData

MASS_KG = 798.0  # FIA 2023 minimum car+driver weight; not observable from telemetry
APEX_CURVATURE_PERCENTILE = 80.0
STRAIGHT_CURVATURE_PERCENTILE = 25.0
DERIVATIVE_PERCENTILE = 95.0
DEFAULT_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "vehicle_params.json")


@dataclass
class VehicleParams:
    """Calibrated vehicle constants shared by every training/evaluation step."""

    mu: float
    aero_lift_area: float
    aero_drag_area: float
    max_accel: float
    max_decel: float
    mass: float = MASS_KG


def _smoothed_longitudinal_accel(s: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Return a_long(s) = v * dv/ds from a savgol-smoothed speed trace."""
    window = min(15, speed.size)
    if window % 2 == 0:
        window -= 1
    speed_smooth = savgol_filter(speed, window_length=window, polyorder=3) if window >= 5 else speed
    dv_ds = np.gradient(speed_smooth, s)
    return speed_smooth * dv_ds


def _pool_apex_samples(session: SpaSessionData, percentile: float) -> tuple[np.ndarray, np.ndarray]:
    """Pool (curvature, speed) across all drivers at the highest-curvature track points."""
    curvature = np.abs(session.curvature)
    threshold = np.percentile(curvature, percentile)
    mask = curvature >= threshold

    kappas = []
    speeds = []
    for lap in session.drivers.values():
        if not lap.lap_complete:
            continue
        kappas.append(curvature[mask])
        speeds.append(lap.speed[mask])
    return np.concatenate(kappas), np.concatenate(speeds)


def _fit_mu_and_lift_area(curvature_apex: np.ndarray, speed_apex: np.ndarray, mass: float) -> tuple[float, float]:
    """Fit (mu, lumped C_L*A) so the Pacejka grip limit matches observed apex lateral accel."""
    radius = 1.0 / np.maximum(curvature_apex, 1e-6)
    observed_lateral_accel = speed_apex ** 2 / radius

    def residuals(params: np.ndarray) -> np.ndarray:
        mu, lift_area = params
        predicted = np.empty_like(speed_apex)
        for i, v in enumerate(speed_apex):
            normal_force = calculate_speed_dependent_normal_load(mass, float(v), 1.0, lift_area)
            predicted[i] = pacejka_combined_friction_limit(mu, normal_force, mass, long_accel=0.0)
        return observed_lateral_accel - predicted

    result = least_squares(residuals, x0=[1.6, 3.0], bounds=([0.3, 1e-3], [3.0, 10.0]))
    mu, lift_area = result.x
    return float(mu), float(lift_area)


def _pool_longitudinal_samples(
    session: SpaSessionData, straight_percentile: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool smoothed longitudinal accel/decel samples, plus straight-line accel-phase speeds."""
    curvature = np.abs(session.curvature)
    straight_threshold = np.percentile(curvature, straight_percentile)
    straight_mask = curvature <= straight_threshold

    accel_values = []
    decel_values = []
    straight_speeds = []
    straight_accel = []
    for lap in session.drivers.values():
        if not lap.lap_complete:
            continue
        a_long = _smoothed_longitudinal_accel(session.s, lap.speed)
        accel_phase = a_long > 0
        decel_phase = a_long < 0

        accel_values.append(a_long[accel_phase])
        decel_values.append(np.abs(a_long[decel_phase]))

        straight_accel_phase = accel_phase & straight_mask
        straight_speeds.append(lap.speed[straight_accel_phase])
        straight_accel.append(a_long[straight_accel_phase])

    return (
        np.concatenate(accel_values),
        np.concatenate(decel_values),
        np.concatenate(straight_speeds),
        np.concatenate(straight_accel),
    )


def _fit_drag_area(speeds: np.ndarray, observed_accel: np.ndarray, max_accel: float, mass: float) -> float:
    """Fit the lumped C_D*A so max_accel - drag_decel(v) matches straight-line accel-phase data."""
    if speeds.size < 5:
        return 1.5  # not enough straight-line variation to identify drag; fall back to a plausible default

    def residuals(params: np.ndarray) -> np.ndarray:
        drag_area = params[0]
        predicted = max_accel - calculate_aero_drag(speeds, 1.0, drag_area) / mass
        return observed_accel - predicted

    result = least_squares(residuals, x0=[1.5], bounds=([1e-3], [10.0]))
    return float(result.x[0])


def calibrate_vehicle_params(
    session: SpaSessionData,
    mass: float = MASS_KG,
    apex_percentile: float = APEX_CURVATURE_PERCENTILE,
    straight_percentile: float = STRAIGHT_CURVATURE_PERCENTILE,
    derivative_percentile: float = DERIVATIVE_PERCENTILE,
) -> VehicleParams:
    """Calibrate shared vehicle parameters from pooled real multi-driver telemetry.

    ``mu`` and the lumped downforce coefficient-area are fit against near-apex
    (high-curvature) samples, where drivers are assumed to be at the lateral
    grip limit. ``max_accel``/``max_decel`` are the given percentile (default
    95th) of smoothed longitudinal accel/decel pooled across all drivers,
    robust to outlier spikes. The lumped drag coefficient-area is then fit
    against straight-line acceleration-phase samples, holding ``max_accel``
    fixed. Mass is fixed (not observable from telemetry), per the 2023 FIA
    minimum car+driver weight.
    """
    if not session.drivers:
        raise ValueError("session has no driver lap data to calibrate from.")

    curvature_apex, speed_apex = _pool_apex_samples(session, apex_percentile)
    mu, aero_lift_area = _fit_mu_and_lift_area(curvature_apex, speed_apex, mass)

    accel_values, decel_values, straight_speeds, straight_accel = _pool_longitudinal_samples(
        session, straight_percentile
    )
    max_accel = float(np.percentile(accel_values, derivative_percentile)) if accel_values.size else 8.0
    max_decel = float(np.percentile(decel_values, derivative_percentile)) if decel_values.size else 30.0
    aero_drag_area = _fit_drag_area(straight_speeds, straight_accel, max_accel, mass)

    return VehicleParams(
        mu=mu,
        aero_lift_area=aero_lift_area,
        aero_drag_area=aero_drag_area,
        max_accel=max_accel,
        max_decel=max_decel,
        mass=mass,
    )


def save_vehicle_params(params: VehicleParams, path: str = DEFAULT_PARAMS_PATH) -> None:
    """Persist calibrated params to JSON so they're computed once and reused as constants."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(asdict(params), handle, indent=2)


def load_vehicle_params(path: str = DEFAULT_PARAMS_PATH) -> VehicleParams | None:
    """Load previously persisted calibrated params, or ``None`` if not yet computed."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return VehicleParams(**data)
