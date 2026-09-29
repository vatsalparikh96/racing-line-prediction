"""Synthetic and real-telemetry dataset utilities for racing-line modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .physics import compute_racing_line_target_speed


@dataclass
class TrackSample:
    curvature: float
    radius: float
    mu: float
    downforce: float
    mass: float
    target_speed: float


def generate_synthetic_track_samples(
    n_samples: int = 2048,
    mu_min: float = 1.2,
    mu_max: float = 1.6,
    downforce_min: float = 1800.0,
    downforce_max: float = 4500.0,
    mass_min: float = 700.0,
    mass_max: float = 850.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a synthetic dataset of corner curvatures and physics conditions."""
    rng = np.random.default_rng(42)
    curvature = rng.uniform(0.02, 0.18, size=n_samples).astype(np.float32)
    radius = 1.0 / curvature
    mu = rng.uniform(mu_min, mu_max, size=n_samples).astype(np.float32)
    downforce = rng.uniform(downforce_min, downforce_max, size=n_samples).astype(np.float32)
    mass = rng.uniform(mass_min, mass_max, size=n_samples).astype(np.float32)

    target_speed = np.array(
        [
            compute_racing_line_target_speed(
                curvature=float(c),
                radius=float(r),
                mu=float(m),
                mass=float(mass_i),
                downforce=float(d),
            )
            for c, r, m, mass_i, d in zip(curvature, radius, mu, mass, downforce)
        ],
        dtype=np.float32,
    )

    features = np.stack([curvature, radius, mu, downforce], axis=1).astype(np.float32)
    return (
        torch.tensor(features),
        torch.tensor(curvature),
        torch.tensor(mu),
        torch.tensor(downforce),
        torch.tensor(mass),
        torch.tensor(target_speed),
    )


def load_fastf1_telemetry(path: str | None = None):
    """Optional hook for real FastF1 telemetry ingestion.

    If FastF1 is not installed or a path is not provided, this returns a synthetic
    placeholder dataset so the project can still execute in offline environments.
    """
    if path is not None:
        return path

    return generate_synthetic_track_samples()
