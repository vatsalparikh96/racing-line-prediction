"""Synthetic telemetry generation for racing-line visualization demos."""

from __future__ import annotations

import math

import numpy as np


def generate_track_geometry(n_points: int = 200):
    """Create a synthetic cornering track approximating a single hairpin-style section."""
    s = np.linspace(0.0, 1.0, n_points)
    x = 0.8 * np.cos(2.0 * np.pi * s) + 0.2 * np.sin(6.0 * np.pi * s)
    y = 0.5 * np.sin(2.0 * np.pi * s) + 0.15 * np.cos(4.0 * np.pi * s)
    curvature = 0.08 + 0.22 * np.sin(2.0 * np.pi * s)
    return x, y, curvature


def generate_telemetry_track(x: np.ndarray, y: np.ndarray, ideal_offset: float = 0.06):
    """Create actual and theoretical trajectory samples with a slight offset and noise."""
    rng = np.random.default_rng(42)
    actual_x = x + ideal_offset * np.sin(np.linspace(0.0, 2.0 * math.pi, len(x))) + 0.02 * rng.normal(size=len(x))
    actual_y = y + ideal_offset * np.cos(np.linspace(0.0, 2.0 * math.pi, len(y))) + 0.02 * rng.normal(size=len(y))
    ideal_x = x + 0.0 * np.ones_like(x)
    ideal_y = y + 0.0 * np.ones_like(y)
    return {
        "actual": {"x": actual_x, "y": actual_y},
        "ideal": {"x": ideal_x, "y": ideal_y},
    }
