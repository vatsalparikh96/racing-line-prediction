"""Frenet-Serret track coordinate utilities.

Converts cartesian (X, Y) telemetry into track-relative (s, d) coordinates,
where s is arc length along a centerline and d is the signed lateral offset
from it. Also provides track-width estimation so that |d(s)| <= W(s)/2.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def compute_track_reference(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
) -> tuple[float, float, float]:
    """Compute a PCA-derived ``(mean_x, mean_y, angle)`` frame, in raw input units.

    Pass the result as ``reference`` to :func:`normalize_and_rotate` for every
    lap in a session so all laps share the exact same coordinate frame as the
    one used to derive this reference (typically the centerline lap), instead
    of each lap recomputing its own slightly different PCA rotation.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2:
        return 0.0, 0.0, 0.0

    mean_x, mean_y = float(x_arr.mean()), float(y_arr.mean())
    x_centered = x_arr - mean_x
    y_centered = y_arr - mean_y

    cov = np.cov(np.vstack([x_centered, y_centered]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    angle = float(np.arctan2(principal[1], principal[0]))
    return mean_x, mean_y, angle


def normalize_and_rotate(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    unit_scale: float = 0.1,
    reference: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Center, rescale, and PCA-rotate track coordinates.

    OpenF1 location X/Y is reported in decimeters; ``unit_scale=0.1`` converts
    it to meters. The rotation aligns the dominant (straight-line) axis of the
    track with the x-axis so straights render horizontally regardless of the
    original telemetry orientation.

    ``reference``, when given, is a ``(mean_x, mean_y, angle)`` tuple from
    :func:`compute_track_reference` (raw input units) that is applied instead
    of recomputing PCA from ``x``/``y`` themselves -- use this to align
    multiple drivers' laps to one shared frame (the centerline's).
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.size < 2:
        return x_arr * unit_scale, y_arr * unit_scale

    if reference is None:
        mean_x, mean_y, angle = compute_track_reference(x_arr, y_arr)
    else:
        mean_x, mean_y, angle = reference

    x_centered = (x_arr - mean_x) * unit_scale
    y_centered = (y_arr - mean_y) * unit_scale

    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    x_rot = cos_a * x_centered - sin_a * y_centered
    y_rot = sin_a * x_centered + cos_a * y_centered
    return x_rot, y_rot


def smooth_path(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    window_length: int = 15,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Lightly smooth a path's x/y before curvature estimation.

    Reduces telemetry-noise amplification in downstream curvature/derivative
    calculations. Falls back to the unsmoothed input when there are too few
    points for the requested window.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    window = min(window_length, x_arr.size)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder or window < 5:
        return x_arr, y_arr

    x_smooth = savgol_filter(x_arr, window_length=window, polyorder=polyorder)
    y_smooth = savgol_filter(y_arr, window_length=window, polyorder=polyorder)
    return x_smooth, y_smooth


def compute_curvature(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
) -> np.ndarray:
    """Estimate signed curvature kappa(s) = (x'y'' - y'x'') / (x'^2 + y'^2)^1.5.

    Expects an already-smoothed path (see :func:`smooth_path`); the analytic
    derivative formula is still noise-sensitive, so smoothing beforehand is
    what keeps this numerically well-behaved.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 3:
        return np.zeros_like(x_arr)

    dx = np.gradient(x_arr)
    dy = np.gradient(y_arr)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = np.power(dx ** 2 + dy ** 2, 1.5)
    denom = np.where(denom < 1e-9, 1e-9, denom)
    return (dx * ddy - dy * ddx) / denom


def cartesian_to_frenet(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    path_x: list[float] | np.ndarray | None = None,
    path_y: list[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert cartesian points to Frenet (s, d) coordinates along a centerline.

    When ``path_x``/``path_y`` are omitted, ``(x, y)`` is treated as its own
    centerline: s is the cumulative arc length and d is zero everywhere.
    Otherwise, each point is projected onto the nearest segment of the
    centerline to recover arc-length ``s`` and signed lateral offset ``d``
    (positive to the left of the direction of travel).
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if path_x is None or path_y is None:
        if x_arr.size < 2:
            return np.zeros_like(x_arr), np.zeros_like(x_arr)
        seg_len = np.hypot(np.diff(x_arr), np.diff(y_arr))
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        return s, np.zeros_like(s)

    path_x_arr = np.asarray(path_x, dtype=float)
    path_y_arr = np.asarray(path_y, dtype=float)
    if path_x_arr.size < 2:
        return np.zeros_like(x_arr), np.zeros_like(x_arr)

    seg_dx = np.diff(path_x_arr)
    seg_dy = np.diff(path_y_arr)
    seg_len = np.hypot(seg_dx, seg_dy)
    seg_len_sq = np.maximum(seg_dx ** 2 + seg_dy ** 2, 1e-12)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])

    ax = path_x_arr[:-1]
    ay = path_y_arr[:-1]

    s = np.zeros(x_arr.shape[0])
    d = np.zeros(x_arr.shape[0])

    for i in range(x_arr.shape[0]):
        px, py = x_arr[i], y_arr[i]
        t = np.clip(((px - ax) * seg_dx + (py - ay) * seg_dy) / seg_len_sq, 0.0, 1.0)
        proj_x = ax + t * seg_dx
        proj_y = ay + t * seg_dy
        dist = np.hypot(px - proj_x, py - proj_y)
        j = int(np.argmin(dist))
        cross = seg_dx[j] * (py - ay[j]) - seg_dy[j] * (px - ax[j])
        s[i] = cum_len[j] + t[j] * seg_len[j]
        d[i] = np.sign(cross) * dist[j] if seg_len[j] > 1e-9 else 0.0

    return s, d


def estimate_track_width(
    d: list[float] | np.ndarray,
    default_width: float = 12.0,
    margin: float = 1.2,
) -> np.ndarray:
    """Estimate a track width envelope W(s) that bounds observed lateral offsets.

    Returns a per-sample width array sized so that ``|d(s)| <= W(s)/2`` holds
    for the supplied offsets, with a safety margin and a sensible minimum.
    """
    d_arr = np.asarray(d, dtype=float)
    if d_arr.size == 0:
        return np.array([], dtype=float)

    observed_half_width = float(np.max(np.abs(d_arr))) * margin
    half_width = max(observed_half_width, default_width / 2.0)
    return np.full(d_arr.shape, 2.0 * half_width)
