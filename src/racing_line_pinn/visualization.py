"""Visualization helpers for comparing actual and ideal racing trajectories."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_racing_line_comparison(actual_x, actual_y, ideal_x, ideal_y, save_path: str = "outputs/racing_line_comparison.png"):
    """Plot actual versus predicted racing line overlay with a track-like background."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(actual_x, actual_y, color="tab:blue", linewidth=2.5, label="Actual driver trajectory")
    ax.plot(ideal_x, ideal_y, color="tab:orange", linewidth=2.0, linestyle="--", label="Ideal racing line")
    ax.scatter(actual_x[0], actual_y[0], color="green", s=26, label="Start")
    ax.set_title("Racing Line vs. Physics-Informed Ideal Line")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path
