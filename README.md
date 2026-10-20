# Physics-Informed ML for F1 Optimal Racing Line & Apex Speed Prediction

This project builds a small physics-informed machine learning workflow to estimate an optimal racing line and apex speed for a circuit under varying downforce conditions. It combines a simple vehicle dynamics model, a neural network surrogate for track-speed prediction, and a plotting pipeline for comparing a theoretical ideal line against telemetry-style trajectory data.

## Goals

- Model maximum cornering speed using tire friction and downforce effects.
- Estimate a theoretically optimal racing line based on curvature and speed limits.
- Create a physics-aware loss signal that penalizes sliding beyond grip limits.
- Produce a visual comparison against a baseline race trajectory.

## Project layout

- `src/racing_line_pinn/` — package code for physics, model, and visualization.
- `tests/` — verification tests for the vehicle dynamics functions.
- `notebooks/` — optional exploration notebooks.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -e .
pytest
.venv\Scripts\streamlit run app.py
```

## Example

```python
from racing_line_pinn.physics import compute_racing_line_target_speed

speed = compute_racing_line_target_speed(
    curvature=0.05,
    radius=20.0,
    mu=1.3,
    mass=750.0,
    downforce=2800.0,
)
print(f"Target speed (m/s): {speed:.2f}")
```

## Notes

This repository includes physics-based guidance, synthetic track generation, a small training loop, telemetry comparison utilities, real-data ingestion from OpenF1, and session-level comparison logic.

The intended workflow is:

1. ingest track or telemetry data,
2. estimate speed limits from physics constraints,
3. train a neural model to fit corner-speed targets,
4. compare model output to actual driver trajectory and speed traces,
5. visualize the delta using track overlays and speed profiles,
6. evaluate model error using RMSE, MAE, and mean error summaries.
