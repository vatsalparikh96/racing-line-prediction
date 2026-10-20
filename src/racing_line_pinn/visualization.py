"""Animated-lap visualization: 2D track map + throttle/brake/steering gauges."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .real_data import SpaSessionData

MAX_STEERING_DEG = 90.0
STEERING_REFERENCE_CURVATURE = 0.05  # scales curvature -> a plausible steering angle (approximation, not measured)
DEFAULT_FRAME_DT = 0.2
SLIDER_STEP_STRIDE = 10


def _cumulative_time(s: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Elapsed real time at each sample, via cumsum(ds / speed)."""
    ds = np.diff(s, prepend=s[0])
    return np.cumsum(ds / np.clip(speed, 1.0, None))


def _derive_steering_deg(curvature: np.ndarray) -> np.ndarray:
    """Approximate steering angle from local path curvature (the source has no steering channel)."""
    return np.clip(curvature / STEERING_REFERENCE_CURVATURE * MAX_STEERING_DEG, -MAX_STEERING_DEG, MAX_STEERING_DEG)


def build_animated_lap(
    session: SpaSessionData,
    driver_code: str,
    model_speed: np.ndarray | None = None,
    frame_dt: float = DEFAULT_FRAME_DT,
) -> go.Figure:
    """Build one animated figure: track map + throttle/brake/steering gauges.

    Left region: the shared centerline (light gray) plus the selected
    driver's actual path (bold) with an animated marker. Right region: 3
    `go.Indicator` gauges. The marker and gauges update together per frame via
    native Plotly `frames`/`updatemenus` (Play/Pause + scrubber) -- no manual
    Streamlit animation loop.

    Frames are sampled on a uniform REAL TIME grid (via each point's elapsed
    time ds/speed), not a uniform index grid, so constant-duration playback
    naturally speeds up on straights and slows in corners.

    When `model_speed` is given (the trained model's predicted speed on the
    same centerline `session.s`), an optional "ghost" marker rides the same
    centerline geometry on its OWN time axis, so it can visibly pull ahead of
    or fall behind the real driver; it freezes once it finishes its lap.
    """
    lap = session.drivers[driver_code]

    t_driver = _cumulative_time(lap.s, lap.speed)
    total_time = float(t_driver[-1]) if t_driver.size else 0.0
    time_grid = np.arange(0.0, total_time + frame_dt, frame_dt) if total_time > 0 else np.array([0.0])

    x_t = np.interp(time_grid, t_driver, lap.x)
    y_t = np.interp(time_grid, t_driver, lap.y)
    throttle_t = np.interp(time_grid, t_driver, lap.throttle)
    brake_t = np.interp(time_grid, t_driver, lap.brake)
    driver_curvature = np.interp(lap.s, session.s, session.curvature)
    curvature_t = np.interp(time_grid, t_driver, driver_curvature)
    steering_t = _derive_steering_deg(curvature_t)

    ghost_x_t: np.ndarray | None = None
    ghost_y_t: np.ndarray | None = None
    if model_speed is not None:
        t_model = _cumulative_time(session.s, np.asarray(model_speed, dtype=float))
        ghost_x_t = np.interp(time_grid, t_model, session.centerline_x, right=session.centerline_x[-1])
        ghost_y_t = np.interp(time_grid, t_model, session.centerline_y, right=session.centerline_y[-1])

    fig = make_subplots(
        rows=3,
        cols=2,
        column_widths=[0.62, 0.38],
        specs=[
            [{"type": "xy", "rowspan": 3}, {"type": "indicator"}],
            [None, {"type": "indicator"}],
            [None, {"type": "indicator"}],
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=session.centerline_x, y=session.centerline_y, mode="lines",
            line=dict(color="rgba(255,255,255,0.25)", width=2), name="Track centerline",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=lap.x, y=lap.y, mode="lines", line=dict(color="#4ea1ff", width=2), name=f"{driver_code} line"),
        row=1, col=1,
    )
    driver_marker_index = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[x_t[0]], y=[y_t[0]], mode="markers",
            marker=dict(color="#4ea1ff", size=14, symbol="circle"), name=driver_code,
        ),
        row=1, col=1,
    )

    ghost_marker_index = None
    if ghost_x_t is not None and ghost_y_t is not None:
        ghost_marker_index = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=[ghost_x_t[0]], y=[ghost_y_t[0]], mode="markers",
                marker=dict(color="#ff7f0e", size=14, symbol="diamond"), name="Model ghost",
            ),
            row=1, col=1,
        )

    throttle_index = len(fig.data)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number", value=float(throttle_t[0]), title={"text": "Throttle %"},
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#36d17a")),
        ),
        row=1, col=2,
    )
    brake_index = len(fig.data)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number", value=float(brake_t[0]), title={"text": "Brake %"},
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#ff4d4d")),
        ),
        row=2, col=2,
    )
    steering_index = len(fig.data)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number", value=float(steering_t[0]),
            title={"text": "Steering \u00b0 (derived, approximate)"},
            gauge=dict(axis=dict(range=[-MAX_STEERING_DEG, MAX_STEERING_DEG]), bar=dict(color="#4ea1ff")),
        ),
        row=3, col=2,
    )

    frames = []
    for i in range(time_grid.size):
        frame_data = [go.Scatter(x=[x_t[i]], y=[y_t[i]])]
        trace_indices = [driver_marker_index]
        if ghost_marker_index is not None:
            frame_data.append(go.Scatter(x=[ghost_x_t[i]], y=[ghost_y_t[i]]))
            trace_indices.append(ghost_marker_index)
        frame_data.extend(
            [go.Indicator(value=float(throttle_t[i])), go.Indicator(value=float(brake_t[i])), go.Indicator(value=float(steering_t[i]))]
        )
        trace_indices.extend([throttle_index, brake_index, steering_index])
        frames.append(go.Frame(data=frame_data, traces=trace_indices, name=f"{time_grid[i]:.2f}"))

    fig.frames = frames

    frame_duration_ms = max(int(frame_dt * 1000), 10)
    fig.update_layout(
        template="plotly_dark",
        title=f"{driver_code} -- animated lap (real elapsed time)",
        paper_bgcolor="#050b14",
        plot_bgcolor="#09131d",
        font=dict(color="#eaf2ff"),
        height=560,
        margin=dict(l=20, r=20, t=60, b=40),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05, y=1.12, xanchor="left", yanchor="top",
                buttons=[
                    dict(
                        label="Play", method="animate",
                        args=[None, {"frame": {"duration": frame_duration_ms, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                    ),
                    dict(
                        label="Pause", method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.05, y=-0.02, len=0.6,
                steps=[
                    dict(
                        method="animate", label=f"{t:.0f}s",
                        args=[[f"{t:.2f}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    )
                    for t in time_grid[::SLIDER_STEP_STRIDE]
                ],
            )
        ],
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, gridcolor="rgba(255,255,255,0.08)", row=1, col=1)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", row=1, col=1)
    return fig

