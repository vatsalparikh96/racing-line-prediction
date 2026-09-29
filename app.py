from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from racing_line_pinn.compare import calculate_speed_metrics
from racing_line_pinn.live_demo import CAR_PRESETS, TRACK_LIBRARY, simulate_session
from racing_line_pinn.real_data import load_fastf1_session
from racing_line_pinn.session_analysis import compare_session_to_model


st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: #050b14;
        color: #eaf2ff;
    }
    [data-testid="stSidebar"] {
        background: #0d1622;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .stMetric {
        background: rgba(18,30,43,0.9);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.02);
    }
    h1, h2, h3 {
        color: #f2f5fb !important;
        letter-spacing: 0.02em;
    }
    .stSlider > div > div > div {
        background: rgba(255,255,255,0.04);
    }
    .stSelectbox, .stCheckbox {
        color: #edf4ff;
    }
    .block-container {
        padding-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def build_lap_figure(session: dict, frame_index: int | None = None):
    if frame_index is None:
        frame_index = len(session["x"]) - 1
    frame_index = max(0, min(frame_index, len(session["x"]) - 1))

    ideal_x = session["x"][: frame_index + 1]
    ideal_y = session["y"][: frame_index + 1]
    actual_x = session["actual_x"][: frame_index + 1]
    actual_y = session["actual_y"][: frame_index + 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ideal_x, y=ideal_y, mode="lines", name="Ideal racing line", line=dict(color="#ff7f0e", width=4)))
    fig.add_trace(go.Scatter(x=actual_x, y=actual_y, mode="lines", name="Actual lap", line=dict(color="#4ea1ff", width=4)))
    fig.add_trace(go.Scatter(x=[session["apex_x"]], y=[session["apex_y"]], mode="markers", name="Apex", marker=dict(color="#ff3d3d", size=12, symbol="star-diamond")))

    fig.update_layout(
        template="plotly_dark",
        title="Track map and line comparison",
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        height=500,
        paper_bgcolor="#050b14",
        plot_bgcolor="#09131d",
        font=dict(color="#eaf2ff"),
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def build_speed_heatmap(session: dict):
    z = np.vstack([session["predicted_speed"], session["actual_speed"]])
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=np.linspace(0, 1, len(session["predicted_speed"])),
        y=["Model", "Actual"],
        colorscale="Turbo",
        hovertemplate="Lap fraction: %{x:.2f}<br>Speed: %{z:.1f} m/s<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Speed heatmap by lap segment",
        xaxis_title="Lap progress",
        yaxis_title="Signal",
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        paper_bgcolor="#050b14",
        plot_bgcolor="#09131d",
        font=dict(color="#eaf2ff"),
        annotations=[
            dict(
                text="Brighter colors = faster speed. Top row = physics model; bottom row = actual lap trace.",
                x=0.5,
                y=1.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color="#dfeaff"),
                align="center",
            )
        ],
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Racing Line Live Demo", layout="wide")
    st.title("F1 Optimal Racing Line Live Demo")
    st.caption("Real-world telemetry path is supported when FastF1 data is available; synthetic track data is a fallback demo mode.")

    with st.sidebar:
        st.header("Vehicle setup")
        data_mode = st.radio("Data source", ["Synthetic demo", "FastF1 telemetry"], index=0)
        track_name = st.selectbox("Track", options=list(TRACK_LIBRARY.keys()))
        car_name = st.selectbox("Car", options=list(CAR_PRESETS.keys()))
        st.caption("Preset tracks are stylized for the demo; real FastF1 sessions are used when available.")

        if car_name == "Custom":
            mu = st.slider("Tyre friction coefficient μ", 1.0, 1.8, 1.35, step=0.01)
            downforce = st.slider("Downforce [N]", 1000, 6000, 3000, step=50)
            mass = st.slider("Vehicle mass [kg]", 600, 950, 800, step=10)
            amplitude = st.slider("Track amplitude", 0.6, 1.8, 1.0, step=0.05)
        else:
            setup = CAR_PRESETS[car_name]
            mu = setup["mu"]
            downforce = setup["downforce"]
            mass = setup["mass"]
            amplitude = 1.0

    if data_mode == "FastF1 telemetry":
        telemetry = load_fastf1_session(circuit=track_name, year=2023, session_name="R")
        if telemetry.data.get("x") and telemetry.data.get("speed"):
            comparison = compare_session_to_model(
                {"x": telemetry.data["x"], "y": telemetry.data["y"], "speed": telemetry.data["speed"]},
                mu=mu,
                downforce=downforce,
                mass=mass,
            )
            session = {
                "x": np.asarray(telemetry.data["x"], dtype=float),
                "y": np.asarray(telemetry.data["y"], dtype=float),
                "actual_x": np.asarray(telemetry.data["x"], dtype=float),
                "actual_y": np.asarray(telemetry.data["y"], dtype=float),
                "predicted_speed": np.asarray(comparison["predicted_speed"], dtype=float),
                "actual_speed": np.asarray(comparison["actual_speed"], dtype=float),
                "curvature": np.asarray(comparison["curvature"], dtype=float),
                "apex_x": float(np.asarray(telemetry.data["x"]).mean()),
                "apex_y": float(np.asarray(telemetry.data["y"]).mean()),
                "track_name": telemetry.circuit,
            }
            st.success("FastF1 telemetry data loaded successfully for comparison.")
        else:
            st.warning("FastF1 data was not available in this environment. Falling back to the synthetic demo mode for the UI.")
            session = simulate_session(
                track_name=track_name,
                mu=mu,
                downforce=downforce,
                mass=mass,
                amplitude=amplitude,
            )
    else:
        session = simulate_session(
            track_name=track_name,
            mu=mu,
            downforce=downforce,
            mass=mass,
            amplitude=amplitude,
        )

    metrics = {
        "Predicted apex speed": f"{session['predicted_speed'].max():.1f} m/s",
        "Mean speed": f"{session['predicted_speed'].mean():.1f} m/s",
        "Peak curvature": f"{session['curvature'].max():.3f}",
    }

    cols = st.columns(3)
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)

    st.subheader("Actual vs ideal lap")
    trajectory_progress = st.slider("Lap progress", 0, len(session["x"]) - 1, len(session["x"]) - 1, key="lap_progress", label_visibility="collapsed")
    st.plotly_chart(build_lap_figure(session, frame_index=trajectory_progress), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Speed heatmap")
        st.caption("Each row is one speed signal. Top = physics model, bottom = actual lap speed. Brighter colors indicate higher speeds.")
        st.plotly_chart(build_speed_heatmap(session), use_container_width=True)
    with right:
        st.subheader("Speed trace")
        speed_df = pd.DataFrame({
            "Lap progress": np.linspace(0, 1, len(session["predicted_speed"])),
            "Ideal speed": session["predicted_speed"],
            "Actual speed": session["actual_speed"],
        })
        st.line_chart(speed_df.set_index("Lap progress"))

    st.caption(f"Track: {track_name} | Car: {car_name} | Setup: μ={mu:.2f}, downforce={downforce} N, mass={mass} kg")


if __name__ == "__main__":
    main()
