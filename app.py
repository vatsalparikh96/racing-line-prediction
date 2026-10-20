from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from racing_line_pinn.calibration import VehicleParams, calibrate_vehicle_params, load_vehicle_params, save_vehicle_params
from racing_line_pinn.evaluation import build_leaderboard
from racing_line_pinn.real_data import SpaSessionData, load_spa_race_session
from racing_line_pinn.training import train_model
from racing_line_pinn.visualization import build_animated_lap

TRAIN_EPOCHS = 200
TRAIN_LEARNING_RATE = 1e-3
TRAIN_HIDDEN_DIM = 64
FRAME_DT = 0.2
PLAYBACK_MULTIPLIERS = {"1x": 1, "2x": 2, "4x": 4}

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
    h1, h2, h3 {
        color: #f2f5fb !important;
        letter-spacing: 0.02em;
    }
    .stSelectbox, .stCheckbox, .stRadio {
        color: #edf4ff;
    }
    .block-container {
        padding-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Spa telemetry, calibrating, and training the ideal-line model...")
def _load_pipeline() -> tuple[SpaSessionData, VehicleParams, np.ndarray] | None:
    session = load_spa_race_session()
    if session is None:
        return None

    vehicle = load_vehicle_params()
    if vehicle is None:
        vehicle = calibrate_vehicle_params(session)
        save_vehicle_params(vehicle)

    _model, predicted_speed = train_model(
        session.s,
        session.curvature,
        vehicle,
        epochs=TRAIN_EPOCHS,
        learning_rate=TRAIN_LEARNING_RATE,
        hidden_dim=TRAIN_HIDDEN_DIM,
    )
    return session, vehicle, predicted_speed.numpy()


def _apply_playback_speed(fig, multiplier: int) -> None:
    """Speed up the Play button's frame rate without re-sampling the animation."""
    frame_duration_ms = max(int(FRAME_DT * 1000 / multiplier), 1)
    fig.update_layout(
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
        ]
    )


def main() -> None:
    st.set_page_config(page_title="Spa Racing Line", layout="wide")
    st.title("F1 Racing Line Prediction - Spa-Francorchamps")
    st.caption("2023 Race session: real telemetry vs. a physics-informed ideal-line model.")

    pipeline = _load_pipeline()
    if pipeline is None:
        st.error("Could not load the 2023 Spa race session from OpenF1 (network unavailable or no session data).")
        return
    session, vehicle, predicted_speed = pipeline

    driver_codes = sorted(session.drivers.keys())
    leaderboard = build_leaderboard(session.drivers, session.s, predicted_speed)

    with st.sidebar:
        st.header("View options")
        driver_code = st.selectbox("Driver", options=driver_codes, index=driver_codes.index(session.reference_driver))
        show_ghost = st.checkbox("Show model ghost", value=False)
        playback_label = st.radio("Playback speed", options=list(PLAYBACK_MULTIPLIERS.keys()), index=1, horizontal=True)
        st.caption(f"Calibrated: \u03bc={vehicle.mu:.2f}, mass={vehicle.mass:.0f} kg")
        if not session.drivers[driver_code].lap_complete:
            st.warning("Partial telemetry only; this driver did not complete a timed lap.")

    fig = build_animated_lap(
        session,
        driver_code,
        model_speed=predicted_speed if show_ghost else None,
        frame_dt=FRAME_DT,
    )
    _apply_playback_speed(fig, PLAYBACK_MULTIPLIERS[playback_label])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Leaderboard - time delta vs. the model")
    leaderboard_rows = [
        {
            "Driver": comparison.driver,
            "Lap Time (s)": round(session.drivers[comparison.driver].lap_time, 3),
            "\u0394 vs Model (s)": round(comparison.total_delta_t, 3),
            "RMSE (m/s)": round(comparison.rmse, 2),
        }
        for comparison in leaderboard
    ]
    leaderboard_df = pd.DataFrame(leaderboard_rows)

    def _highlight_selected(row: pd.Series) -> list[str]:
        is_selected = row["Driver"] == driver_code
        return ["background-color: #1d3a5f" if is_selected else "" for _ in row]

    st.dataframe(leaderboard_df.style.apply(_highlight_selected, axis=1), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
