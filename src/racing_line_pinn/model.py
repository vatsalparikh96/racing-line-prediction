"""Simple PINN-style predictive model for racing-line speed estimation."""

from __future__ import annotations

import torch
from torch import nn

from .calibration import VehicleParams
from .physics import AIR_DENSITY_SEA_LEVEL


class RacingLinePINN(nn.Module):
    """A compact MLP predicting the shared "ideal line" speed from curvature alone.

    Vehicle parameters (mu, mass, aero) are calibrated constants (see
    :mod:`racing_line_pinn.calibration`), not model inputs -- there is one
    shared ideal-line model per session, not a model per vehicle setup.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def physics_loss(
    predicted_speed: torch.Tensor,
    curvature: torch.Tensor,
    target_speed: torch.Tensor,
    vehicle: VehicleParams,
    grip_weight: float = 5.0,
    lap_time_weight: float = 0.05,
) -> torch.Tensor:
    """Multi-term physics-informed loss: data fit + friction-limit penalty + lap time.

    - Data term: MSE between predicted and target speed.
    - Friction penalty: ``relu(predicted_speed - safe_speed) ** 2``, where
      ``safe_speed`` is the max cornering speed the calibrated tyres/aero can
      sustain at the predicted speed (speed-dependent downforce), a discrete
      analogue of the Pacejka combined friction ellipse limit.
    - Lap-time term: ``mean(1 / predicted_speed)``, a discrete proxy for the
      lap-time integral ``integral(1/v ds)`` over the batch of track samples,
      rewarding the model for finding higher (but still grip-legal) speeds.
    """
    eps = 1e-8
    effective_downforce = 0.5 * AIR_DENSITY_SEA_LEVEL * vehicle.aero_lift_area * predicted_speed.detach() ** 2
    normal_force = vehicle.mass * 9.81 + effective_downforce
    lateral_limit = vehicle.mu * normal_force / vehicle.mass
    safe_speed = torch.sqrt(torch.clamp(lateral_limit / torch.clamp(torch.abs(curvature), min=eps), min=eps))
    grip_penalty = torch.relu(predicted_speed - safe_speed)

    data_term = (predicted_speed - target_speed) ** 2
    lap_time_term = 1.0 / torch.clamp(predicted_speed, min=1.0)

    return torch.mean(data_term + grip_weight * grip_penalty ** 2 + lap_time_weight * lap_time_term)

