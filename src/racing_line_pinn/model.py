"""Simple PINN-style predictive model for racing-line speed estimation."""

from __future__ import annotations

import math

import torch
from torch import nn


class RacingLinePINN(nn.Module):
    """A compact MLP that predicts speed from track curvature and setup inputs."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32):
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


def physics_loss(predicted_speed: torch.Tensor, curvature: torch.Tensor, mu: torch.Tensor, downforce: torch.Tensor, mass: torch.Tensor, target_speed: torch.Tensor) -> torch.Tensor:
    """Physics-informed loss that penalizes grip violations and pushes exit speed higher."""
    eps = 1e-8
    lateral_limit = mu * (mass * 9.81 + downforce) / mass
    safe_speed = torch.sqrt(torch.clamp(lateral_limit * (1.0 / torch.clamp(torch.abs(curvature), min=eps)), min=eps))
    grip_penalty = torch.relu(predicted_speed - safe_speed)
    exit_speed_bonus = -0.01 * predicted_speed
    return torch.mean((predicted_speed - target_speed) ** 2 + 5.0 * grip_penalty ** 2 + 0.5 * exit_speed_bonus)


def synthetic_track_points(n: int = 200) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce synthetic curvature and setup samples for model demos."""
    t = torch.linspace(0.0, 1.0, n)
    curvature = 0.02 + 0.12 * torch.sin(2.0 * math.pi * t)
    mu = torch.full_like(t, 1.25)
    downforce = 2000.0 + 2500.0 * t
    mass = torch.full_like(t, 750.0)
    return curvature, mu, downforce, mass
