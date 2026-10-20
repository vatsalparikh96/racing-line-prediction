"""Training loop for the physics-informed racing-line speed regressor."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .calibration import VehicleParams
from .dataset import build_training_samples
from .model import RacingLinePINN, physics_loss


def train_model(
    s: np.ndarray,
    curvature: np.ndarray,
    vehicle: VehicleParams,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    device: str | None = None,
) -> tuple[nn.Module, torch.Tensor]:
    """Train the shared ideal-line model on one session's curvature profile.

    Returns the trained model and its predicted speed-vs-arc-length curve
    (same order/length as ``s``) so callers can reuse it without a second
    inference pass.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    curvature_t, target_speed_t, _corner_cap_t = build_training_samples(s, curvature, vehicle)
    model = RacingLinePINN(input_dim=1, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    x = curvature_t.unsqueeze(-1).to(device)
    y = target_speed_t.to(device)
    c = curvature_t.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x)
        loss = physics_loss(pred, c, y, vehicle)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        predicted_speed = model(x).cpu()

    return model, predicted_speed


def predict_speed(model: nn.Module, curvature: float) -> float:
    """Predict the shared ideal-line speed at a single curvature value."""
    tensor = torch.tensor([[curvature]], dtype=torch.float32)
    with torch.no_grad():
        return float(model(tensor).item())

