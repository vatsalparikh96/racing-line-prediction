"""Training loop for a simple physics-informed racing-line regressor."""

from __future__ import annotations

import torch
from torch import nn

from .dataset import generate_synthetic_track_samples
from .model import RacingLinePINN, physics_loss


def train_model(
    epochs: int = 200,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    device: str | None = None,
):
    """Train a compact model using synthetic track data and physics-aware loss."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    features, curvature, mu, downforce, mass, target_speed = generate_synthetic_track_samples()
    model = RacingLinePINN(input_dim=4, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    x = features.to(device)
    y = target_speed.to(device)
    c = curvature.to(device)
    mu_t = mu.to(device)
    d = downforce.to(device)
    m = mass.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x)
        loss = physics_loss(pred, c, mu_t, d, m, y)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

    return model


def predict_speed(model: nn.Module, curvature: float, radius: float, mu: float, downforce: float, mass: float):
    """Predict speed from a single track state."""
    tensor = torch.tensor([[curvature, radius, mu, downforce]], dtype=torch.float32)
    with torch.no_grad():
        return float(model(tensor).item())
