"""Command-line entrypoints for the racing-line project."""

from __future__ import annotations

import argparse

from .training import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate a physics-informed F1 racing-line estimator.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer size.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = train_model(epochs=args.epochs, learning_rate=args.lr, hidden_dim=args.hidden_dim)
    print(f"Model trained successfully: {type(model).__name__}")


if __name__ == "__main__":
    main()
