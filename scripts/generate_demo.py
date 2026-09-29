from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_line_pinn.telemetry import generate_telemetry_track, generate_track_geometry
from racing_line_pinn.visualization import plot_racing_line_comparison


def main():
    x, y, curvature = generate_track_geometry(220)
    telemetry = generate_telemetry_track(x, y, ideal_offset=0.05)
    output_path = plot_racing_line_comparison(
        telemetry["actual"]["x"],
        telemetry["actual"]["y"],
        telemetry["ideal"]["x"],
        telemetry["ideal"]["y"],
        save_path="outputs/racing_line_comparison.png",
    )
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
