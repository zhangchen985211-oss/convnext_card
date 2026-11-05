"""Create the fixed 11-node CAD graph used by ConvNeXt-CARD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cad_graph import build_speedplus_graph, load_kpts_mat, save_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kpts", required=True, help="path to kpts.mat")
    parser.add_argument("--output", required=True, help="output graph JSON")
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    points = load_kpts_mat(args.kpts)
    adjacency, distances = build_speedplus_graph(points)
    save_graph(output, points, adjacency, distances)
    print(f"saved CAD graph to {output.resolve()}")


if __name__ == "__main__":
    main()
