#!/usr/bin/env python3
"""Regenerate the data-driven figures used by the paper.

Figure 7 is intentionally a separate command because it scans all validation
frames and selects a qualitative example; run
``paper_figures/figures/fig7_multitracker_comparison.py`` after this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PAPER_ROOT.parent
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src/inference"))

from figures.fig4_root_cause import generate_figure4
from figures.fig5_pipeline import generate_figure5
from figures.fig6_gamma import generate_figure6


def main() -> None:
    output_dir = PAPER_ROOT / "output"
    csv_dir = PAPER_ROOT / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Figure 4 — Root Cause Analysis...")
    generate_figure4(output_dir, csv_dir)
    print("Generating Figure 5 — Verified Pipeline Operating Points...")
    generate_figure5(output_dir, csv_dir)
    print("Generating Figure 6 — Data-driven Gamma Sweep...")
    generate_figure6(output_dir, csv_dir)
    print(f"\nFigures 4–6 saved in {output_dir}")
    print(f"CSV data saved in {csv_dir}")
    print("Run fig7_multitracker_comparison.py separately for the qualitative panel.")


if __name__ == "__main__":
    main()
