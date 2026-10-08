#!/usr/bin/env python3
"""Generate the verified operating-point figure from measured CSV artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src/inference"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from paper_figures.metrics import ieee_style, save_figure
except ModuleNotFoundError:  # supports running from inside paper_figures/
    from metrics import ieee_style, save_figure

MOT_DIR = PROJECT_ROOT / "runs" / "mot_eval"


def load_overall(path: Path) -> dict[str, float | int]:
    """Read one stored readable MOT summary and return its OVERALL row."""
    frame = pd.read_csv(path, index_col=0)
    if "OVERALL" not in frame.index:
        raise ValueError(f"Missing OVERALL row in {path}")
    row = frame.loc["OVERALL"]
    return {
        "MOTA": float(row["MOTA(%)"]),
        "IDF1": float(row["IDF1(%)"]),
        "ID_switches": int(row["IDs"]),
        "source": str(path.relative_to(PROJECT_ROOT)),
    }


def load_gamma_point(path: Path, gamma: float) -> dict[str, float | int]:
    frame = pd.read_csv(path)
    row = frame[np.isclose(frame["gamma"].astype(float), gamma)]
    if row.empty:
        raise ValueError(f"Gamma {gamma} missing in {path}")
    point = row.iloc[0]
    return {
        "MOTA": float(point["MOTA(%)"]),
        "IDF1": float(point["IDF1(%)"]),
        "ID_switches": int(point["IDs"]),
        "source": str(path.relative_to(PROJECT_ROOT)) + f" [gamma={gamma:g}]",
    }


def generate_figure5(output_dir: Path, csv_dir: Path) -> None:
    ieee_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # These are measured operating points, not a claim that every row is a
    # detector-paired cumulative ablation. The source column is exported with
    # the figure data to make that distinction explicit.
    stages = [
        ("Baseline", load_overall(MOT_DIR / "mot_summary_botsort_readable.csv")),
        ("GM-FLIT + NSA", load_gamma_point(MOT_DIR / "flit_nsa_sweep_summary.csv", 0.55)),
        ("HardMining detector", load_overall(MOT_DIR / "mot_summary_hardmining_readable.csv")),
        ("HardMining + SAHI512", load_overall(MOT_DIR / "mot_summary_hardmining_sahi512_readable.csv")),
        ("FLIT-Tracker", load_overall(MOT_DIR / "mot_summary_hardmining_sahi512_flit_nsa_readable.csv")),
    ]
    rows = []
    for label, values in stages:
        rows.append({"Stage": label, **values})
    df = pd.DataFrame(rows)
    df.to_csv(csv_dir / "fig5_data.csv", index=False)

    x = np.arange(len(df))
    width = 0.34
    fig, ax1 = plt.subplots(figsize=(5.8, 3.8))
    ax2 = ax1.twinx()
    color_mota = "#2C3E50"
    color_idf1 = "#5DADE2"
    color_ids = "#E67E22"

    bars1 = ax1.bar(x - width / 2, df["MOTA"], width, color=color_mota, label="MOTA")
    bars2 = ax1.bar(x + width / 2, df["IDF1"], width, color=color_idf1, label="IDF1")
    line_ids, = ax2.plot(
        x,
        df["ID_switches"],
        "o-",
        color=color_ids,
        linewidth=1.4,
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label="ID switches",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        ["Baseline", "GM-FLIT\n+ NSA", "HardMining\ndetector", "HardMining\n+ SAHI512", "FLIT-\nTracker"],
        fontsize=8,
    )
    ax1.set_ylabel("MOTA / IDF1 (%)", fontsize=10)
    ax2.set_ylabel("ID switches", fontsize=10)
    ax1.set_ylim(60, 85)
    ax2.set_ylim(350, 2600)
    ax1.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax2.grid(False)

    for index, (mota, idf1) in enumerate(zip(df["MOTA"], df["IDF1"])):
        ax1.annotate(f"{mota:.1f}", (index - width / 2, mota), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, color=color_mota, fontweight="bold")
        ax1.annotate(f"{idf1:.1f}", (index + width / 2, idf1), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, color=color_idf1, fontweight="bold")
        ax2.annotate(f"{int(df.loc[index, 'ID_switches'])}", (index, df.loc[index, "ID_switches"]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5, color=color_ids, fontweight="bold")

    ax1.legend([bars1, bars2, line_ids], ["MOTA", "IDF1", "ID switches"], loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False, fontsize=8)
    ax1.set_title("Measured operating points across the validated pipeline", fontsize=11, pad=16)
    fig.text(0.01, 0.01, "Rows are sourced from stored MOT summaries; they are not all detector-paired ablations.", fontsize=7, color="dimgray")
    fig.subplots_adjust(left=0.12, right=0.87, bottom=0.25, top=0.82)
    save_figure(fig, output_dir / "fig5_pipeline_evolution")


if __name__ == "__main__":
    generate_figure5(PROJECT_ROOT / "paper_figures" / "output", PROJECT_ROOT / "paper_figures" / "csv")
