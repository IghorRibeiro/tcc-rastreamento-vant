#!/usr/bin/env python3
"""Generate the GM-FLIT gamma sweep figure from measured CSV artifacts."""
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


def load_sweep() -> pd.DataFrame:
    files = [
        MOT_DIR / "flit_nsa_sweep_summary.csv",
        MOT_DIR / "flit_nsa_sweep_summary_v2.csv",
    ]
    frames = [pd.read_csv(path).assign(source=str(path.relative_to(PROJECT_ROOT))) for path in files]
    sweep = pd.concat(frames, ignore_index=True)
    sweep["gamma"] = sweep["gamma"].astype(float)
    sweep = sweep.sort_values("gamma").drop_duplicates("gamma", keep="last").reset_index(drop=True)
    required = {"gamma", "MOTA(%)", "IDF1(%)", "IDs"}
    missing = required - set(sweep.columns)
    if missing:
        raise ValueError(f"Missing columns in gamma sweep: {sorted(missing)}")
    return sweep


def generate_figure6(output_dir: Path, csv_dir: Path) -> None:
    ieee_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    df = load_sweep()
    df.to_csv(csv_dir / "fig6_data.csv", index=False)

    gamma = df["gamma"].to_numpy()
    mota = df["MOTA(%)"].to_numpy()
    idf1 = df["IDF1(%)"].to_numpy()
    ids = df["IDs"].to_numpy()
    opt_index = int(np.lexsort((ids, -idf1, -mota))[0])
    opt_gamma = float(gamma[opt_index])

    fig, ax1 = plt.subplots(figsize=(5.8, 3.8))
    ax2 = ax1.twinx()
    ax1.plot(gamma, mota, "o-", color="black", label="MOTA", linewidth=1.4, markersize=5)
    ax1.plot(gamma, idf1, "s--", color="dimgray", label="IDF1", linewidth=1.4, markersize=5)
    ax2.plot(gamma, ids, "^ -".replace(" ", ""), color="royalblue", label="ID switches", linewidth=1.4, markersize=5)

    plateau_mask = mota >= (mota.max() - 0.3)
    if plateau_mask.any():
        plateau_indices = np.where(plateau_mask)[0]
        ax1.axvspan(gamma[plateau_indices[0]] - 0.04, gamma[plateau_indices[-1]] + 0.04, alpha=0.08, color="gray")

    ax1.axvline(opt_gamma, color="crimson", linestyle="--", linewidth=1.0)
    ax1.annotate(
        rf"$\gamma^*={opt_gamma:.2f}$",
        xy=(opt_gamma, mota[opt_index]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
        color="crimson",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "crimson", "alpha": 0.8},
    )

    ax1.set_xlabel(r"GM-FLIT weight, $\gamma$", fontsize=10)
    ax1.set_ylabel("MOTA / IDF1 (%)", fontsize=10)
    ax2.set_ylabel("ID switches", fontsize=10)
    ax1.set_xlim(max(0.1, gamma.min() - 0.05), min(0.9, gamma.max() + 0.05))
    ax1.set_ylim(65, 79)
    ax2.set_ylim(1400, 2700)
    ax1.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax2.grid(False)
    ax1.legend([ax1.lines[0], ax1.lines[1], ax2.lines[0]], ["MOTA", "IDF1", "ID switches"], loc="lower left", frameon=False, fontsize=8)
    ax1.set_title("GM-FLIT sensitivity on the held-out validation split", fontsize=11, pad=10)
    fig.text(0.01, 0.01, r"Data are read from the two stored sweep CSVs; $\gamma^*$ is selected by MOTA, then IDF1, then IDs.", fontsize=7, color="dimgray")
    fig.subplots_adjust(left=0.12, right=0.87, bottom=0.22, top=0.84)
    save_figure(fig, output_dir / "fig6_gamma_sweep")


if __name__ == "__main__":
    generate_figure6(PROJECT_ROOT / "paper_figures" / "output", PROJECT_ROOT / "paper_figures" / "csv")
