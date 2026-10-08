#!/usr/bin/env python3
"""Create reviewer-ready plots from per-video tracker results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("SR@0.5", "SR@0.5", "higher"),
    ("mean_IoU_present", "Mean IoU (target-present frames)", "higher"),
    ("pred_error_H5_mean", "Prediction error H5 (px)", "lower"),
    ("pred_error_H10_mean", "Prediction error H10 (px)", "lower"),
    ("pred_error_H15_mean", "Prediction error H15 (px)", "lower"),
    ("pred_error_H30_mean", "Prediction error H30 (px)", "lower"),
    ("FPS_from_log", "FPS from per-frame log", "higher"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("scripts/reviewer_validation/results_final/summary_mean_std_ci95.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/reviewer_validation/results_final/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary)
    summary["label"] = summary["method"].str.replace("_", " ", regex=False)

    for metric, ylabel, direction in METRICS:
        df = summary[summary.metric == metric].copy()
        if df.empty:
            continue
        df = df.sort_values("mean", ascending=(direction == "lower"))
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.bar(df["label"], df["mean"], yerr=df["std_between_sequences"], capsize=4, color=["#3b82f6", "#f59e0b", "#64748b"][: len(df)])
        ax.set_ylabel(ylabel)
        ax.set_title(f"Tracker comparison: {ylabel}")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{metric}.png", dpi=220)
        fig.savefig(args.output_dir / f"{metric}.pdf")
        plt.close(fig)

    # Combined prediction-error panel for the talk.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, horizon in zip(axes, (15, 30)):
        metric = f"pred_error_H{horizon}_mean"
        df = summary[summary.metric == metric].copy().sort_values("mean")
        ax.bar(df["label"], df["mean"], yerr=df["std_between_sequences"], capsize=4, color=["#3b82f6", "#f59e0b", "#64748b"][: len(df)])
        ax.set_title(f"H{horizon}")
        ax.set_ylabel("Erro médio (px)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
    fig.suptitle("Variabilidade do erro de predição entre as 20 sequências")
    fig.tight_layout()
    fig.savefig(args.output_dir / "prediction_error_H15_H30.png", dpi=240)
    fig.savefig(args.output_dir / "prediction_error_H15_H30.pdf")
    plt.close(fig)
    print(f"Saved figures in {args.output_dir}")


if __name__ == "__main__":
    main()
