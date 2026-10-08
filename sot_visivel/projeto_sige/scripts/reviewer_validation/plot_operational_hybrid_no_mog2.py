#!/usr/bin/env python3
"""Plot tracker metrics and operational prediction metrics for the hybrid without MOG2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HYBRID = "Híbrido_CV_Damped_sem_MOG2"
METHODS = [HYBRID, "DeepSORT", "ByteTrack", "BoT-SORT_YOLO26n", "BoT-SORT_ajustado_n"]
LABELS = {
    HYBRID: "Híbrido sem MOG2",
    "DeepSORT": "DeepSORT",
    "ByteTrack": "ByteTrack",
    "BoT-SORT_YOLO26n": "BoT-SORT padrão",
    "BoT-SORT_ajustado_n": "BoT-SORT ajustado",
}
COLORS = ["#16a34a", "#ef4444", "#3b82f6", "#64748b", "#94a3b8"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("scripts/reviewer_validation/results_with_hybrid_no_mog2/summary_mean_std_ci95.csv"))
    parser.add_argument("--exact", type=Path, default=Path("scripts/reviewer_validation/results_with_hybrid_no_mog2/hybrid_no_mog2_exact_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/reviewer_validation/results_with_hybrid_no_mog2/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary)
    exact = pd.read_csv(args.exact).set_index("metric")

    def common_value(method: str, metric: str, column: str = "mean") -> float:
        row = summary[(summary.method == method) & (summary.metric == metric)]
        return float(row[column].iloc[0])

    def common_std(method: str, metric: str) -> float:
        row = summary[(summary.method == method) & (summary.metric == metric)]
        return float(row["std_between_sequences"].iloc[0])

    # Tracking metrics are all from the common evaluator.
    panels = [
        ("SR@0.5", "SR@0,5", lambda method: common_value(method, "SR@0.5"), lambda method: common_std(method, "SR@0.5")),
        ("mean_IoU_present", "IoU médio", lambda method: common_value(method, "mean_IoU_present"), lambda method: common_std(method, "mean_IoU_present")),
        ("FPS_from_log", "FPS médio", lambda method: common_value(method, "FPS_from_log"), lambda method: common_std(method, "FPS_from_log")),
        ("pred_error_H15_exact", "Erro operacional H15 (px)", lambda method: float(exact.loc["pred_error_H15_mean_exact", "mean"]) if method == HYBRID else common_value(method, "pred_error_H15_mean"), lambda method: float(exact.loc["pred_error_H15_mean_exact", "std_between_sequences"]) if method == HYBRID else common_std(method, "pred_error_H15_mean")),
        ("pred_error_H30_exact", "Erro operacional H30 (px)", lambda method: float(exact.loc["pred_error_H30_mean_exact", "mean"]) if method == HYBRID else common_value(method, "pred_error_H30_mean"), lambda method: float(exact.loc["pred_error_H30_mean_exact", "std_between_sequences"]) if method == HYBRID else common_std(method, "pred_error_H30_mean")),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes_list = list(axes.flat)
    for ax, (metric, ylabel, value_fn, std_fn) in zip(axes_list, panels):
        vals = [value_fn(m) for m in METHODS]
        errs = [std_fn(m) for m in METHODS]
        bars = ax.bar(range(len(METHODS)), vals, yerr=errs, capsize=4, color=COLORS, edgecolor="#1f2937", linewidth=0.5)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for idx, value in enumerate(vals):
            ax.text(idx, value, f"{value:.3f}" if metric in {"SR@0.5", "mean_IoU_present"} else f"{value:.1f}", ha="center", va="bottom", fontsize=8.5)
    axes_list[-1].axis("off")
    fig.suptitle("Resultado operacional do híbrido final sem MOG2", fontsize=15)
    fig.text(0.5, 0.01, "SR, IoU e FPS: avaliador comum. H15/H30: erro do Kalman Damped=0,95 do híbrido; demais métodos: referência do avaliador comum.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(args.output_dir / "comparison_operational_hybrid_no_mog2.png", dpi=240)
    fig.savefig(args.output_dir / "comparison_operational_hybrid_no_mog2.pdf")
    plt.close(fig)
    print(f"Saved operational hybrid figure in {args.output_dir}")


if __name__ == "__main__":
    main()
