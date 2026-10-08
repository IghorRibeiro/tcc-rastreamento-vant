#!/usr/bin/env python3
"""Plot the final common-evaluator comparison with the hybrid without MOG2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("scripts/reviewer_validation/results_with_hybrid_no_mog2/summary_mean_std_ci95.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/reviewer_validation/results_with_hybrid_no_mog2/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary)
    methods = [
        "Híbrido_CV_Damped_sem_MOG2",
        "DeepSORT",
        "ByteTrack",
        "BoT-SORT_YOLO26n",
        "BoT-SORT_ajustado_n",
    ]
    labels = {
        "Híbrido_CV_Damped_sem_MOG2": "Híbrido final\nsem MOG2",
        "DeepSORT": "DeepSORT",
        "ByteTrack": "ByteTrack",
        "BoT-SORT_YOLO26n": "BoT-SORT\npadrão",
        "BoT-SORT_ajustado_n": "BoT-SORT\najustado",
    }
    colors = ["#16a34a", "#ef4444", "#3b82f6", "#64748b", "#94a3b8"]
    metric_specs = [
        ("SR@0.5", "SR@0,5", "higher", True),
        ("mean_IoU_present", "IoU médio", "higher", True),
        ("pred_error_H15_mean", "Erro H15 (px)", "lower", False),
        ("pred_error_H30_mean", "Erro H30 (px)", "lower", False),
        ("FPS_from_log", "FPS médio", "higher", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.flat
    for ax, (metric, ylabel, _, is_fraction) in zip(axes, metric_specs):
        rows = summary[summary.metric == metric].set_index("method")
        present = [m for m in methods if m in rows.index]
        vals = [float(rows.loc[m, "mean"]) for m in present]
        errs = [float(rows.loc[m, "std_between_sequences"]) if metric not in {"SR@0.5", "mean_IoU_present", "FPS_from_log"} else (float(rows.loc[m, "std_between_sequences"]) if metric == "mean_IoU_present" else 0.0) for m in present]
        # SR and FPS are shown without error bars here to keep the panel readable;
        # all uncertainty remains in the CSV summary.
        if metric in {"SR@0.5", "FPS_from_log"}:
            errs = [0.0] * len(present)
        bars = ax.bar(range(len(present)), vals, yerr=errs, capsize=4, color=[colors[methods.index(m)] for m in present], edgecolor="#1f2937", linewidth=0.5)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([labels[m] for m in present], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for idx, value in enumerate(vals):
            ax.text(idx, value, f"{value:.3f}" if is_fraction else f"{value:.1f}", ha="center", va="bottom", fontsize=8.5)
    axes[-1].axis("off")
    fig.suptitle("Comparação final: híbrido sem MOG2 e trackers de referência", fontsize=15)
    fig.text(0.5, 0.01, "Todos os métodos foram avaliados nas mesmas 20 sequências e 24.804 imagens; barras de erro representam o desvio-padrão entre sequências quando exibido.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(args.output_dir / "comparison_final_no_mog2.png", dpi=240)
    fig.savefig(args.output_dir / "comparison_final_no_mog2.pdf")
    plt.close(fig)
    print(f"Saved final no-MOG2 comparison figure in {args.output_dir}")


if __name__ == "__main__":
    main()
