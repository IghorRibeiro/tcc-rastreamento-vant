#!/usr/bin/env python3
"""Paired statistical comparisons across the same DUT sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


METRICS = [
    "SR@0.5",
    "mean_IoU_present",
    "success_AUC",
    "time_ms_mean",
    "pred_error_H5_mean",
    "pred_error_H10_mean",
    "pred_error_H15_mean",
    "pred_error_H30_mean",
]


def bootstrap_ci(values: np.ndarray, seed: int = 2026, n_boot: int = 10000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-video", type=Path, default=Path("scripts/reviewer_validation/results/all_methods_per_video.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/reviewer_validation/results"))
    args = parser.parse_args()
    df = pd.read_csv(args.per_video)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(df["method"].drop_duplicates())
    rows = []
    for i, method_a in enumerate(methods):
        for method_b in methods[i + 1 :]:
            left = df[df.method == method_a].set_index("video")
            right = df[df.method == method_b].set_index("video")
            common = left.index.intersection(right.index)
            for metric in METRICS:
                if metric not in left or metric not in right:
                    continue
                a = pd.to_numeric(left.loc[common, metric], errors="coerce").to_numpy(dtype=float)
                b = pd.to_numeric(right.loc[common, metric], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(a) & np.isfinite(b)
                a, b = a[valid], b[valid]
                if len(a) < 2:
                    continue
                diff = a - b
                diff_ci_low, diff_ci_high = bootstrap_ci(diff)
                sd_diff = np.std(diff, ddof=1)
                try:
                    t_stat, p_t = ttest_rel(a, b, nan_policy="omit")
                except Exception:
                    t_stat, p_t = np.nan, np.nan
                try:
                    w_stat, p_w = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
                except ValueError:
                    w_stat, p_w = np.nan, np.nan
                rows.append({
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "n_pairs": len(diff),
                    "mean_a": np.mean(a),
                    "mean_b": np.mean(b),
                    "mean_difference_a_minus_b": np.mean(diff),
                    "difference_ci95_low": diff_ci_low,
                    "difference_ci95_high": diff_ci_high,
                    "paired_cohen_d": np.mean(diff) / sd_diff if sd_diff > 0 else np.nan,
                    "paired_t_stat": t_stat,
                    "paired_t_pvalue": p_t,
                    "wilcoxon_stat": w_stat,
                    "wilcoxon_pvalue": p_w,
                })
    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "paired_significance_tests.csv", index=False, float_format="%.8f")
    print(out.to_string(index=False))
    print(f"Saved {len(out)} tests to {args.output_dir / 'paired_significance_tests.csv'}")


if __name__ == "__main__":
    main()
