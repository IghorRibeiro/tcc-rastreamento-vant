"""Hypothesis tests H1-H4 and complementary single-target families, from results/ only.

Output: results/hypothesis_tests.csv, one row per test. Holm adjustment is applied within each
family; Bonferroni-level intervals use 1 - alpha/m. The sample unit is the sequence.

  F1 (Track 3, normalised error, CLEAR correspondence)
     H1  damped - persistent at H30, baseline and HIT-Tracker (two tests)
     H2  Spearman between IDF1 and persistent normalised H30 error, baseline and HIT-Tracker
     H3  HIT-Tracker - baseline at H30 on the common support (persistent filter)
     H4  slope of time per frame vs number of targets, hybrid - baseline (cluster bootstrap)
  F2 (DUT Anti-UAV, pixels, persistent filter)
     hybrid (CSRT) - each reference method at H30 on the common support
  F3 (DUT Anti-UAV, pixels, persistent filter)
     CSRT - KCF, MOSSE and MedianFlow at H30 (the per-video errors of the local-tracker table)
  F4 (DUT Anti-UAV, pixels)
     damped - persistent at H30 for each of the nine methods

Paired differences: Wilcoxon two-sided + rank-biserial; BCa interval of the mean difference.
Correlations: Spearman + paired BCa interval. Slopes: cluster BCa bootstrap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from . import inference
except ImportError:                      # executed as a script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import inference

NAN = float("nan")
ROOT = Path(__file__).resolve().parents[1]

LABELS = {"baseline": "baseline (YOLO26n + BoT-SORT)", "hit_tracker": "HIT-Tracker"}
REFERENCES = ["BoT-SORT", "ByteTrack", "DeepSORT", "OSTrack", "ODTrack"]
LOCAL_TRACKERS = ["KCF", "MOSSE", "MedianFlow"]
NINE_METHODS = ["BoT-SORT", "DeepSORT", "ByteTrack", "OSTrack", "ODTrack", "CSRT", "KCF", "MOSSE", "MedianFlow"]


class Family:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.items = []        # (row, interval_fn(level) -> (low, high))

    def add(self, row, interval_fn):
        self.items.append(({"family": self.name, "family_description": self.description, **row}, interval_fn))

    def finalize(self, alpha):
        rows = [r for r, _ in self.items]
        m = len(rows)
        adjusted = inference.holm([r.get("p_raw", NAN) for r in rows]) if m else []
        out = []
        for (row, fn), p_adj in zip(self.items, adjusted):
            row["family_size"] = m
            row["p_holm"] = float(p_adj)
            level = 1.0 - alpha / m if m else NAN
            row["bonferroni_level"] = level
            lo, hi = fn(level) if fn else (NAN, NAN)
            row["bonferroni_low"], row["bonferroni_high"] = lo, hi
            row["significant_holm_5pct"] = bool(np.isfinite(p_adj) and p_adj < alpha)
            out.append(row)
        return out


def paired(diffs, seed, n_boot, **info):
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    w = inference.wilcoxon_rb(d)
    lo, hi, note = inference.bca_mean(d, 0.95, seed, n_boot)
    row = {**info, "n": int(d.size), "estimate": float(d.mean()) if d.size else NAN,
           "median_diff": float(np.median(d)) if d.size else NAN, "ci95_low": lo, "ci95_high": hi,
           "statistic": w["W"], "statistic_name": "W (Wilcoxon)", "p_raw": w["p"],
           "rank_biserial": w["rank_biserial"],
           "method": "paired two-sided Wilcoxon; BCa interval of the mean difference",
           "note": "; ".join(x for x in (w["note"], note) if x)}
    return row, (lambda level: inference.bca_mean(d, level, seed, n_boot)[:2])


def spearman(x, y, seed, n_boot, **info):
    r = inference.correlation(x, y, "spearman", 0.95, seed, n_boot)
    row = {**info, "n": r["n"], "estimate": r["r"], "median_diff": NAN, "ci95_low": r["ci_low"],
           "ci95_high": r["ci_high"], "statistic": r["r"], "statistic_name": "rho (Spearman)", "p_raw": r["p"],
           "rank_biserial": NAN, "method": "Spearman; paired BCa interval", "note": r["note"]}

    def fn(level):
        rr = inference.correlation(x, y, "spearman", level, seed, n_boot)
        return rr["ci_low"], rr["ci_high"]

    return row, fn


def difference_by_key(df, key, value, side_column, a, b):
    """Per-key difference a - b of ``value`` between two labelled sides."""
    pa = df[df[side_column] == a].set_index(key)[value]
    pb = df[df[side_column] == b].set_index(key)[value]
    joined = pa.to_frame("a").join(pb.to_frame("b"), how="inner").dropna()
    return joined["a"] - joined["b"]


def run(results_dir: Path, seed: int, n_boot: int, alpha: float) -> pd.DataFrame:
    read = lambda name: pd.read_csv(results_dir / name)       # noqa: E731
    rows = []

    # ---------------- F1: Track 3 (H1-H4) ----------------
    f1 = Family("F1", "Track 3: damping (H1), IDF1 vs prediction (H2), HIT-Tracker vs baseline (H3), cost slope (H4)")
    pred = read("mot_prediction_per_seq.csv")
    pred = pred[pred["mode"] == "clear"]
    h30 = pred[pred["H"] == 30]
    for v in ("baseline", "hit_tracker"):
        sel = h30[h30["variant"] == v].copy()
        sel["side"] = sel["damping"].map({1.0: "persistent", 0.95: "damped"})
        d = difference_by_key(sel, "seq", "mean_norm", "side", "damped", "persistent")
        row, fn = paired(d, seed, n_boot, test=f"H1: damped - persistent, H30, {LABELS[v]}",
                         comparison="gamma=0.95 - gamma=1.0",
                         variable="normalised H30 error (per-sequence mean)", unit="fraction of target diagonal")
        f1.add(row, fn)
    tracking = read("mot_tracking_per_seq.csv")
    for v in ("baseline", "hit_tracker"):
        e = h30[(h30["variant"] == v) & (h30["damping"] == 1.0)].set_index("seq")["mean_norm"]
        i = tracking[tracking["variant"] == v].set_index("seq")["idf1"]
        j = e.to_frame("e").join(i.to_frame("i"), how="inner").dropna()
        row, fn = spearman(j["i"].to_numpy(), j["e"].to_numpy(), seed, n_boot,
                           test=f"H2: Spearman IDF1 x persistent normalised H30, {LABELS[v]}",
                           comparison="correlation", variable="sequence IDF1 x normalised H30 error",
                           unit="rho")
        f1.add(row, fn)
    support = read("mot_common_support_per_seq.csv")
    s = support[(support["variant_a"] == "hit_tracker") & (support["variant_b"] == "baseline")
                & (support["H"] == 30) & (support["damping"] == 1.0)]
    row, fn = paired((s["mean_norm_a"] - s["mean_norm_b"]).to_numpy(), seed, n_boot,
                     test="H3: HIT-Tracker - baseline, H30 (common support, persistent)",
                     comparison="HIT-Tracker - baseline",
                     variable="normalised H30 error on the common support (per-sequence mean)",
                     unit="fraction of target diagonal")
    f1.add(row, fn)

    sums = read("cost_sufficient_stats.csv")
    cols = ["n", "sx", "sy", "sxx", "sxy", "syy"]
    per_variant = {v: sums[sums["variant"] == v].set_index("seq") for v in ("hybrid_csrt_n15", "baseline")}
    seqs = per_variant["hybrid_csrt_n15"].index.intersection(per_variant["baseline"].index)
    matrices = {v: per_variant[v].loc[list(seqs), cols].to_numpy(float) for v in per_variant}
    key = "hybrid_csrt_n15 - baseline"
    pair = [("hybrid_csrt_n15", "baseline")]
    boot = inference.cluster_slopes(matrices, conf_levels=(0.95,), seed=seed, n_resamples=n_boot, diffs=pair)[key]
    row = {"test": "H4: time-vs-targets slope, hybrid - baseline", "variable": "ms per target per frame",
           "unit": "ms/target", "comparison": "hybrid - baseline", "n": boot["K"],
           "estimate": boot["estimate"], "median_diff": NAN, "ci95_low": boot["ci"][0.95][0],
           "ci95_high": boot["ci"][0.95][1], "statistic": NAN, "statistic_name": "-", "p_raw": boot["p"],
           "rank_biserial": NAN, "method": "paired cluster bootstrap over sequences, BCa; two-sided bootstrap p",
           "note": ""}

    def slope_interval(level):
        b2 = inference.cluster_slopes(matrices, conf_levels=(level,), seed=seed, n_resamples=n_boot, diffs=pair)[key]
        return b2["ci"][level]

    f1.add(row, slope_interval)
    rows += f1.finalize(alpha)

    # ---------------- F2: DUT, hybrid vs references on the common support ----------------
    f2 = Family("F2", "DUT Anti-UAV: hybrid (CSRT) vs each reference method, H30 on the common support")
    shared = read("sot_common_support_per_video.csv")
    for ref in REFERENCES:
        s = shared[(shared["reference"] == ref) & (shared["H"] == 30) & (shared["damping"] == 1.0)]
        if s.empty:
            continue
        row, fn = paired((s["mean_px_hybrid"] - s["mean_px_reference"]).to_numpy(), seed, n_boot,
                         test=f"Hybrid CSRT - {ref}, H30", comparison=f"CSRT - {ref}",
                         variable="H30 error on the common support (per-video mean)", unit="px")
        f2.add(row, fn)
    rows += f2.finalize(alpha)

    # ---------------- F3: DUT, local trackers ----------------
    f3 = Family("F3", "DUT Anti-UAV: CSRT vs KCF, MOSSE and MedianFlow, H30 (persistent)")
    sot = read("sot_prediction_per_video.csv")
    h30p = sot[(sot["H"] == 30) & (sot["damping"] == 1.0)]
    csrt = h30p[h30p["method"] == "CSRT"].set_index("video")["mean_px"]
    for other in LOCAL_TRACKERS:
        o = h30p[h30p["method"] == other].set_index("video")["mean_px"]
        j = csrt.to_frame("a").join(o.to_frame("b"), how="inner").dropna()
        row, fn = paired((j["a"] - j["b"]).to_numpy(), seed, n_boot, test=f"CSRT - {other}, H30",
                         comparison=f"CSRT - {other}", variable="H30 error (per-video mean)", unit="px")
        f3.add(row, fn)
    rows += f3.finalize(alpha)

    # ---------------- F4: DUT, damping ----------------
    f4 = Family("F4", "DUT Anti-UAV: damped - persistent at H30 for the nine methods")
    for method in NINE_METHODS:
        s = sot[(sot["method"] == method) & (sot["H"] == 30)].copy()
        if s.empty:
            continue
        s["side"] = s["damping"].map({1.0: "persistent", 0.95: "damped"})
        d = difference_by_key(s, "video", "mean_px", "side", "damped", "persistent")
        row, fn = paired(d.to_numpy(), seed, n_boot, test=f"Damped - persistent, H30, {method}",
                         comparison="gamma=0.95 - gamma=1.0", variable="H30 error (per-video mean)", unit="px")
        f4.add(row, fn)
    rows += f4.finalize(alpha)

    columns = ["family", "test", "comparison", "variable", "unit", "n", "estimate", "median_diff", "ci95_low",
               "ci95_high", "bonferroni_level", "bonferroni_low", "bonferroni_high", "statistic_name", "statistic",
               "p_raw", "p_holm", "family_size", "significant_holm_5pct", "rank_biserial", "method", "note",
               "family_description"]
    df = pd.DataFrame(rows)
    return df[[c for c in columns if c in df.columns]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "statistics.yaml")
    parser.add_argument("--out", type=Path, default=None, help="default: <results>/hypothesis_tests.csv")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    df = run(args.results, int(cfg["seed"]), int(cfg["bootstrap_resamples"]), float(cfg["alpha"]))
    out = args.out or (args.results / "hypothesis_tests.csv")
    df.to_csv(out, index=False, float_format="%.10g")
    print(f"{len(df)} tests written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
