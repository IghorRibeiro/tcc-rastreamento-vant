"""Recompute the quantities behind the TCC tables and text from the per-sequence files in results/.

Writes results/summary.csv in long format (table, row, column, value). ``verify_tcc.py`` compares
these values with the numbers printed in the TCC; the README tables are copied from this file.

Unit conventions: MOTA/IDF1 in percent; SA, SR, IoU and AUC in [0, 1]; errors in pixels unless the
column says "norm" (fraction of the target diagonal); time in milliseconds.
"""

from __future__ import annotations

import argparse
import math
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

ROOT = Path(__file__).resolve().parents[1]

SOT_METHODS = ["BoT-SORT", "DeepSORT", "ByteTrack", "OSTrack", "ODTrack", "CSRT", "KCF", "MOSSE", "MedianFlow"]
MOT_VARIANTS = ["baseline", "hit_tracker", "hybrid_csrt_n15"]
LEADERBOARD = {"lb_fasttrack": "FastTracker", "lb_bytetrack": "ByteTrack", "lb_ocsort": "OC-SORT",
               "lb_deepocsort": "Deep OC-SORT", "lb_tracktrack": "TrackTrack"}


class Summary:
    def __init__(self):
        self.rows = []

    def add(self, table, row, column, value):
        self.rows.append({"table": table, "row": row, "column": column, "value": float(value)})

    def frame(self):
        return pd.DataFrame(self.rows)


def pooled(df: pd.DataFrame) -> dict:
    """Pooled MOT metrics (percent) over the sequences of ``df``."""
    gt, pred = df["gt_objects"].sum(), df["predictions"].sum()
    fn, fp, ids = df["fn"].sum(), df["fp"].sum(), df["id_switches"].sum()
    return {"MOTA": 100.0 * (1.0 - (fn + fp + ids) / gt), "IDF1": 100.0 * 2.0 * df["idtp"].sum() / (gt + pred),
            "IDs": ids, "FP": fp, "FN": fn, "sequences": len(df)}


def compute(results: Path, seed: int = 2026, n_boot: int = 10000) -> pd.DataFrame:
    read = lambda name: pd.read_csv(results / name)       # noqa: E731
    out = Summary()

    # ---- Table 4: detectors
    for _, r in read("detectors.csv").iterrows():
        for col in ("precision", "recall", "map50", "map50_95"):
            out.add("T4", r["model"], col, r[col])

    # ---- Tables 5, 13 and 14: single target (DUT Anti-UAV)
    track = read("sot_tracking_per_video.csv")
    fps = read("sot_fps.csv").set_index("method")["fps"]
    pred = read("sot_prediction_per_video.csv")
    for m in SOT_METHODS:
        t = track[track["method"] == m]
        for col in ("sa05", "sr05", "mean_iou", "auc"):
            out.add("T5", m, col, t[col].mean())
        if m in fps.index:
            out.add("T5", m, "fps", fps[m])
        for gamma, tag in ((1.0, "persistent"), (0.95, "damped")):
            p = pred[(pred["method"] == m) & (pred["damping"] == gamma)]
            for h in (5, 10, 15, 30):
                v = p[p["H"] == h]["mean_px"]
                out.add("T14", m, f"H{h}_mean_{tag}", v.mean())
                out.add("T14", m, f"H{h}_std_{tag}", v.std(ddof=1))
                if h == 30:
                    out.add("T14", m, f"H30_max_{tag}", v.max())
        a = pred[(pred["method"] == m) & (pred["damping"] == 1.0) & (pred["H"] == 30)]["mean_px"].mean()
        b = pred[(pred["method"] == m) & (pred["damping"] == 0.95) & (pred["H"] == 30)]["mean_px"].mean()
        out.add("T14", m, "H30_damping_reduction_pct", 100.0 * (a - b) / a)

    # ---- Tables 6, 7 and 10: detector and association interventions (Track 3)
    mot = read("mot_tracking_per_seq.csv")
    runs = read("incremental_runs_per_seq.csv")
    base = pooled(mot[mot["variant"] == "baseline"])
    for col, v in base.items():
        out.add("T6", "BoT-SORT", col, v)
        out.add("T10", "baseline", col, v)
    for config, name in LEADERBOARD.items():
        for col, v in pooled(runs[runs["config"] == config]).items():
            out.add("T6", name, col, v)
    for config, row in (("hem", "hem"), ("hem_sahi512", "hem_sahi512"), ("gm_nsa_g055", "gm_nsa")):
        for col, v in pooled(runs[runs["config"] == config]).items():
            out.add("T10", row, col, v)
    for col, v in pooled(runs[runs["config"] == "hem_sahi512"]).items():
        out.add("T7", "512", col, v)
    for config in sorted(runs["config"].unique()):
        if config.startswith("gm_nsa_g"):
            p = pooled(runs[runs["config"] == config])
            out.add("F5", config, "gamma", float(config.replace("gm_nsa_g", "")) / 100.0)
            for col in ("MOTA", "IDF1", "IDs"):
                out.add("F5", config, col, p[col])

    # ---- Table 12 and cost: baseline, HIT-Tracker and hybrid on Track 3
    cost = read("cost_per_seq.csv")
    sums = read("cost_sufficient_stats.csv")
    cols = ["n", "sx", "sy", "sxx", "sxy", "syy"]
    matrices = {}
    for v in MOT_VARIANTS:
        p = pooled(mot[mot["variant"] == v])
        for col, val in p.items():
            out.add("T12", v, col, val)
        c = cost[cost["variant"] == v]
        mean_ms = float((c["frames"] * c["mean_time_ms"]).sum() / c["frames"].sum())
        out.add("T12", v, "mean_time_ms", mean_ms)
        out.add("T12", v, "fps", 1000.0 / mean_ms)
        s = sums[sums["variant"] == v].set_index("seq")
        matrices[v] = s.loc[sorted(s.index), cols].to_numpy(float)
    boot = inference.cluster_slopes(matrices, conf_levels=(0.95,), seed=seed, n_resamples=n_boot,
                                    diffs=[("hybrid_csrt_n15", "baseline")])
    for name, res in boot.items():
        out.add("COST", name, "slope_ms_per_target", res["estimate"])
        out.add("COST", name, "ci95_low", res["ci"][0.95][0])
        out.add("COST", name, "ci95_high", res["ci"][0.95][1])

    # ---- Table 15: identity-aware prediction on Track 3 (CLEAR correspondence)
    mp = read("mot_prediction_per_seq.csv")
    mp = mp[mp["mode"] == "clear"]
    for v in MOT_VARIANTS:
        for gamma, tag in ((1.0, "persistent"), (0.95, "damped")):
            for h in (5, 10, 15, 30):
                sel = mp[(mp["variant"] == v) & (mp["damping"] == gamma) & (mp["H"] == h)]
                out.add("T15", f"{v}_{tag}", f"H{h}_norm", sel["mean_norm"].mean())
                if h == 30:
                    out.add("T15", f"{v}_{tag}", "H30_px", sel["mean_px"].mean())
    cs = read("mot_common_support_per_seq.csv")
    for (a, b), label in ((("hit_tracker", "baseline"), "hit_vs_baseline"),
                          (("hybrid_csrt_n15", "baseline"), "hybrid_vs_baseline")):
        for gamma, tag in ((1.0, "persistent"), (0.95, "damped")):
            sel = cs[(cs["variant_a"] == a) & (cs["variant_b"] == b) & (cs["damping"] == gamma) & (cs["H"] == 30)]
            out.add("SUPPORT", f"{label}_{tag}", "norm_a", sel["mean_norm_a"].mean())
            out.add("SUPPORT", f"{label}_{tag}", "norm_b", sel["mean_norm_b"].mean())
            out.add("SUPPORT", f"{label}_{tag}", "diff", (sel["mean_norm_a"] - sel["mean_norm_b"]).mean())
    size = read("mot_target_size_per_seq.csv")
    out.add("TRACK3", "targets", "mean_diagonal_px", (size["gt_boxes"] * size["mean_diagonal_px"]).sum()
            / size["gt_boxes"].sum())

    # ---- Section 4.2.2: root-cause analysis (baseline, 30 sequences)
    rc = read("root_cause_per_seq.csv")
    for factor in ("mean_density", "mean_motion", "mean_edge_density", "mean_local_contrast",
                   "mean_target_bg_contrast"):
        for outcome in ("IDF1", "MOTA", "ids_per_thousand"):
            pe = inference.correlation(rc[factor], rc[outcome], "pearson", 0.95, seed, n_boot)
            sp = inference.correlation(rc[factor], rc[outcome], "spearman", 0.95, seed, n_boot)
            out.add("ROOT", f"{factor}~{outcome}", "pearson_r", pe["r"])
            out.add("ROOT", f"{factor}~{outcome}", "pearson_ci_low", pe["ci_low"])
            out.add("ROOT", f"{factor}~{outcome}", "pearson_ci_high", pe["ci_high"])
            out.add("ROOT", f"{factor}~{outcome}", "spearman_rho", sp["r"])
            out.add("ROOT", f"{factor}~{outcome}", "spearman_p", sp["p"])
    co = inference.correlation(rc["mean_edge_density"], rc["mean_local_contrast"], "pearson", 0.95, seed, n_boot)
    out.add("ROOT", "edge_density~local_contrast", "pearson_r", co["r"])

    # ---- Hypotheses (values come from hypothesis_tests.csv when it exists)
    tests = results / "hypothesis_tests.csv"
    if tests.exists():
        for _, r in pd.read_csv(tests).iterrows():
            key = f"{r['family']}: {r['test']}"
            for col in ("estimate", "median_diff", "ci95_low", "ci95_high", "p_raw", "p_holm"):
                if np.isfinite(r[col]):
                    out.add("TESTS", key, col, r[col])
    return out.frame()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "statistics.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    df = compute(args.results, int(cfg["seed"]), int(cfg["bootstrap_resamples"]))
    path = args.results / "summary.csv"
    df.to_csv(path, index=False, float_format="%.10g")
    print(f"{len(df)} values written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
