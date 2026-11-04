"""Check the numbers of the TCC against the values recomputed from results/.

For every number the README cites, ``EXPECTED`` holds the value printed in the TCC. The script

  1. recomputes the value from the per-sequence files (``results/summary.csv``, built by
     ``make_tables.py`` and ``hypothesis_tests.py``) and requires agreement within half a unit of the
     last printed digit (0.05 for one decimal place);
  2. with ``--pdf``, also requires the number, written with a decimal comma, to appear in the text
     of the TCC PDF (needs PyMuPDF), so a stale expectation cannot pass silently.

Exit status is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# (table, row, column, value printed in the TCC, decimals, optional flags)
# Percent columns (MOTA, IDF1, IDs reductions) are in percent; SA/SR/IoU/AUC in [0, 1].
E = []


def add(table, row, column, value, decimals, **flags):
    E.append((table, row, column, value, decimals, flags))


# Table 4
for model, vals in {"YOLO26n": (0.956, 0.899, 0.944, 0.616), "YOLO26m": (0.969, 0.923, 0.961, 0.662)}.items():
    for col, v in zip(("precision", "recall", "map50", "map50_95"), vals):
        add("T4", model, col, v, 3)

# Table 5 (and the SA/SR columns of Table 13)
T5 = {"BoT-SORT": (0.828, 0.832, 0.659, 0.662, 42.7), "DeepSORT": (0.830, 0.834, 0.662, 0.664, 4.5),
      "ByteTrack": (0.833, 0.836, 0.660, 0.663, 58.7), "OSTrack": (0.803, 0.831, 0.668, 0.672, 13.7),
      "ODTrack": (0.819, 0.860, 0.707, 0.709, 7.3), "CSRT": (0.786, 0.811, 0.646, 0.648, 66.4),
      "KCF": (0.739, 0.741, 0.582, 0.586, 227.2), "MOSSE": (0.819, 0.823, 0.648, 0.651, 406.4),
      "MedianFlow": (0.773, 0.787, 0.637, 0.640, 258.8)}
for method, vals in T5.items():
    for col, v, d in zip(("sa05", "sr05", "mean_iou", "auc", "fps"), vals, (3, 3, 3, 3, 1)):
        if method in ("KCF", "MOSSE", "MedianFlow") and col in ("mean_iou", "auc"):
            continue                                    # not printed in the TCC
        add("T5", method, col, v, d)

# Table 6
for name, vals in {"BoT-SORT": (76.0, 71.2, 2355, 65554), "FastTracker": (72.7, 50.5, 3948, 84792),
                   "ByteTrack": (71.7, 49.4, 4582, 84768), "OC-SORT": (71.5, 51.8, 4454, 85034),
                   "Deep OC-SORT": (71.2, 52.5, 4165, 89823), "TrackTrack": (63.1, 68.9, 836, 168281)}.items():
    for col, v, d in zip(("MOTA", "IDF1", "IDs", "FN"), vals, (1, 1, 0, 0)):
        add("T6", name, col, v, d)

# Tables 7, 10 and 12
for col, v, d in zip(("MOTA", "IDF1", "IDs"), (79.6, 80.3, 712), (1, 1, 0)):
    add("T7", "512", col, v, d)
for row, vals in {"baseline": (76.0, 71.2, 2355), "gm_nsa": (76.9, 74.8, 1611), "hem": (76.7, 71.2, 2374),
                  "hem_sahi512": (79.6, 80.3, 712)}.items():
    for col, v, d in zip(("MOTA", "IDF1", "IDs"), vals, (1, 1, 0)):
        add("T10", row, col, v, d)
for row, vals in {"baseline": (76.0, 71.2, 2355, 65554, 54.8), "hit_tracker": (79.9, 82.7, 494, 51886, 23.0),
                  "hybrid_csrt_n15": (60.9, 59.4, 1453, 135181, 2.6)}.items():
    for col, v, d in zip(("MOTA", "IDF1", "IDs", "FN", "fps"), vals, (1, 1, 0, 0, 1)):
        add("T12", row, col, v, d)
add("T6", "BoT-SORT", "FN", 65554, 0)

# Tables 13 and 14 (H30 error, persistent filter, mean and standard deviation over the 20 videos)
H30 = {"BoT-SORT": (226.2, 303.6, 1142), "DeepSORT": (261.9, 446.0, 1980), "ByteTrack": (270.1, 462.9, 2059),
       "OSTrack": (204.4, 206.3, 753), "ODTrack": (167.4, 160.8, 655), "CSRT": (139.8, 97.6, 306),
       "KCF": (179.5, 189.4, None), "MOSSE": (197.9, 248.1, None), "MedianFlow": (162.2, 126.4, None)}
for method, (mean, std, mx) in H30.items():
    add("T14", method, "H30_mean_persistent", mean, 1)
    add("T14", method, "H30_std_persistent", std, 1)
    if mx is not None:
        add("T14", method, "H30_max_persistent", mx, 0)
add("T14", "CSRT", "H30_mean_damped", 111.1, 1)
add("T14", "CSRT", "H30_std_damped", 76.0, 1)
add("T14", "CSRT", "H30_damping_reduction_pct", 20.5, 1)
for method, red in {"BoT-SORT": 30.0, "ByteTrack": 35.5, "OSTrack": 13.2, "ODTrack": 15.9}.items():
    add("T14", method, "H30_damping_reduction_pct", red, 1)
for method, red in {"DeepSORT": 35, "KCF": 24, "MOSSE": 30, "MedianFlow": 24}.items():
    add("T14", method, "H30_damping_reduction_pct", red, 0, in_text=False)       # printed in Figure 7 only

# Table 15
T15 = {"baseline_persistent": (0.192, 0.406, 0.662, 1.613, 21.8), "baseline_damped": (0.210, 0.456, 0.741, 1.698, 24.1),
       "hit_tracker_persistent": (0.190, 0.401, 0.656, 1.601, 21.4),
       "hit_tracker_damped": (0.211, 0.460, 0.749, 1.710, 24.2),
       "hybrid_csrt_n15_persistent": (0.224, 0.443, 0.698, 1.630, 21.3),
       "hybrid_csrt_n15_damped": (0.232, 0.474, 0.755, 1.705, 23.9)}
for row, vals in T15.items():
    for col, v, d in zip(("H5_norm", "H10_norm", "H15_norm", "H30_norm", "H30_px"), vals, (3, 3, 3, 3, 1)):
        add("T15", row, col, v, d)
add("SUPPORT", "hit_vs_baseline_persistent", "norm_a", 1.566, 3)
add("SUPPORT", "hit_vs_baseline_persistent", "norm_b", 1.611, 3)
add("SUPPORT", "hit_vs_baseline_persistent", "diff", -0.045, 3)
add("SUPPORT", "hybrid_vs_baseline_persistent", "norm_a", 1.624, 3)
add("SUPPORT", "hybrid_vs_baseline_persistent", "norm_b", 1.394, 3)
add("TRACK3", "targets", "mean_diagonal_px", 15.9, 1)

# Cost per target (Section 4.2.7)
add("COST", "hybrid_csrt_n15", "slope_ms_per_target", 32.9, 1)
add("COST", "hybrid_csrt_n15", "ci95_low", 14.7, 1)
add("COST", "hybrid_csrt_n15", "ci95_high", 53.5, 1)
add("COST", "baseline", "slope_ms_per_target", 0.13, 2)
add("COST", "hit_tracker", "slope_ms_per_target", 0.06, 2)
add("COST", "hybrid_csrt_n15 - baseline", "slope_ms_per_target", 32.7, 1)

# Root-cause analysis (Section 4.2.2)
ROOT_CASES = [("mean_density~IDF1", "pearson_r", 0.18), ("mean_density~MOTA", "pearson_r", 0.20),
              ("mean_motion~IDF1", "pearson_r", 0.37), ("mean_edge_density~IDF1", "pearson_r", -0.60),
              ("mean_edge_density~IDF1", "pearson_ci_low", -0.78), ("mean_edge_density~IDF1", "pearson_ci_high", -0.31),
              ("mean_edge_density~IDF1", "spearman_rho", -0.55), ("mean_local_contrast~IDF1", "pearson_r", -0.56),
              ("mean_local_contrast~IDF1", "spearman_rho", -0.21), ("mean_local_contrast~IDF1", "spearman_p", 0.26),
              ("mean_target_bg_contrast~MOTA", "pearson_r", 0.71), ("mean_target_bg_contrast~IDF1", "pearson_r", 0.60),
              ("mean_edge_density~ids_per_thousand", "pearson_r", 0.52),
              ("mean_local_contrast~ids_per_thousand", "pearson_r", 0.65)]
for row, col, v in ROOT_CASES:
    add("ROOT", row, col, v, 2)
add("ROOT", "edge_density~local_contrast", "pearson_r", 0.91, 2)

# Hypothesis tests (Holm-adjusted p-values and effects)
T = "TESTS"
add(T, "F1: H1: damped - persistent, H30, baseline (YOLO26n + BoT-SORT)", "estimate", 0.085, 3)
add(T, "F1: H1: damped - persistent, H30, baseline (YOLO26n + BoT-SORT)", "p_holm", 0.26, 2)
add(T, "F1: H1: damped - persistent, H30, HIT-Tracker", "estimate", 0.110, 3)
add(T, "F1: H1: damped - persistent, H30, HIT-Tracker", "p_holm", 0.24, 2)
add(T, "F1: H2: Spearman IDF1 x persistent normalised H30, baseline (YOLO26n + BoT-SORT)", "estimate", -0.29, 2)
add(T, "F1: H2: Spearman IDF1 x persistent normalised H30, baseline (YOLO26n + BoT-SORT)", "p_holm", 0.26, 2)
add(T, "F1: H2: Spearman IDF1 x persistent normalised H30, HIT-Tracker", "estimate", -0.28, 2)
add(T, "F1: H3: HIT-Tracker - baseline, H30 (common support, persistent)", "estimate", -0.045, 3)
add(T, "F1: H3: HIT-Tracker - baseline, H30 (common support, persistent)", "ci95_low", -0.082, 3)
add(T, "F1: H3: HIT-Tracker - baseline, H30 (common support, persistent)", "ci95_high", -0.025, 3)
add(T, "F1: H4: time-vs-targets slope, hybrid - baseline", "p_holm", 0.001, 3)
add(T, "F3: CSRT - MedianFlow, H30", "p_holm", 0.06, 2)
add(T, "F3: CSRT - KCF, H30", "p_holm", 0.70, 2)
add(T, "F3: CSRT - MOSSE, H30", "p_holm", 0.81, 2)
add(T, "F2: Hybrid CSRT - ByteTrack, H30", "p_holm", 0.10, 2)
add(T, "F2: Hybrid CSRT - DeepSORT, H30", "p_holm", 0.10, 2)
add(T, "F3: CSRT - MedianFlow, H30", "median_diff", -2.6, 1, in_text=False)
add(T, "F3: CSRT - MOSSE, H30", "median_diff", 0.8, 1, in_text=False)

# Quantities derived in the text: (description, expression over summary values, printed value, decimals).
# Differences between percentages are taken between the values as printed (one decimal), as in the TCC;
# the unrounded MOTA differences are 3.96 and 15.04 percentage points.
def r1(g, *key):
    return round(g(*key), 1)


DERIVED = [
    ("HIT-Tracker MOTA gain over baseline (pp)", lambda g: r1(g, "T12", "hit_tracker", "MOTA") - r1(g, "T12", "baseline", "MOTA"), 3.9, 1),
    ("HIT-Tracker IDF1 gain over baseline (pp)", lambda g: r1(g, "T12", "hit_tracker", "IDF1") - r1(g, "T12", "baseline", "IDF1"), 11.5, 1),
    ("HIT-Tracker identity-switch reduction (%)",
     lambda g: 100 * (1 - g("T12", "hit_tracker", "IDs") / g("T12", "baseline", "IDs")), 79.0, 1),
    ("GM + NSA-Kalman IDF1 gain (pp)", lambda g: r1(g, "T10", "gm_nsa", "IDF1") - r1(g, "T10", "baseline", "IDF1"), 3.6, 1),
    ("GM + NSA-Kalman identity-switch reduction (%)",
     lambda g: 100 * (1 - g("T10", "gm_nsa", "IDs") / g("T10", "baseline", "IDs")), 31.6, 1),
    ("Hybrid MOTA loss on Track 3 (pp)", lambda g: r1(g, "T12", "baseline", "MOTA") - r1(g, "T12", "hybrid_csrt_n15", "MOTA"), 15.1, 1),
    ("Hybrid H30 error below BoT-SORT (%)",
     lambda g: 100 * (1 - g("T14", "CSRT", "H30_mean_persistent") / g("T14", "BoT-SORT", "H30_mean_persistent")), 38.2, 1),
    ("Hybrid H30 error below ODTrack (%)",
     lambda g: 100 * (1 - g("T14", "CSRT", "H30_mean_persistent") / g("T14", "ODTrack", "H30_mean_persistent")), 16.5, 1),
    ("HIT-Tracker common-support H30 error reduction (%)",
     lambda g: 100 * (1 - g("SUPPORT", "hit_vs_baseline_persistent", "norm_a") / g("SUPPORT", "hit_vs_baseline_persistent", "norm_b")), 2.8, 1),
    ("Hybrid vs baseline common-support error increase (%)",
     lambda g: 100 * (g("SUPPORT", "hybrid_vs_baseline_persistent", "norm_a") / g("SUPPORT", "hybrid_vs_baseline_persistent", "norm_b") - 1), 16.5, 1),
]


def pt(value: float, decimals: int) -> str:
    """Format as printed in the TCC: decimal comma, thousands dot, true minus sign."""
    text = f"{abs(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("−" + text) if value < 0 else text


def pdf_text(path: Path) -> str:
    import pymupdf

    return "\n".join(page.get_text() for page in pymupdf.open(path)).replace("−", "−")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--pdf", type=Path, default=None, help="TCC PDF; also checks that each number appears in it")
    args = parser.parse_args()

    summary = pd.read_csv(args.results / "summary.csv")
    index = {(r.table, r.row, r.column): r.value for r in summary.itertuples()}

    def get(table, row, column):
        return index[(table, row, column)]

    text = pdf_text(args.pdf) if args.pdf else None
    failures = 0
    checked = 0
    for table, row, column, expected, decimals, flags in E:
        key = (table, row, column)
        if key not in index:
            print(f"MISSING  {table} / {row} / {column}")
            failures += 1
            continue
        got = index[key]
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        ok = abs(got - expected) <= tol
        in_pdf = True
        if text is not None and flags.get("in_text", True):
            candidates = {pt(expected, decimals)}
            if expected < 0:
                candidates.add(pt(expected, decimals).replace("−", "-"))
            if decimals == 0 and abs(expected) >= 1000:
                candidates.add(f"{int(abs(expected)):,}".replace(",", "."))
            in_pdf = any(c in text for c in candidates)
        checked += 1
        if not (ok and in_pdf):
            failures += 1
            why = "value" if not ok else "not found in PDF"
            print(f"FAIL ({why})  {table} / {row} / {column}: results={got:.6g} tcc={expected} (±{tol:.4g})")
    for description, fn, expected, decimals in DERIVED:
        got = fn(get)
        checked += 1
        if abs(got - expected) > 0.5 * 10 ** (-decimals) + 1e-9:
            failures += 1
            print(f"FAIL (value)  {description}: results={got:.6g} tcc={expected}")
    print(f"{checked} numbers checked, {failures} failure(s)" + (" (PDF text checked)" if text else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
