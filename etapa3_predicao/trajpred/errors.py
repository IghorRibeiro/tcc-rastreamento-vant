"""Identity-aware trajectory-prediction error for multiple targets (TCC, Section 3.4.4.3).

For a track k at frame t the prediction p_k(t+H | t) is compared with the target g = pi_t(k)
that the track was following at t (not with a fixed target of the sequence):

    e_H(k, t)  = || p_k(t+H | t) - c_g(t+H) ||
    e~_H(k, t) = e_H(k, t) / d_g(t+H)          (d = diagonal of the target box)

``pi_t`` is the CLEAR MOT correspondence (mode ``clear``) or the global IDF1 mapping (mode
``idf1``). Pairs in which the track is not associated at t, or the target does not exist at
t+H, are discarded and counted. The Kalman filter is the one of ``trajpred.kalman``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .kalman import corrected_states, forward

MODES = ("clear", "idf1")


class GroundTruth:
    """Per-target lookup of centre and diagonal for one sequence."""

    def __init__(self, gt_df: pd.DataFrame):
        self.by_id: dict[int, dict[int, tuple[float, float, float]]] = {}
        for oid, frame, x, y, w, h in zip(gt_df["id"], gt_df["frame"], gt_df["x"], gt_df["y"],
                                          gt_df["w"], gt_df["h"]):
            x, y, w, h = float(x), float(y), float(w), float(h)
            self.by_id.setdefault(int(oid), {})[int(frame)] = (x + w / 2.0, y + h / 2.0, math.hypot(w, h))
        counts = gt_df.groupby("frame")["id"].count()
        self.mean_density = float(counts.mean()) if len(counts) else float("nan")


def track_centers(pred_df: pd.DataFrame) -> dict:
    """{track_id: {frame: (cx, cy)}}; the first record of a frame wins."""
    out: dict[int, dict[int, tuple[float, float]]] = {}
    for tid, frame, x, y, w, h in zip(pred_df["id"], pred_df["frame"], pred_df["x"], pred_df["y"],
                                      pred_df["w"], pred_df["h"]):
        centers = out.setdefault(int(tid), {})
        centers.setdefault(int(frame), (float(x) + float(w) / 2.0, float(y) + float(h) / 2.0))
    return out


def pair_errors(pred_df: pd.DataFrame, gt: GroundTruth, corr: dict, horizons, dampings):
    """Errors of every (track, t, H, damping, mode) pair.

    Returns ``(pairs, counts)``; ``counts`` has, per (mode, damping, H), the number of filter
    steps, steps without association, pairs with an absent target and valid pairs.
    """
    tracks = track_centers(pred_df)
    clear = corr["clear"]
    idmap = corr["idf1"]

    columns = ([], [], [], [], [], [], [], [])
    rec_tid, rec_frame, rec_h, rec_damp, rec_mode, rec_g, rec_e, rec_en = columns
    counts = {(m, d, h): {"steps": 0, "unassociated": 0, "target_absent": 0, "valid_pairs": 0}
              for m in MODES for d in dampings for h in horizons}

    for tid in sorted(tracks):
        centers = tracks[tid]
        g_id = idmap.get(tid)
        for damping in dampings:
            for frame, state in corrected_states(centers, damping):
                g_clear = clear.get((frame, tid))
                for horizon in horizons:
                    targets = []
                    for mode, g in (("clear", g_clear), ("idf1", g_id)):
                        cell = counts[(mode, damping, horizon)]
                        cell["steps"] += 1
                        if g is None:
                            cell["unassociated"] += 1
                            continue
                        target = gt.by_id.get(g, {}).get(frame + horizon)
                        if target is None:
                            cell["target_absent"] += 1
                            continue
                        cell["valid_pairs"] += 1
                        targets.append((mode, g, target))
                    if not targets:
                        continue
                    predicted = forward(state, horizon, damping)
                    for mode, g, (tx, ty, diag) in targets:
                        error = float(np.hypot(predicted[0, 0] - tx, predicted[1, 0] - ty))
                        rec_tid.append(tid)
                        rec_frame.append(frame)
                        rec_h.append(horizon)
                        rec_damp.append(damping)
                        rec_mode.append(mode)
                        rec_g.append(g)
                        rec_e.append(error)
                        rec_en.append(error / diag if diag > 0 else math.nan)

    pairs = pd.DataFrame({
        "tid": np.asarray(rec_tid, dtype=np.int64),
        "frame": np.asarray(rec_frame, dtype=np.int64),
        "H": np.asarray(rec_h, dtype=np.int64),
        "damping": np.asarray(rec_damp, dtype=np.float64),
        "mode": pd.Categorical(rec_mode, categories=list(MODES)),
        "g": np.asarray(rec_g, dtype=np.int64),
        "e_px": np.asarray(rec_e, dtype=np.float64),
        "e_norm": np.asarray(rec_en, dtype=np.float64),
    })
    count_rows = [{"mode": m, "damping": d, "H": h, **v} for (m, d, h), v in counts.items()]
    return pairs, pd.DataFrame(count_rows)


def _std(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if len(series) > 1 else float("nan")


def aggregate_sequence(pairs: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """One row per (mode, damping, H): mean, std, median and p95 in pixels and normalised."""
    rows = []
    for (mode, damping, horizon), cell in counts.groupby(["mode", "damping", "H"], sort=True):
        sel = pairs[(pairs["mode"] == mode) & (pairs["damping"] == damping) & (pairs["H"] == horizon)]
        e = sel["e_px"]
        en = sel["e_norm"].dropna()
        rows.append({
            "mode": mode, "damping": float(damping), "H": int(horizon), "n": int(len(sel)),
            "mean_px": float(e.mean()) if len(e) else math.nan,
            "std_px": _std(e),
            "median_px": float(e.median()) if len(e) else math.nan,
            "p95_px": float(np.percentile(e, 95)) if len(e) else math.nan,
            "mean_norm": float(en.mean()) if len(en) else math.nan,
            "std_norm": _std(en),
            "median_norm": float(en.median()) if len(en) else math.nan,
            "n_tracks": int(sel["tid"].nunique()),
            "steps": int(cell["steps"].sum()),
            "unassociated": int(cell["unassociated"].sum()),
            "target_absent": int(cell["target_absent"].sum()),
            "valid_pairs": int(cell["valid_pairs"].sum()),
        })
    return pd.DataFrame(rows)


def common_support(pairs_a: pd.DataFrame, pairs_b: pd.DataFrame, mode: str = "clear") -> pd.DataFrame:
    """Mean error of A and B over the same (target, frame, H, damping) tuples."""
    keys = ["damping", "H", "g", "frame"]
    cols = keys + ["e_px", "e_norm"]
    merged = pairs_a[pairs_a["mode"] == mode][cols].merge(pairs_b[pairs_b["mode"] == mode][cols],
                                                         on=keys, suffixes=("_a", "_b"))
    rows = []
    for (damping, horizon), grp in merged.groupby(["damping", "H"], sort=True):
        rows.append({
            "damping": float(damping), "H": int(horizon), "n_common": int(len(grp)),
            "mean_px_a": float(grp["e_px_a"].mean()), "mean_px_b": float(grp["e_px_b"].mean()),
            "mean_norm_a": float(grp["e_norm_a"].mean()), "mean_norm_b": float(grp["e_norm_b"].mean()),
        })
    return pd.DataFrame(rows)
