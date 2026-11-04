"""MOT evaluation with the two track-to-target correspondences used by the identity-aware error.

* ``clear``: per-frame correspondence of the CLEAR MOT protocol (MATCH and SWITCH events,
  IoU >= 0.5), the same one that produces MOTA and the identity-switch count;
* ``idf1``: global one-to-one mapping between tracks and targets that maximises the
  identity true positives (IDF1), validated against the IDF1 of py-motmetrics.

The per-frame loop is the one used for the Track 3 evaluation of the LAFusion 2026 paper.
"""

from __future__ import annotations

import numpy as np

# np.asfarray was removed in NumPy 2.0 and is still used by py-motmetrics.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

METRICS = [
    "mota", "motp", "idf1", "idp", "idr", "idtp", "idfp", "idfn",
    "num_switches", "num_false_positives", "num_misses", "num_objects",
    "num_predictions", "num_matches", "num_unique_objects",
]

MOT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "conf", "a", "b", "c"]


def load_mot(path) -> pd.DataFrame:
    """Read a MOTChallenge text file (frame, id, x, y, w, h, ...); an empty file gives an empty frame."""
    try:
        df = pd.read_csv(path, header=None, names=MOT_COLUMNS[:6], usecols=range(6))
    except pd.errors.EmptyDataError:
        df = pd.DataFrame({c: pd.Series(dtype="float64") for c in MOT_COLUMNS[:6]})
    df["frame"] = df["frame"].astype(np.int64)
    df["id"] = df["id"].astype(np.int64)
    for col in ("x", "y", "w", "h"):
        df[col] = df[col].astype(np.float64)
    return df


def build_accumulator(gt_df: pd.DataFrame, pred_df: pd.DataFrame, max_iou: float = 0.5):
    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt_df["frame"]) | set(pred_df["frame"]))
    gt_groups = {k: g for k, g in gt_df.groupby("frame", sort=False)}
    pred_groups = {k: g for k, g in pred_df.groupby("frame", sort=False)}
    gt_empty = gt_df.iloc[0:0]
    pred_empty = pred_df.iloc[0:0]

    for frame in all_frames:
        gt_frame = gt_groups.get(frame, gt_empty)
        pred_frame = pred_groups.get(frame, pred_empty)
        dist_matrix = mm.distances.iou_matrix(gt_frame[["x", "y", "w", "h"]].values,
                                              pred_frame[["x", "y", "w", "h"]].values, max_iou=max_iou)
        acc.update(gt_frame["id"].values, pred_frame["id"].values, dist_matrix)

    return acc, [int(f) for f in all_frames]


def compute_metrics(acc, name: str) -> dict:
    summary = mm.metrics.create().compute(acc, metrics=METRICS, name=name)
    return {key: float(summary.loc[name, key]) for key in METRICS}


def correspondences(acc, all_frames: list[int], gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """``clear``: {(frame, track_id): target_id}; ``idf1``: {track_id: target_id}."""
    events = acc.events.reset_index()
    frame_lut = np.asarray(all_frames, dtype=np.int64)
    events["frame"] = frame_lut[events["FrameId"].to_numpy(dtype=np.int64)]
    types = events["Type"].astype(str)

    matched = events[types.isin(["MATCH", "SWITCH"])]
    clear = {}
    for frame, hid, oid in zip(matched["frame"], matched["HId"], matched["OId"]):
        clear[(int(frame), int(hid))] = int(oid)

    raw = events[(types == "RAW") & events["D"].notna() & events["OId"].notna() & events["HId"].notna()]
    oids = sorted({int(v) for v in gt_df["id"].unique()})
    hids = sorted({int(v) for v in pred_df["id"].unique()})
    o_index = {o: i for i, o in enumerate(oids)}
    h_index = {h: j for j, h in enumerate(hids)}
    idtp_matrix = np.zeros((len(oids), len(hids)), dtype=np.float64)
    if len(raw):
        counts = raw.groupby([raw["OId"].astype(np.int64), raw["HId"].astype(np.int64)]).size()
        for (oid, hid), n in counts.items():
            if int(oid) in o_index and int(hid) in h_index:
                idtp_matrix[o_index[int(oid)], h_index[int(hid)]] = float(n)

    idmap = {}
    idtp_total = 0.0
    if idtp_matrix.size:
        rows, cols = linear_sum_assignment(-idtp_matrix)
        for r, c in zip(rows, cols):
            if idtp_matrix[r, c] > 0:
                idmap[hids[c]] = oids[r]
                idtp_total += idtp_matrix[r, c]

    return {"clear": clear, "idf1": idmap, "idtp_mapping": idtp_total}


def evaluate_sequence(gt_df: pd.DataFrame, pred_df: pd.DataFrame, name: str, max_iou: float = 0.5):
    """MOT metrics, correspondences, and a check that the rebuilt IDF1 equals the library's."""
    acc, frames = build_accumulator(gt_df, pred_df, max_iou=max_iou)
    metrics = compute_metrics(acc, name)
    corr = correspondences(acc, frames, gt_df, pred_df)
    denom = metrics["num_objects"] + metrics["num_predictions"]
    rebuilt = 2.0 * corr["idtp_mapping"] / denom if denom > 0 else float("nan")
    metrics["idf1_rebuilt"] = rebuilt
    metrics["idf1_matches"] = bool(np.isfinite(rebuilt) and np.isfinite(metrics["idf1"])
                                   and abs(rebuilt - metrics["idf1"]) < 1e-9)
    return metrics, corr


def overall(rows: list[dict]) -> dict:
    """Pooled MOTA and IDF1 over sequences, with the formulas of py-motmetrics' overall row."""
    fn = sum(r["num_misses"] for r in rows)
    fp = sum(r["num_false_positives"] for r in rows)
    ids = sum(r["num_switches"] for r in rows)
    gt = sum(r["num_objects"] for r in rows)
    pred = sum(r["num_predictions"] for r in rows)
    idtp = sum(r["idtp"] for r in rows)
    return {
        "mota": 1.0 - (fn + fp + ids) / gt if gt else float("nan"),
        "idf1": 2.0 * idtp / (gt + pred) if (gt + pred) else float("nan"),
        "num_switches": ids, "num_false_positives": fp, "num_misses": fn,
        "num_objects": gt, "num_predictions": pred, "sequences": len(rows),
    }
