"""Avaliador MOT identico a src/inference/evaluate_mot.py + correspondencias.

O laco por quadro (conjunto de quadros, ordem das linhas, iou_matrix com
max_iou = 0,5, MOTAccumulator(auto_id=True)) e o mesmo do LAFusion; a unica
diferenca e que os quadros sao agrupados com groupby (mesma ordem de linhas)
em vez de filtrados um a um, o que so muda o tempo de execucao.

Alem das metricas, extrai:
  * pi_CLEAR: correspondencia por quadro (eventos MATCH/SWITCH) usada na MOTA;
  * pi_ID: mapeamento global do IDF1 (maximiza IDTP), validado contra o IDF1
    do proprio py-motmetrics.
"""

from __future__ import annotations

import numpy as np

# Mesmo shim do evaluate_mot.py (np.asfarray removido no NumPy 2.0)
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


def build_accumulator(gt_df: pd.DataFrame, pred_df: pd.DataFrame, max_iou: float = 0.5):
    """Replica o laco de evaluate_mot.py::main para uma sequencia."""
    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt_df["frame"]) | set(pred_df["frame"]))
    gt_groups = {k: g for k, g in gt_df.groupby("frame", sort=False)}
    pred_groups = {k: g for k, g in pred_df.groupby("frame", sort=False)}
    gt_empty = gt_df.iloc[0:0]
    pred_empty = pred_df.iloc[0:0]

    for frame in all_frames:
        gt_frame = gt_groups.get(frame, gt_empty)
        pred_frame = pred_groups.get(frame, pred_empty)

        gt_ids = gt_frame["id"].values
        pred_ids = pred_frame["id"].values

        gt_boxes = gt_frame[["x", "y", "w", "h"]].values
        pred_boxes = pred_frame[["x", "y", "w", "h"]].values

        dist_matrix = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=max_iou)
        acc.update(gt_ids, pred_ids, dist_matrix)

    return acc, [int(f) for f in all_frames]


def compute_metrics(acc, name: str) -> dict:
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=METRICS, name=name)
    out = {}
    for key in METRICS:
        value = summary.loc[name, key]
        out[key] = float(value) if value is not None else float("nan")
    return out


def correspondences(acc, all_frames: list[int], gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """pi_CLEAR (por quadro) e pi_ID (global, IDF1) a partir dos eventos do acumulador."""
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

    return {
        "clear": clear,
        "idf1": idmap,
        "idtp_mapeamento": idtp_total,
        "n_matches_clear": len(clear),
    }


def evaluate_mot_sequence(gt_df: pd.DataFrame, pred_df: pd.DataFrame, name: str, max_iou: float = 0.5):
    """Metricas MOT + correspondencias + checagem do IDF1 reconstruido."""
    acc, frames = build_accumulator(gt_df, pred_df, max_iou=max_iou)
    metrics = compute_metrics(acc, name)
    corr = correspondences(acc, frames, gt_df, pred_df)
    denom = metrics["num_objects"] + metrics["num_predictions"]
    idf1_rebuilt = 2.0 * corr["idtp_mapeamento"] / denom if denom > 0 else float("nan")
    metrics["idf1_reconstruido"] = idf1_rebuilt
    metrics["idf1_confere"] = bool(np.isfinite(idf1_rebuilt) and np.isfinite(metrics["idf1"])
                                   and abs(idf1_rebuilt - metrics["idf1"]) < 1e-9)
    return metrics, corr


def overall(rows: list[dict]) -> dict:
    """Agregado geral com as formulas do overall do py-motmetrics."""
    fn = sum(r["num_misses"] for r in rows)
    fp = sum(r["num_false_positives"] for r in rows)
    ids = sum(r["num_switches"] for r in rows)
    gt = sum(r["num_objects"] for r in rows)
    pred = sum(r["num_predictions"] for r in rows)
    idtp = sum(r["idtp"] for r in rows)
    return {
        "mota": 1.0 - (fn + fp + ids) / gt if gt else float("nan"),
        "idf1": 2.0 * idtp / (gt + pred) if (gt + pred) else float("nan"),
        "num_switches": ids,
        "num_false_positives": fp,
        "num_misses": fn,
        "num_objects": gt,
        "num_predictions": pred,
        "sequencias": len(rows),
    }
