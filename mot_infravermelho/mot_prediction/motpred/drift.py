"""Deriva do rastreador local depois de cada ancoragem (H4).

Mesma funcao para o MOT (Track 3, varios alvos) e para o SOT do SIGE (DUT, um
alvo): o GT entra como {quadro: (ids, caixas)} e as ancoragens como uma tabela
(id, frame, x, y, w, h, conf, kind). Definicoes em HYPOTHESES.md, 2.10.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import iou_xywh


def drift_analysis(traj_df: pd.DataFrame, anchors_df: pd.DataFrame, gt_by_frame: dict,
                   gt_boxes_by_id: dict, n_red: int, max_age: int, min_anchor_iou: float,
                   last_frame: int) -> tuple[pd.DataFrame, dict]:
    """Devolve (curva por idade, resumo da sequencia)."""
    boxes = {}
    for tid, frame, x, y, w, h in zip(traj_df["id"], traj_df["frame"], traj_df["x"], traj_df["y"],
                                      traj_df["w"], traj_df["h"]):
        boxes.setdefault((int(tid), int(frame)), (float(x), float(y), float(w), float(h)))

    ages, errs, ious = [], [], []
    n_segments = 0
    n_anchored = 0
    n_eligible = 0
    n_failed = 0
    anchor_ious = []

    if anchors_df is None or anchors_df.empty:
        return pd.DataFrame(columns=["age", "n", "mean_e_norm", "mean_iou", "frac_iou50"]), {
            "segmentos": 0, "segmentos_ancorados": 0, "segmentos_elegiveis": 0, "falhas": 0,
            "taxa_falha": math.nan, "taxa_deriva": math.nan, "e_norm_a0": math.nan,
            "e_norm_afim": math.nan, "iou_a0": math.nan, "iou_afim": math.nan, "n_pontos": 0}

    for tid, grp in anchors_df.sort_values(["id", "frame"]).groupby("id", sort=True):
        rows = list(grp.itertuples(index=False))
        for i, anchor in enumerate(rows):
            n_segments += 1
            t0 = int(anchor.frame)
            t_next = int(rows[i + 1].frame) if i + 1 < len(rows) else math.inf
            det_box = (float(anchor.x), float(anchor.y), float(anchor.w), float(anchor.h))
            ids, gt_boxes = gt_by_frame.get(t0, (np.empty(0, dtype=np.int64), np.empty((0, 4))))
            if len(ids) == 0:
                continue
            overlaps = [iou_xywh(det_box, tuple(b)) for b in gt_boxes]
            best = int(np.argmax(overlaps))
            if overlaps[best] < min_anchor_iou:
                continue
            n_anchored += 1
            anchor_ious.append(overlaps[best])
            g = int(ids[best])
            g_track = gt_boxes_by_id.get(g, {})
            eligible = t0 + n_red - 1 <= last_frame
            failed = False
            age = 0
            t = t0
            while t < t_next and age <= max_age:
                box = boxes.get((int(tid), t))
                if box is None:
                    if age <= n_red - 1:
                        failed = True
                    break
                gt_box = g_track.get(t)
                if gt_box is not None:
                    cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
                    gx, gy = gt_box[0] + gt_box[2] / 2.0, gt_box[1] + gt_box[3] / 2.0
                    diag = math.hypot(gt_box[2], gt_box[3])
                    ages.append(age)
                    errs.append(math.hypot(cx - gx, cy - gy) / diag if diag > 0 else math.nan)
                    ious.append(iou_xywh(box, gt_box))
                t += 1
                age += 1
            if eligible:
                n_eligible += 1
                n_failed += int(failed)

    points = pd.DataFrame({"age": ages, "e_norm": errs, "iou": ious})
    if points.empty:
        curve = pd.DataFrame(columns=["age", "n", "mean_e_norm", "mean_iou", "frac_iou50"])
    else:
        curve = points.groupby("age").agg(n=("e_norm", "size"), mean_e_norm=("e_norm", "mean"),
                                          mean_iou=("iou", "mean"),
                                          frac_iou50=("iou", lambda s: float((s >= 0.5).mean()))).reset_index()

    window = points[(points["age"] <= n_red - 1)].dropna()
    slope = math.nan
    if len(window) >= 3 and window["age"].nunique() >= 2:
        slope = float(np.polyfit(window["age"].to_numpy(float), window["e_norm"].to_numpy(float), 1)[0])

    def _at(age, col):
        sel = points[points["age"] == age][col].dropna()
        return float(sel.mean()) if len(sel) else math.nan

    summary = {
        "segmentos": n_segments,
        "segmentos_ancorados": n_anchored,
        "segmentos_elegiveis": n_eligible,
        "falhas": n_failed,
        "taxa_falha": n_failed / n_eligible if n_eligible else math.nan,
        "taxa_deriva": slope,
        "e_norm_a0": _at(0, "e_norm"),
        "e_norm_afim": _at(n_red - 1, "e_norm"),
        "iou_a0": _at(0, "iou"),
        "iou_afim": _at(n_red - 1, "iou"),
        "iou_ancora_medio": float(np.mean(anchor_ious)) if anchor_ious else math.nan,
        "n_pontos": int(len(window)),
    }
    return curve, summary


def gt_structures_from_sige(gt_frames: dict) -> tuple[dict, dict, int]:
    """Converte o GT do SIGE ({quadro: caixa ou None}) para as estruturas MOT (id = 1)."""
    by_frame, by_id = {}, {1: {}}
    for frame, box in gt_frames.items():
        if box is None:
            continue
        by_frame[int(frame)] = (np.array([1], dtype=np.int64), np.array([box], dtype=np.float64))
        by_id[1][int(frame)] = tuple(float(v) for v in box)
    last = max(gt_frames) if gt_frames else 0
    return by_frame, by_id, int(last)
