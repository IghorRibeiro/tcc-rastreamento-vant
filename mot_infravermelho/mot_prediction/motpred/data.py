"""Leitura de dados do Track 3: GT, trajetorias MOT, video, densidade.

Reutiliza src/inference/common.py do LAFusion (split de validacao, nome do
video, load_mot_file) para garantir as MESMAS 30 sequencias e o mesmo parser.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .common import lafusion_common, rpath

MOT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "conf", "a", "b", "c"]


def val_sequences(cfg: dict) -> list[str]:
    return list(lafusion_common(cfg).get_val_sequences())


def gt_path(cfg: dict, seq: str) -> Path:
    return rpath(cfg["paths"]["track3_label_root"]) / seq / "gt" / "gt.txt"


def video_path(cfg: dict, seq: str) -> Path:
    name = lafusion_common(cfg).seq_to_video_filename(seq)
    return rpath(cfg["paths"]["track3_video_root"]) / name


def trajectory_path(cfg: dict, run_suffix: str, seq: str) -> Path:
    """Arquivo MOT de uma rodada do LAFusion: runs/mot_eval/<seq>_<sufixo>.txt."""
    return rpath(cfg["paths"]["mot_eval_dir"]) / f"{seq}_{run_suffix}.txt"


def load_mot(cfg: dict, path: str | Path) -> pd.DataFrame:
    """load_mot_file do LAFusion; arquivo vazio vira DataFrame vazio."""
    path = Path(path)
    try:
        df = lafusion_common(cfg).load_mot_file(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=MOT_COLUMNS)
    if df.empty:
        df = pd.DataFrame({c: pd.Series(dtype="float64") for c in MOT_COLUMNS})
    df["frame"] = df["frame"].astype(np.int64)
    df["id"] = df["id"].astype(np.int64)
    for col in ("x", "y", "w", "h"):
        df[col] = df[col].astype(np.float64)
    return df


class GTIndex:
    """Estruturas de consulta do ground truth de uma sequencia."""

    def __init__(self, gt_df: pd.DataFrame):
        self.df = gt_df
        self.by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.by_id: dict[int, dict[int, tuple[float, float, float]]] = {}
        self.boxes_by_id: dict[int, dict[int, tuple[float, float, float, float]]] = {}
        for frame, grp in gt_df.groupby("frame", sort=True):
            ids = grp["id"].to_numpy(dtype=np.int64)
            boxes = grp[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
            self.by_frame[int(frame)] = (ids, boxes)
        for oid, frame, x, y, w, h in zip(gt_df["id"], gt_df["frame"], gt_df["x"], gt_df["y"],
                                          gt_df["w"], gt_df["h"]):
            x, y, w, h = float(x), float(y), float(w), float(h)
            self.by_id.setdefault(int(oid), {})[int(frame)] = (x + w / 2.0, y + h / 2.0, math.hypot(w, h))
            self.boxes_by_id.setdefault(int(oid), {})[int(frame)] = (x, y, w, h)
        counts = gt_df.groupby("frame")["id"].count()
        self.count_by_frame = {int(f): int(n) for f, n in counts.items()}
        # Mesma definicao de paper_figures/metrics/density.py::compute_density_stats
        self.mean_density = float(counts.mean()) if len(counts) else float("nan")
        self.max_density = int(counts.max()) if len(counts) else 0
        self.n_frames_annotated = int(len(counts))
        self.last_frame = int(gt_df["frame"].max()) if len(gt_df) else 0

    def n_gt(self, frame: int) -> int:
        return self.count_by_frame.get(int(frame), 0)


def load_gt(cfg: dict, seq: str) -> tuple[pd.DataFrame, GTIndex]:
    path = gt_path(cfg, seq)
    if not path.exists():
        raise SystemExit(f"gt.txt nao encontrado para {seq}: {path}")
    df = load_mot(cfg, path)
    return df, GTIndex(df)


def iter_video_frames(path: str | Path, max_frames: int | None = None):
    """Quadros do .mp4 com indice 1-based (igual ao run_sahi_flit_nsa.py)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {path}")
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            yield index, frame
            if max_frames is not None and index >= max_frames:
                break
    finally:
        cap.release()


def video_properties(path: str | Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"aberto": False}
    try:
        return {
            "aberto": True,
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "quadros": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "largura": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "altura": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def iou_xywh(a, b) -> float:
    """IoU de caixas (x, y, w, h); mesma formula de uav_common.iou."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
