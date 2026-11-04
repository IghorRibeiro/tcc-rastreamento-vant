"""Single-target (DUT Anti-UAV) trajectory prediction from the trajectories saved by the SIGE code.

File formats (SIGE 2026 repository):
  * prediction file, one line per frame:  ``frame,x,y,w,h,conf[,time_ms]`` (one record per frame)
  * official ground truth ``videoNN_gt.txt``: one ``x y w h`` line per frame, negative values = no target
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .kalman import HORIZONS, corrected_states, forward


def read_predictions(path: Path) -> dict[int, tuple[float, float, float, float, float]]:
    """{frame: (x, y, w, h, conf)}; the first record of a frame wins."""
    predictions: dict[int, tuple[float, float, float, float, float]] = {}
    if not Path(path).exists():
        return predictions
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = [p.strip() for p in raw.replace(";", ",").split(",")]
        if len(parts) < 6:
            parts = raw.split()
        if len(parts) < 6:
            continue
        try:
            frame = int(float(parts[0]))
            x, y, w, h, conf = (float(v) for v in parts[1:6])
        except ValueError:
            continue
        predictions.setdefault(frame, (x, y, w, h, conf))
    return predictions


def read_official_gt(path: Path) -> dict[int, tuple[float, float, float, float] | None]:
    frames: dict[int, tuple[float, float, float, float] | None] = {}
    if not Path(path).exists():
        return frames
    for index, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw.split()
        if len(parts) != 4:
            continue
        x, y, w, h = (float(v) for v in parts)
        frames[index] = None if min(x, y, w, h) < 0 else (x, y, w, h)
    return frames


def centers_from_gt(gt_frames) -> dict:
    return {f: (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0) for f, b in gt_frames.items() if b is not None}


def centers_from_predictions(predictions) -> dict:
    return {f: (p[0] + p[2] / 2.0, p[1] + p[3] / 2.0) for f, p in predictions.items()}


def error_pairs(centers: dict, gt_centers: dict, damping: float, horizons=HORIZONS) -> dict:
    """{horizon: {frame: error_px}} for every frame at which the target exists at frame + horizon."""
    out = {h: {} for h in horizons}
    for frame, state in corrected_states(centers, damping):
        for horizon in horizons:
            target = gt_centers.get(frame + horizon)
            if target is None:
                continue
            predicted = forward(state, horizon, damping)
            out[horizon][frame] = float(np.hypot(predicted[0, 0] - target[0], predicted[1, 0] - target[1]))
    return out


def summarize(errors_by_frame: dict) -> dict:
    values = np.asarray(list(errors_by_frame.values()), dtype=float)
    if values.size == 0:
        return {"n": 0, "mean_px": math.nan, "std_px": math.nan, "median_px": math.nan}
    return {"n": int(values.size), "mean_px": float(values.mean()),
            "std_px": float(values.std(ddof=1)) if values.size > 1 else 0.0, "median_px": float(np.median(values))}


def per_video_table(method: str, videos: dict, dampings=(1.0, 0.95), horizons=HORIZONS) -> pd.DataFrame:
    """One row per (video, damping, H) with the error summary.

    ``videos`` maps video name -> (prediction_path, ground_truth_path).
    """
    rows = []
    for video, (pred_path, gt_path) in sorted(videos.items()):
        centers = centers_from_predictions(read_predictions(pred_path))
        gt_centers = centers_from_gt(read_official_gt(gt_path))
        for damping in dampings:
            pairs = error_pairs(centers, gt_centers, damping, horizons)
            for horizon in horizons:
                rows.append({"method": method, "video": video, "damping": damping, "H": horizon,
                             **summarize(pairs[horizon])})
    return pd.DataFrame(rows)


def common_support_table(pairs_hybrid: dict, pairs_reference: dict, reference: str, video: str, damping: float,
                         horizons=HORIZONS) -> list[dict]:
    """Mean error of the hybrid architecture and of a reference over the frames both report (same H, video)."""
    rows = []
    for horizon in horizons:
        a, b = pairs_hybrid[horizon], pairs_reference[horizon]
        shared = sorted(set(a) & set(b))
        rows.append({"reference": reference, "video": video, "damping": damping, "H": horizon,
                     "n_common": len(shared),
                     "mean_px_hybrid": float(np.mean([a[f] for f in shared])) if shared else math.nan,
                     "mean_px_reference": float(np.mean([b[f] for f in shared])) if shared else math.nan})
    return rows
