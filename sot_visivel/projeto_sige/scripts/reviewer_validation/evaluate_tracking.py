#!/usr/bin/env python3
"""Evaluate tracker prediction files against local YOLO-format DUT labels.

This evaluator deliberately does not depend on the original DUT MP4/GT path.
It reports both the legacy thresholded accuracy used by the manuscript and
additional statistics requested by the reviewers: per-sequence mean/std,
median, range, and 95% confidence intervals.  Prediction errors are computed
with the same constant-velocity Kalman convention used by the project's
historical evaluation scripts, but without damping so that tracker comparisons
are not confounded by the proposed predictor.

python3 scripts/reviewer_validation/evaluate_tracking.py --methods \
    ODTrack=/Users/ighorribeiro/PycharmProjects/TesteYolo26/avaliacao_tracking/predicoes/reviewer_validation/odtrack \
    OSTrack=/Users/ighorribeiro/PycharmProjects/TesteYolo26/avaliacao_tracking/predicoes/reviewer_validation/ostrack \
    ByteTrack=/Users/ighorribeiro/PycharmProjects/TesteYolo26/avaliacao_tracking/predicoes/reviewer_validation/bytetrack \
    DeepSORT=/Users/ighorribeiro/PycharmProjects/TesteYolo26/avaliacao_tracking/predicoes/reviewer_validation/deepsort \
    "BoT-SORT_YOLO26n"=/Users/ighorribeiro/PycharmProjects/TesteYolo26/avaliacao_tracking/predicoes/exp01_yolo26n_botsort_padrao
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover - optional dependency
    student_t = None


HORIZONS = (5, 10, 15, 30)
IOU_THRESHOLDS = np.linspace(0.0, 1.0, 21)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional name=prediction_dir pairs. Default evaluates reviewer ByteTrack, DeepSORT and existing BoT-SORT n.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--img-width", type=int, default=1920)
    parser.add_argument("--img-height", type=int, default=1080)
    parser.add_argument(
        "--damping",
        type=float,
        default=1.0,
        help="Fator de amortecimento de velocidade aplicado ao Kalman de predicao "
             "(1.0 = sem damping, comportamento anterior; 0.95 = mesmo damping do "
             "pipeline Hibrido oficial). Aplica-se igualmente a TODOS os metodos "
             "avaliados nesta chamada -- use chamadas separadas para reportar "
             "resultados com e sem damping.",
    )
    return parser.parse_args()


def parse_method_args(project_root: Path, values: list[str] | None) -> dict[str, Path]:
    if values:
        result = {}
        for item in values:
            if "=" not in item:
                raise ValueError(f"Method must be NAME=DIR, got {item!r}")
            name, directory = item.split("=", 1)
            result[name] = Path(directory).expanduser()
        return result
    return {
        "ByteTrack": project_root / "avaliacao_tracking" / "predicoes" / "reviewer_validation" / "bytetrack",
        "DeepSORT": project_root / "avaliacao_tracking" / "predicoes" / "reviewer_validation" / "deepsort",
        "BoT-SORT YOLO26n": project_root / "avaliacao_tracking" / "predicoes" / "exp01_yolo26n_botsort_padrao",
        "BoT-SORT ajustado n": project_root / "avaliacao_tracking" / "predicoes" / "exp03_yolo26n_botsort_ajustado",
    }


def load_gt(label_path: Path, img_width: int, img_height: int) -> tuple[float, float, float, float] | None:
    if not label_path.exists():
        return None
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    values = lines[0].split()
    if len(values) < 5:
        return None
    _, cx, cy, width, height = map(float, values[:5])
    return (
        (cx - width / 2.0) * img_width,
        (cy - height / 2.0) * img_height,
        width * img_width,
        height * img_height,
    )


def load_predictions(prediction_path: Path) -> dict[int, tuple[float, float, float, float, float, float]]:
    """Return frame -> x,y,w,h,confidence,time_ms; first row wins per frame."""
    predictions = {}
    if not prediction_path.exists():
        return predictions
    for raw in prediction_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.replace(";", ",").split(",")]
        if len(parts) < 6:
            parts = raw.split()
        if len(parts) < 6:
            continue
        try:
            frame = int(float(parts[0]))
            x, y, w, h, conf = map(float, parts[1:6])
            time_ms = float(parts[6]) if len(parts) >= 7 else math.nan
        except ValueError:
            continue
        predictions.setdefault(frame, (x, y, w, h, conf, time_ms))
    return predictions


def iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def gt_centers(gt: dict[int, tuple[float, float, float, float] | None]) -> dict[int, tuple[float, float] | None]:
    return {
        frame: None if box is None else (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        for frame, box in gt.items()
    }


def kalman_prediction_errors(
    centers: dict[int, tuple[float, float]],
    gt_center: dict[int, tuple[float, float] | None],
    damping: float = 1.0,
) -> dict[int, list[float]]:
    """Predict future center from tracker measurements with CV Kalman.

    damping: multiplies the velocity components (vx, vy) of the filter state
    after every correct() step, exactly like run_hybrid_no_mog2.py's
    damp_state(). damping=1.0 (default) reproduces the previous undamped
    behaviour used for the cross-method comparison in Table 1; damping=0.95
    reproduces the proposed Kalman Damped predictor and is the single source
    of truth for any "damped" number reported for any tracker (CSRT, KCF,
    MOSSE, MedianFlow, or the full Híbrido pipeline).
    """
    errors = {h: [] for h in HORIZONS}
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    initialized = False
    for frame in sorted(centers):
        cx, cy = centers[frame]
        if not initialized:
            state = np.array([[cx], [cy], [0.0], [0.0]], np.float32)
            kalman.statePre = state.copy()
            kalman.statePost = state.copy()
            initialized = True
            continue
        kalman.predict()
        kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        if damping != 1.0:
            damped = kalman.statePost.copy()
            damped[2, 0] *= damping
            damped[3, 0] *= damping
            kalman.statePost = damped
        state = kalman.statePost.copy()
        for horizon in HORIZONS:
            future = frame + horizon
            target = gt_center.get(future)
            if target is None:
                continue
            predicted = state.copy()
            for _ in range(horizon):
                predicted = kalman.transitionMatrix @ predicted
            errors[horizon].append(float(np.linalg.norm(predicted[:2, 0] - np.array(target))))
    return errors


def load_sequence_gt(images_dir: Path, labels_dir: Path, video_id: str, width: int, height: int) -> dict[int, tuple[float, float, float, float] | None]:
    label_paths = sorted(labels_dir.glob(f"{video_id}_*.txt"))
    if not label_paths:
        raise FileNotFoundError(f"No labels found for {video_id}")
    return {int(path.stem.rsplit("_", 1)[1]): load_gt(path, width, height) for path in label_paths}


def evaluate_video(images_dir: Path, labels_dir: Path, prediction_path: Path, video_id: str, width: int, height: int, damping: float = 1.0) -> dict:
    gt = load_sequence_gt(images_dir, labels_dir, video_id, width, height)
    pred = load_predictions(prediction_path)
    frames = sorted(set(gt) | set(pred))
    present_ious = []
    success_by_threshold = {float(th): [] for th in IOU_THRESHOLDS}
    threshold_hits = 0
    total = len(frames)
    absent_hits = 0
    for frame in frames:
        gt_box = gt.get(frame)
        prediction = pred.get(frame)
        pred_box = None if prediction is None else prediction[:4]
        conf = 0.0 if prediction is None else prediction[4]
        if gt_box is not None:
            overlap = iou(gt_box, pred_box) if pred_box is not None else 0.0
            present_ious.append(overlap)
            threshold_hits += int(overlap >= 0.5)
            for th in IOU_THRESHOLDS:
                success_by_threshold[float(th)].append(float(overlap >= th))
        else:
            absent_hits += int(prediction is None or conf < 0.25)

    present_count = len(present_ious)
    sa_legacy = (threshold_hits + absent_hits) / total if total else math.nan
    sr05 = threshold_hits / present_count if present_count else math.nan
    mean_iou = float(np.mean(present_ious)) if present_ious else math.nan
    success_curve = [np.mean(success_by_threshold[float(th)]) for th in IOU_THRESHOLDS if success_by_threshold[float(th)]]
    success_auc = float(np.trapezoid(success_curve, IOU_THRESHOLDS[: len(success_curve)])) if success_curve else math.nan

    centers = {
        frame: (prediction[0] + prediction[2] / 2.0, prediction[1] + prediction[3] / 2.0)
        for frame, prediction in pred.items()
    }
    errors = kalman_prediction_errors(centers, gt_centers(gt), damping=damping)
    row = {
        "video": video_id,
        "n_frames": total,
        "n_gt_present": present_count,
        "n_predictions": len(pred),
        "coverage": len(pred) / total if total else math.nan,
        "SA_legacy_IoU50": sa_legacy,
        "SR@0.5": sr05,
        "mean_IoU_present": mean_iou,
        "success_AUC": success_auc,
        "time_ms_mean": float(np.nanmean([p[5] for p in pred.values()])) if pred else math.nan,
    }
    if row["time_ms_mean"] and not math.isnan(row["time_ms_mean"]):
        row["FPS_from_log"] = 1000.0 / row["time_ms_mean"]
    else:
        row["FPS_from_log"] = math.nan
    for horizon, values in errors.items():
        row[f"pred_error_H{horizon}_mean"] = float(np.mean(values)) if values else math.nan
        row[f"pred_error_H{horizon}_median"] = float(np.median(values)) if values else math.nan
        row[f"pred_error_H{horizon}_std_within"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        row[f"pred_error_H{horizon}_n"] = len(values)
    return row


def ci95(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return math.nan, math.nan
    sem = float(np.std(clean, ddof=1) / math.sqrt(len(clean)))
    multiplier = float(student_t.ppf(0.975, len(clean) - 1)) if student_t is not None else 1.96
    mean = float(np.mean(clean))
    return mean - multiplier * sem, mean + multiplier * sem


def summarize(df: pd.DataFrame, method: str) -> pd.DataFrame:
    metric_names = [
        "coverage", "SA_legacy_IoU50", "SR@0.5", "mean_IoU_present", "success_AUC",
        "time_ms_mean", "FPS_from_log",
        *[f"pred_error_H{h}_mean" for h in HORIZONS],
    ]
    rows = []
    for metric in metric_names:
        values = pd.to_numeric(df[metric], errors="coerce")
        low, high = ci95(values)
        rows.append({
            "method": method,
            "metric": metric,
            "n_sequences": int(values.notna().sum()),
            "mean": float(values.mean()),
            "std_between_sequences": float(values.std(ddof=1)),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
            "ci95_low": low,
            "ci95_high": high,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    images_dir = root / "datasets" / "anti_uav_tracking_test_final" / "images" / "test"
    labels_dir = root / "datasets" / "anti_uav_tracking_test_final" / "labels" / "test"
    output_dir = (args.output_dir or root / "scripts" / "reviewer_validation" / "results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    damping_tag = "undamped" if args.damping == 1.0 else f"damped{args.damping:.2f}".replace(".", "")
    methods = parse_method_args(root, args.methods)
    video_ids = sorted({p.name.split("_")[0] for p in images_dir.glob("video*_*.jpg")})
    all_rows: list[dict] = []
    summary_rows: list[pd.DataFrame] = []

    for method, directory in methods.items():
        if not directory.exists():
            print(f"Skipping {method}: missing {directory}")
            continue
        method_rows = []
        for video_id in video_ids:
            sample = next(iter(images_dir.glob(f"{video_id}_*.jpg")))
            image = cv2.imread(str(sample))
            if image is None:
                raise RuntimeError(f"Could not read {sample}")
            height, width = image.shape[:2]
            prediction_path = directory / f"{video_id}.txt"
            row = evaluate_video(images_dir, labels_dir, prediction_path, video_id, width, height, damping=args.damping)
            row["method"] = method
            row["damping"] = args.damping
            method_rows.append(row)
        method_df = pd.DataFrame(method_rows)
        method_df.to_csv(output_dir / f"per_video_{method.replace(' ', '_').replace('/', '_')}_{damping_tag}.csv", index=False, float_format="%.8f")
        all_rows.extend(method_rows)
        summary_rows.append(summarize(method_df, method))
        print(f"{method} [{damping_tag}]: {len(method_df)} sequences; mean SR@0.5={method_df['SR@0.5'].mean():.4f}; mean H30={method_df['pred_error_H30_mean'].mean():.2f}px")

    if not all_rows:
        raise RuntimeError("No prediction directory was found")
    all_df = pd.DataFrame(all_rows)
    summary_df = pd.concat(summary_rows, ignore_index=True)
    summary_df["damping"] = args.damping
    all_df.to_csv(output_dir / f"all_methods_per_video_{damping_tag}.csv", index=False, float_format="%.8f")
    summary_df.to_csv(output_dir / f"summary_mean_std_ci95_{damping_tag}.csv", index=False, float_format="%.8f")
    pivot = summary_df.pivot(index="metric", columns="method", values="mean")
    pivot.to_csv(output_dir / f"summary_mean_pivot_{damping_tag}.csv", float_format="%.8f")
    print(f"Saved results ({damping_tag}) in {output_dir}")


if __name__ == "__main__":
    main()