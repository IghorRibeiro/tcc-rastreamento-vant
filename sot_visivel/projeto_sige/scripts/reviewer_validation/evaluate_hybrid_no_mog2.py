#!/usr/bin/env python3
"""Evaluate the hybrid's own damped Kalman predictor without MOG2."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluate_tracking import gt_centers, load_predictions, load_sequence_gt

HORIZONS = (5, 10, 15, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--damping", type=float, default=0.95)
    parser.add_argument("--img-width", type=int, default=1920)
    parser.add_argument("--img-height", type=int, default=1080)
    return parser.parse_args()


def damped_errors(centers: dict[int, tuple[float, float]], targets: dict[int, tuple[float, float] | None], damping: float) -> dict[int, list[float]]:
    errors = {h: [] for h in HORIZONS}
    if not centers:
        return errors
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
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
        state = kalman.statePost.copy()
        state[2, 0] *= damping
        state[3, 0] *= damping
        kalman.statePost = state
        for horizon in HORIZONS:
            target = targets.get(frame + horizon)
            if target is None:
                continue
            state_pred = state.copy()
            for _ in range(horizon):
                state_pred = kalman.transitionMatrix @ state_pred
                state_pred[2, 0] *= damping
                state_pred[3, 0] *= damping
            errors[horizon].append(float(np.linalg.norm(state_pred[:2, 0] - np.array(target))))
    return errors


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    images_dir = root / "datasets" / "anti_uav_tracking_test_final" / "images" / "test"
    labels_dir = root / "datasets" / "anti_uav_tracking_test_final" / "labels" / "test"
    prediction_dir = (args.prediction_dir or root / "avaliacao_tracking" / "predicoes" / "reviewer_validation" / "hybrid_no_mog2").resolve()
    output_dir = (args.output_dir or root / "scripts" / "reviewer_validation" / "results_with_hybrid_no_mog2").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_ids = [f"video{i:02d}" for i in range(1, 21)]
    rows = []
    for video_id in video_ids:
        sample = next(images_dir.glob(f"{video_id}_*.jpg"))
        image = cv2.imread(str(sample))
        height, width = image.shape[:2]
        gt = load_sequence_gt(images_dir, labels_dir, video_id, width, height)
        pred = load_predictions(prediction_dir / f"{video_id}.txt")
        centers = {frame: (p[0] + p[2] / 2.0, p[1] + p[3] / 2.0) for frame, p in pred.items()}
        errors = damped_errors(centers, gt_centers(gt), args.damping)
        row = {"video": video_id, "damping": args.damping, "n_predictions": len(pred)}
        for horizon, values in errors.items():
            row[f"pred_error_H{horizon}_mean_exact"] = float(np.mean(values)) if values else math.nan
            row[f"pred_error_H{horizon}_median_exact"] = float(np.median(values)) if values else math.nan
            row[f"pred_error_H{horizon}_std_exact"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            row[f"pred_error_H{horizon}_n_exact"] = len(values)
        rows.append(row)
    per_video = pd.DataFrame(rows)
    per_video.to_csv(output_dir / "hybrid_no_mog2_exact_per_video.csv", index=False, float_format="%.8f")
    summary_rows = []
    for horizon in HORIZONS:
        metric = f"pred_error_H{horizon}_mean_exact"
        values = per_video[metric].dropna()
        summary_rows.append({"metric": metric, "n_sequences": len(values), "mean": values.mean(), "std_between_sequences": values.std(ddof=1), "median": values.median(), "min": values.min(), "max": values.max()})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "hybrid_no_mog2_exact_summary.csv", index=False, float_format="%.8f")
    print(summary.to_string(index=False))
    print(f"Saved exact hybrid prediction metrics in {output_dir}")


if __name__ == "__main__":
    main()
