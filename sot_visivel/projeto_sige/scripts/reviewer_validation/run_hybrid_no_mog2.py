#!/usr/bin/env python3
"""Run the final hybrid tracker without MOG2 on local DUT image sequences.

Pipeline: YOLO26n initialization/re-detection every N frames + CSRT local
tracking + constant-velocity Kalman state with damping.  No MOG2, background
subtraction, or external MP4/GT path is used.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--period", type=int, default=15)
    parser.add_argument("--init-conf", type=float, default=0.50)
    parser.add_argument("--switch-conf", type=float, default=0.70)
    parser.add_argument("--damping", type=float, default=0.95)
    parser.add_argument("--video", default=None)
    return parser.parse_args()


def tracker_factory():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT is not available in this OpenCV installation")


def detect(detector: YOLO, frame: np.ndarray, device: str, confidence: float):
    results = detector(frame, device=device, conf=confidence, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    confs = boxes.conf.detach().cpu().numpy()
    best = int(np.argmax(confs))
    x1, y1, x2, y2 = boxes.xyxy[best].detach().cpu().numpy().tolist()
    x1, y1, x2, y2 = [int(round(v)) for v in (x1, y1, x2, y2)]
    x, y = max(0, x1), max(0, y1)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    return (x, y, w, h, float(confs[best]))


def init_kalman(cx: float, cy: float) -> cv2.KalmanFilter:
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    state = np.array([[cx], [cy], [0.0], [0.0]], np.float32)
    kalman.statePre = state.copy()
    kalman.statePost = state.copy()
    return kalman


def damp_state(kalman: cv2.KalmanFilter, damping: float) -> None:
    state = kalman.statePost.copy()
    state[2, 0] *= damping
    state[3, 0] *= damping
    kalman.statePost = state


def sequence_paths(images_dir: Path, video_id: str) -> list[Path]:
    return sorted(images_dir.glob(f"{video_id}_*.jpg"))


def run_sequence(
    detector: YOLO,
    image_paths: list[Path],
    output_path: Path,
    device: str,
    period: int,
    init_conf: float,
    switch_conf: float,
    damping: float,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = None
    last_conf = 0.0
    kalman = None
    rows = []
    detection_calls = 0
    frame_times = []

    for index, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read {image_path}")
        frame_number = int(image_path.stem.rsplit("_", 1)[1])
        t0 = time.perf_counter()
        current = None

        if tracker is None:
            detection_calls += 1
            current = detect(detector, frame, device, init_conf)
            if current is not None:
                x, y, w, h, last_conf = current
                tracker = tracker_factory()
                tracker.init(frame, (x, y, w, h))
                cx, cy = x + w / 2.0, y + h / 2.0
                kalman = init_kalman(cx, cy)
        else:
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(round(value)) for value in box]
                w, h = max(1, w), max(1, h)
                current = (x, y, w, h, last_conf)
                cx, cy = x + w / 2.0, y + h / 2.0
                if kalman is not None:
                    kalman.predict()
                    kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
                    damp_state(kalman, damping)
                if index % period == 0:
                    detection_calls += 1
                    candidate = detect(detector, frame, device, switch_conf)
                    if candidate is not None and candidate[4] >= switch_conf:
                        x, y, w, h, last_conf = candidate
                        tracker = tracker_factory()
                        tracker.init(frame, (x, y, w, h))
                        current = candidate
                        cx, cy = x + w / 2.0, y + h / 2.0
                        kalman = init_kalman(cx, cy)
            else:
                tracker = None
                kalman = None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        frame_times.append(elapsed_ms)
        if current is not None:
            x, y, w, h, conf = current
            rows.append((frame_number, x, y, w, h, conf, elapsed_ms))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    mean_ms = float(np.mean(frame_times)) if frame_times else float("nan")
    return {
        "video": image_paths[0].stem.split("_")[0] if image_paths else output_path.stem,
        "frames": len(image_paths),
        "predictions": len(rows),
        "detector_calls": detection_calls,
        "mean_time_ms": mean_ms,
        "fps": 1000.0 / mean_ms if mean_ms > 0 else float("nan"),
        "output": str(output_path),
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    images_dir = root / "datasets" / "anti_uav_tracking_test_final" / "images" / "test"
    model_path = (args.model or root / "pesos_treinados" / "yolo26n_dut_best.pt").resolve()
    output_dir = (args.output_dir or root / "avaliacao_tracking" / "predicoes" / "reviewer_validation" / "hybrid_no_mog2").resolve()
    detector = YOLO(str(model_path))
    video_ids = [args.video] if args.video else [f"video{i:02d}" for i in range(1, 21)]
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    print(f"Model: {model_path}")
    print(f"Hybrid: YOLO26n + CSRT + Kalman CV Damped={args.damping}; MOG2=disabled")
    for video_id in video_ids:
        paths = sequence_paths(images_dir, video_id)
        if not paths:
            raise FileNotFoundError(f"No images found for {video_id} in {images_dir}")
        result = run_sequence(detector, paths, output_dir / f"{video_id}.txt", args.device, args.period, args.init_conf, args.switch_conf, args.damping)
        summaries.append(result)
        print(f"Hybrid no MOG2 {video_id}: {result['frames']} frames, {result['predictions']} predictions -> {result['output']}")
    with (output_dir / "run_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved run summary to {output_dir / 'run_summary.csv'}")


if __name__ == "__main__":
    main()
