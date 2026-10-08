#!/usr/bin/env python3
"""Run OpenCV local trackers with periodic YOLO re-anchoring.

The runner follows the project's KCF/CSRT protocol while exposing the tracker
as a command-line parameter. It writes one CSV-like TXT file per sequence with
frame,x,y,w,h,confidence,time_ms, plus a run_summary.csv containing FPS.

Examples:
  python scripts/tracking/run_opencv_trackers.py --tracker mosse \
    --dataset-root '/Volumes/1T Ighor/Drone_Tracking/Datasets/Anti-UAV-Tracking/Anti-UAV-Tracking-V0' \
    --model pesos_treinados/yolo26n_dut_best.pt --device mps \
    --output-dir avaliacao_tracking/predicoes/mosse_hibrido
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


TRACKER_NAMES = ("csrt", "kcf", "mosse", "medianflow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", choices=TRACKER_NAMES, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory containing video01/video01.mp4, ...")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--videos", nargs="*", default=None, help="Optional IDs such as video01 video07")
    parser.add_argument("--init-conf", type=float, default=0.5)
    parser.add_argument("--switch-conf", type=float, default=0.7)
    parser.add_argument("--yolo-period", type=int, default=15)
    return parser.parse_args()


def create_tracker(name: str):
    if name == "csrt":
        return cv2.TrackerCSRT_create()
    if name == "kcf":
        return cv2.TrackerKCF_create()
    legacy = getattr(cv2, "legacy", None)
    if legacy is None:
        raise RuntimeError("Este OpenCV não expõe cv2.legacy; instale opencv-contrib-python.")
    if name == "mosse" and hasattr(legacy, "TrackerMOSSE_create"):
        return legacy.TrackerMOSSE_create()
    if name == "medianflow" and hasattr(legacy, "TrackerMedianFlow_create"):
        return legacy.TrackerMedianFlow_create()
    raise RuntimeError(f"Tracker {name} não está disponível nesta instalação do OpenCV.")


def best_detection(detector: YOLO, frame: np.ndarray, device: str, confidence: float):
    results = detector(frame, device=device, conf=confidence, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    confidences = boxes.conf.detach().cpu().numpy()
    index = int(confidences.argmax())
    x1, y1, x2, y2 = boxes.xyxy[index].detach().cpu().numpy()
    x1, y1, x2, y2 = [int(round(v)) for v in (x1, y1, x2, y2)]
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1), float(confidences[index])


def run_sequence(video_id: str, args: argparse.Namespace, detector: YOLO) -> dict:
    video_path = args.dataset_root / video_id / f"{video_id}.mp4"
    if not video_path.exists():
        print(f"AVISO: {video_path} não encontrado; sequência ignorada")
        return {"video": video_id, "status": "missing", "frames": 0, "predictions": 0, "fps": np.nan}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir {video_path}")
    tracker = None
    last_confidence = 0.0
    frame_number = 0
    predictions = []
    total_elapsed = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_number += 1
        start = time.perf_counter()
        current = None

        if tracker is None:
            current = best_detection(detector, frame, args.device, args.init_conf)
            if current is not None:
                x, y, w, h, confidence = current
                tracker = create_tracker(args.tracker)
                tracker.init(frame, (x, y, w, h))
                last_confidence = confidence
        else:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(round(v)) for v in bbox]
                current = (x, y, max(1, w), max(1, h), last_confidence)
                if frame_number % args.yolo_period == 0:
                    detection = best_detection(detector, frame, args.device, args.switch_conf)
                    if detection is not None:
                        x, y, w, h, confidence = detection
                        tracker = create_tracker(args.tracker)
                        tracker.init(frame, (x, y, w, h))
                        last_confidence = confidence
                        current = detection
            else:
                tracker = None
                last_confidence = 0.0

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        total_elapsed += elapsed_ms
        if current is not None:
            x, y, w, h, confidence = current
            predictions.append((frame_number, x, y, w, h, confidence, elapsed_ms))

    cap.release()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"{video_id}.txt"
    with output_file.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(",".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in row) + "\n")
    fps = 1000.0 * frame_number / total_elapsed if total_elapsed > 0 else np.nan
    print(f"{args.tracker} {video_id}: {frame_number} frames, {len(predictions)} predictions, {fps:.2f} FPS")
    return {"video": video_id, "status": "ok", "frames": frame_number, "predictions": len(predictions), "time_ms_total": total_elapsed, "time_ms_mean_all_frames": total_elapsed / frame_number if frame_number else np.nan, "fps": fps}


def main() -> None:
    args = parse_args()
    args.model = args.model.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    detector = YOLO(str(args.model))
    video_ids = args.videos or [f"video{i:02d}" for i in range(1, 21)]
    rows = [run_sequence(video_id, args, detector) for video_id in video_ids]
    with (args.output_dir / "run_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Resultados {args.tracker} salvos em {args.output_dir}")


if __name__ == "__main__":
    main()
