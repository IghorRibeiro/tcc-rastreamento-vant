#!/usr/bin/env python3
"""Run alternative trackers on the DUT Anti-UAV image sequences.

The project contains extracted frames rather than the original MP4 files.  This
runner therefore processes each sorted videoXX_*.jpg sequence directly and
writes predictions in the same CSV-like format consumed by the legacy evaluator:
frame,x,y,w,h,confidence,time_ms

Trackers:
  - bytetrack: Ultralytics ByteTrack with a YOLO detector.
  - deepsort: deep-sort-realtime with the MobileNet embedder and the same YOLO
    detector evaluated on every frame.

Examples:
  python run_alt_trackers.py --tracker bytetrack --video video01
  python run_alt_trackers.py --tracker both
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


VIDEO_RE = "video*_*.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Root of the TesteYolo26 project.",
    )
    parser.add_argument(
        "--tracker",
        choices=("bytetrack", "deepsort", "both"),
        default="both",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Only process one sequence, e.g. video01. Default: all 20 videos.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="YOLO weights. Default: pesos_treinados/yolo26n_dut_best.pt.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Ultralytics device, e.g. mps, cpu, 0.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: avaliacao_tracking/predicoes/reviewer_validation.",
    )
    return parser.parse_args()


def video_ids(images_dir: Path, only: str | None) -> list[str]:
    found = sorted({p.name.split("_")[0] for p in images_dir.glob(VIDEO_RE)})
    if only is not None:
        if only not in found:
            raise FileNotFoundError(f"Sequence {only!r} not found under {images_dir}")
        return [only]
    return found


def frame_paths(images_dir: Path, video_id: str) -> list[Path]:
    paths = sorted(images_dir.glob(f"{video_id}_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No frames found for {video_id} under {images_dir}")
    return paths


def frame_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def xyxy_to_xywh(xyxy: Iterable[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def choose_yolo_box(result) -> tuple[tuple[float, float, float, float], float] | None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    confs = boxes.conf.detach().cpu().numpy()
    idx = int(np.argmax(confs))
    return xyxy_to_xywh(boxes.xyxy[idx].detach().cpu().numpy()), float(confs[idx])


def write_prediction(handle, frame: int, bbox: tuple[float, float, float, float], conf: float, elapsed_ms: float) -> None:
    x, y, w, h = bbox
    handle.write(f"{frame},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{conf:.6f},{elapsed_ms:.3f}\n")


def run_bytetrack(model_path: Path, images_dir: Path, video_id: str, output_file: Path, device: str, conf: float) -> None:
    """Run Ultralytics ByteTrack frame by frame with persistent state per video."""
    model = YOLO(str(model_path))
    paths = frame_paths(images_dir, video_id)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions = 0
    with output_file.open("w", encoding="utf-8") as handle:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Could not decode {path}")
            start = time.perf_counter()
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf,
                device=device,
                verbose=False,
            )[0]
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            boxes = result.boxes
            if boxes is None or len(boxes) == 0 or boxes.id is None:
                continue
            # Single-object protocol: use the highest-confidence active track.
            confs = boxes.conf.detach().cpu().numpy()
            idx = int(np.argmax(confs))
            bbox = xyxy_to_xywh(boxes.xyxy[idx].detach().cpu().numpy())
            box_conf = float(confs[idx])
            write_prediction(handle, frame_number(path), bbox, box_conf, elapsed_ms)
            predictions += 1
    print(f"ByteTrack {video_id}: {len(paths)} frames, {predictions} predictions -> {output_file}")


def run_deepsort(model_path: Path, images_dir: Path, video_id: str, output_file: Path, device: str, conf: float) -> None:
    """Run DeepSORT with a MobileNet appearance embedder and YOLO detections."""
    from deep_sort_realtime.deepsort_tracker import DeepSort

    model = YOLO(str(model_path))
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.7,
        nn_budget=None,
        embedder="mobilenet",
        embedder_gpu=False,
        half=False,
    )
    paths = frame_paths(images_dir, video_id)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions = 0
    with output_file.open("w", encoding="utf-8") as handle:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Could not decode {path}")
            start = time.perf_counter()
            result = model(frame, conf=conf, device=device, verbose=False)[0]
            detections = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                for coords, score, cls in zip(xyxy, confs, classes):
                    x, y, w, h = xyxy_to_xywh(coords)
                    detections.append(([x, y, w, h], float(score), int(cls)))
            tracks = tracker.update_tracks(detections, frame=frame)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            active = [track for track in tracks if track.is_confirmed() and track.time_since_update <= 1]
            if not active:
                continue
            # Single-object protocol: prefer the most recently detected track,
            # then confidence, avoiding duplicate boxes in a frame.
            active.sort(key=lambda track: (track.get_det_conf() or 0.0, -track.time_since_update), reverse=True)
            chosen = active[0]
            x1, y1, x2, y2 = chosen.to_ltrb()
            bbox = xyxy_to_xywh((x1, y1, x2, y2))
            box_conf = float(chosen.get_det_conf() or 0.0)
            write_prediction(handle, frame_number(path), bbox, box_conf, elapsed_ms)
            predictions += 1
    print(f"DeepSORT {video_id}: {len(paths)} frames, {predictions} predictions -> {output_file}")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    images_dir = project_root / "datasets" / "anti_uav_tracking_test_final" / "images" / "test"
    model_path = (args.model or (project_root / "pesos_treinados" / "yolo26n_dut_best.pt")).resolve()
    output_root = (args.output_root or (project_root / "avaliacao_tracking" / "predicoes" / "reviewer_validation")).resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ids = video_ids(images_dir, args.video)
    print(f"Project: {project_root}")
    print(f"Model: {model_path}")
    print(f"Videos: {', '.join(ids)}")
    requested = ("bytetrack", "deepsort") if args.tracker == "both" else (args.tracker,)
    for tracker_name in requested:
        for video_id in ids:
            output_file = output_root / tracker_name / f"{video_id}.txt"
            if tracker_name == "bytetrack":
                run_bytetrack(model_path, images_dir, video_id, output_file, args.device, args.conf)
            else:
                run_deepsort(model_path, images_dir, video_id, output_file, args.device, args.conf)


if __name__ == "__main__":
    main()
