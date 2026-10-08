#!/usr/bin/env python3
"""
HardMining + SAHI-512 + GM-FLIT + NSA-Kalman + BoT-SORT.

Pipeline:
    OpenCV
        -> SAHI detection
        -> Ultralytics Results/Boxes
        -> BoT-SORT
        -> GM-FLIT association
        -> NSA-Kalman
        -> MOTChallenge output

This runner is intentionally based on the existing run_sahi_tracker.py
implementation so that the detector, SAHI configuration and tracker API
remain comparable with the previous experiments.

IMPORTANT:
    GM-FLIT and NSA-Kalman are applied through the existing project patches.
    The patches are installed before BOTSORT instances are created and are
    removed in a finally block after the experiment.

Expected experiment:
    HardMining + SAHI512 + GM-FLIT(gamma=0.55) + NSA-Kalman + BoT-SORT
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List

import cv2
import numpy as np
import torch
import yaml

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from ultralytics.trackers import BOTSORT
from ultralytics.engine.results import Results


# ----------------------------------------------------------------------
# Project paths
# ----------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "postprocessing"))

from common import (  # noqa: E402
    MOT_EVAL_DIR,
    VIDEO_ROOT,
    get_val_sequences,
    seq_to_video_filename,
)

from robust_flit_association import (  # noqa: E402
    patch_flit,
    unpatch_flit,
)

from nsa_kalman_patch import (  # noqa: E402
    patch_nsa_kalman,
    unpatch_nsa_kalman,
)


# ----------------------------------------------------------------------
# Fixed experiment configuration
# ----------------------------------------------------------------------

FRAME_W = 640
FRAME_H = 512

SAHI_SLICE = 512
SAHI_OVERLAP = 0.20

CONF_THRESHOLD = 0.25

GM_FLIT_GAMMA = 0.55

TRACKER_DEFAULT = "botsort.yaml"


# ----------------------------------------------------------------------
# Tracker configuration
# ----------------------------------------------------------------------

def resolve_tracker_cfg(arg: str) -> Path:
    """
    Resolve tracker YAML from:
        1. explicit filesystem path;
        2. Ultralytics tracker configuration directory;
        3. ultra-botsort alias.
    """

    from ultralytics import __file__ as ULT_FILE

    ultra_cfg = Path(ULT_FILE).parent / "cfg" / "trackers"

    path = Path(arg)

    if path.exists():
        return path.resolve()

    candidate = ultra_cfg / arg

    if candidate.exists():
        return candidate.resolve()

    if arg == "ultra-botsort":
        default = ultra_cfg / "botsort.yaml"

        if default.exists():
            return default.resolve()

    raise FileNotFoundError(
        f"Tracker configuration not found: {arg}"
    )


def load_tracker_config(cfg_path: Path) -> SimpleNamespace:
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    return SimpleNamespace(**cfg)


# ----------------------------------------------------------------------
# SAHI detection
# ----------------------------------------------------------------------

def build_sahi_detections(
    frame: np.ndarray,
    det_model: AutoDetectionModel,
    slice_h: int,
    slice_w: int,
    overlap: float,
    conf_thresh: float,
) -> np.ndarray:
    """
    Run SAHI on one frame.

    Returns:
        ndarray with shape (N, 6):

        [x1, y1, x2, y2, score, class_id]
    """

    result = get_sliced_prediction(
        frame,
        detection_model=det_model,
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_match_threshold=0.6,
        verbose=0,
    )

    height, width = frame.shape[:2]

    detections = []

    for obj in result.object_prediction_list:

        score = float(obj.score.value)

        if score < conf_thresh:
            continue

        x1, y1, x2, y2 = obj.bbox.to_xyxy()

        x1 = max(0.0, min(width - 1.0, float(x1)))
        y1 = max(0.0, min(height - 1.0, float(y1)))
        x2 = max(0.0, min(width - 1.0, float(x2)))
        y2 = max(0.0, min(height - 1.0, float(y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        class_id = obj.category.id

        detections.append(
            [
                x1,
                y1,
                x2,
                y2,
                score,
                float(class_id),
            ]
        )

    if not detections:
        return np.empty((0, 6), dtype=np.float32)

    return np.asarray(detections, dtype=np.float32)


# ----------------------------------------------------------------------
# Sequence processing
# ----------------------------------------------------------------------

def run_tracker_on_sequence(
    seq_name: str,
    det_model: AutoDetectionModel,
    tracker_cfg_path: Path,
    device: str,
    suffix: str,
) -> tuple[int, float]:

    video_path = VIDEO_ROOT / seq_to_video_filename(seq_name)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found for {seq_name}: {video_path}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    tracker_args = load_tracker_config(tracker_cfg_path)

    # IMPORTANT:
    # The patches are already active when BOTSORT is instantiated.
    tracker = BOTSORT(args=tracker_args)

    names = {0: "drone"}

    output_lines: List[str] = []

    frame_idx = 0

    start_time = time.perf_counter()

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_idx += 1

            # ----------------------------------------------------------
            # 1. SAHI detection
            # ----------------------------------------------------------

            detections = build_sahi_detections(
                frame=frame,
                det_model=det_model,
                slice_h=SAHI_SLICE,
                slice_w=SAHI_SLICE,
                overlap=SAHI_OVERLAP,
                conf_thresh=CONF_THRESHOLD,
            )

            # ----------------------------------------------------------
            # 2. Construct official Ultralytics Results
            # ----------------------------------------------------------

            box_tensor = torch.from_numpy(
                detections
            ).to(
                device=device,
                dtype=torch.float32,
            )

            result = Results(
                orig_img=frame,
                path="frame.jpg",
                names=names,
                boxes=box_tensor,
            )

            # Same representation used by the previous SAHI runner.
            det = result.boxes.cpu().numpy()

            # ----------------------------------------------------------
            # 3. BoT-SORT
            #
            # GM-FLIT modifies the association distance used internally.
            # NSA-Kalman modifies the Kalman measurement model.
            # ----------------------------------------------------------

            tracks = tracker.update(det, frame)

            if len(tracks):

                for track in tracks:

                    x1, y1, x2, y2 = track[:4]

                    track_id = int(track[4])

                    width = x2 - x1
                    height = y2 - y1

                    output_lines.append(
                        f"{frame_idx},"
                        f"{track_id},"
                        f"{x1:.2f},"
                        f"{y1:.2f},"
                        f"{width:.2f},"
                        f"{height:.2f},"
                        f"1,-1,-1,-1"
                    )

    finally:
        cap.release()

    elapsed = time.perf_counter() - start_time

    output_path = (
        MOT_EVAL_DIR
        / f"{seq_name}_{suffix}.txt"
    )

    output_path.write_text(
        "\n".join(output_lines)
    )

    return frame_idx, elapsed


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "HardMining + SAHI512 + GM-FLIT + "
            "NSA-Kalman + BoT-SORT"
        )
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="Path to HardMining YOLO checkpoint.",
    )

    parser.add_argument(
        "--device",
        default="mps",
        help="mps | cpu | cuda:0",
    )

    parser.add_argument(
        "--tracker-cfg",
        default=TRACKER_DEFAULT,
        help="BoT-SORT YAML configuration.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        help="Detection confidence threshold.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=GM_FLIT_GAMMA,
        help="Geman-McClure FLIT gamma.",
    )

    parser.add_argument(
        "--only-seq",
        nargs="*",
        default=None,
        help="Optional validation sequences to process.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Experiment name
    # ------------------------------------------------------------------

    suffix = "hardmining_sahi512_flit_nsa"

    MOT_EVAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("HARDMINING + SAHI512 + GM-FLIT + NSA-KALMAN + BOTSORT")
    print("=" * 72)
    print(f"Model:          {args.weights}")
    print(f"Device:         {args.device}")
    print(f"SAHI slice:     {SAHI_SLICE}")
    print(f"SAHI overlap:   {SAHI_OVERLAP}")
    print(f"Confidence:     {args.conf}")
    print(f"GM-FLIT gamma:  {args.gamma}")
    print(f"Tracker:        {args.tracker_cfg}")
    print(f"Output suffix:  {suffix}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Load detector
    # ------------------------------------------------------------------

    det_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # Resolve tracker
    # ------------------------------------------------------------------

    tracker_cfg_path = resolve_tracker_cfg(
        args.tracker_cfg
    )

    print(
        f"Tracker config: {tracker_cfg_path}"
    )

    # ------------------------------------------------------------------
    # Validation sequences
    # ------------------------------------------------------------------

    val_sequences = get_val_sequences()

    if args.only_seq:

        requested = set(args.only_seq)

        val_sequences = [
            seq
            for seq in val_sequences
            if seq in requested
        ]

    print(
        f"Validation sequences: {len(val_sequences)}"
    )

    # ------------------------------------------------------------------
    # Apply patches
    #
    # IMPORTANT:
    # They must be active before BOTSORT is instantiated.
    # ------------------------------------------------------------------

    flit_patched = False
    nsa_patched = False

    timing_rows = []

    total_frames = 0

    total_start = time.perf_counter()

    try:

        print()
        print("Applying GM-FLIT patch...")

        patch_flit(
            gamma=args.gamma,
            frame_w=FRAME_W,
            frame_h=FRAME_H,
        )

        flit_patched = True

        print("GM-FLIT: ACTIVE")

        print("Applying NSA-Kalman patch...")

        patch_nsa_kalman()

        nsa_patched = True

        print("NSA-Kalman: ACTIVE")

        print()
        print("=" * 72)
        print("PATCH STATUS")
        print("=" * 72)
        print("GM-FLIT:       ACTIVE")
        print("NSA-Kalman:    ACTIVE")
        print("SAHI:          512")
        print("BoT-SORT:      ACTIVE")
        print("=" * 72)

        # --------------------------------------------------------------
        # Run validation
        # --------------------------------------------------------------

        for seq_name in val_sequences:

            print()
            print(
                f"Processing {seq_name}..."
            )

            n_frames, elapsed = run_tracker_on_sequence(
                seq_name=seq_name,
                det_model=det_model,
                tracker_cfg_path=tracker_cfg_path,
                device=args.device,
                suffix=suffix,
            )

            fps = (
                n_frames / elapsed
                if elapsed > 0
                else 0.0
            )

            total_frames += n_frames

            timing_rows.append(
                {
                    "sequence": seq_name,
                    "tracker": suffix,
                    "n_frames": n_frames,
                    "elapsed_seconds": round(
                        elapsed,
                        3,
                    ),
                    "fps": round(
                        fps,
                        3,
                    ),
                }
            )

            print(
                f"  -> {n_frames} frames | "
                f"{elapsed:.2f}s | "
                f"{fps:.2f} FPS"
            )

    finally:

        # --------------------------------------------------------------
        # ALWAYS restore original tracker implementation.
        # --------------------------------------------------------------

        if nsa_patched:

            print()
            print(
                "Removing NSA-Kalman patch..."
            )

            unpatch_nsa_kalman()

        if flit_patched:

            print(
                "Removing GM-FLIT patch..."
            )

            unpatch_flit()

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    overall_fps = (
        total_frames / total_elapsed
        if total_elapsed > 0
        else 0.0
    )

    timing_rows.append(
        {
            "sequence": "OVERALL",
            "tracker": suffix,
            "n_frames": total_frames,
            "elapsed_seconds": round(
                total_elapsed,
                3,
            ),
            "fps": round(
                overall_fps,
                3,
            ),
        }
    )

    timing_csv = (
        MOT_EVAL_DIR
        / f"timing_{suffix}.csv"
    )

    with timing_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "tracker",
                "n_frames",
                "elapsed_seconds",
                "fps",
            ],
        )

        writer.writeheader()
        writer.writerows(timing_rows)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("EXPERIMENT FINISHED")
    print("=" * 72)
    print(f"Experiment:     {suffix}")
    print(f"Sequences:      {len(val_sequences)}")
    print(f"Frames:         {total_frames}")
    print(
        f"Total time:     {total_elapsed:.2f}s"
    )
    print(
        f"Overall FPS:    {overall_fps:.2f}"
    )
    print(
        f"Timing CSV:     {timing_csv}"
    )
    print(
        f"Output pattern: *_${suffix}.txt"
        .replace("$", "")
    )
    print("=" * 72)


if __name__ == "__main__":
    main()