"""Run the hybrid architecture extended to multiple targets on Track 3 videos and save MOTChallenge trajectories.

Needs a checkout of DroneTrackingMOT at tag v1.0-lafusion2026 (etapa2_enxames/README.md) for the Geman-McClure
association cost, the YOLO26n weights and, optionally, the tiled-inference helper:

    python etapa3_predicao/run_hybrid_mot.py --mot-repo <DroneTrackingMOT> --weights best.pt \\
        --videos <data-root>/MultiUAV_Train/TrainVideos --sequences UAV009 --out-dir runs/hybrid_csrt_n15

Per sequence it writes ``<out-dir>/traj/<seq>.txt`` (frame,id,x,y,w,h,conf,-1,-1,-1) and ``<out-dir>/frames/<seq>.csv``
with the processing time of every frame (detection + tracking, without video decoding), measured after a warm-up of the
detector. Settings come from configs/hybrid_mot.yaml.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trajpred.hybrid_mot import HybridMOT, detections_from_sahi, detections_from_yolo, make_local_tracker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def video_file(videos: Path, seq: str) -> Path:
    return videos / f"MultiUAV-{re.search(r'\d+', seq).group()}.mp4"


def iter_frames(path: Path, max_frames: int | None):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
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


def build_tracker(cfg: dict, mot_repo: Path, weights: Path, device: str):
    from ultralytics import YOLO
    from ultralytics.trackers.utils import matching

    sys.path.insert(0, str(mot_repo))                       # DroneTrackingMOT provides the ``src`` package
    from src.tracking.gm_cost import make_gm_get_dists

    get_dists = make_gm_get_dists(gamma=float(cfg["association"]["gamma"]), frame_w=int(cfg["frame"]["width"]),
                                  frame_h=int(cfg["frame"]["height"]))
    conf = float(cfg["det_conf"])
    if cfg["sahi"]["enabled"]:
        from sahi import AutoDetectionModel

        from src.tracking.run_sahi_tracker import build_sahi_detections

        model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
                                                   confidence_threshold=conf, device=device)
        size, overlap = int(cfg["sahi"]["slice"]), float(cfg["sahi"]["overlap"])
        detect = lambda frame: detections_from_sahi(build_sahi_detections, model, frame, size, overlap, conf)  # noqa: E731
    else:
        yolo = YOLO(str(weights))
        detect = lambda frame: detections_from_yolo(yolo, frame, device, conf)  # noqa: E731
    make_local_tracker(str(cfg["local_tracker"]))          # fail early if OpenCV lacks the tracker
    return lambda: HybridMOT(cfg, detect, make_local_tracker, get_dists, matching.linear_assignment), detect


def run_sequence(cfg, new_tracker, detect, video: Path, out_dir: Path, seq: str, max_frames):
    height, width = int(cfg["frame"]["height"]), int(cfg["frame"]["width"])
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(int(cfg.get("warmup_calls", 2))):
        detect(blank)
    tracker = new_tracker()
    lines, rows = [], []
    for index, frame in iter_frames(video, max_frames):
        start = time.perf_counter()
        outputs, stats, _anchors = tracker.step(index, frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        for tid, x, y, w, h, conf in outputs:
            lines.append(f"{index},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.6f},-1,-1,-1")
        rows.append({"frame": index, "time_ms": elapsed_ms, **stats})
    (out_dir / "traj").mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    (out_dir / "traj" / f"{seq}.txt").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "frames" / f"{seq}.csv", index=False)
    mean_ms = float(np.mean([r["time_ms"] for r in rows])) if rows else float("nan")
    print(f"  {seq}: {len(rows)} frames, {mean_ms:.1f} ms/frame ({1000.0 / mean_ms:.1f} FPS)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mot-repo", type=Path, required=True, help="DroneTrackingMOT checkout (tag v1.0-lafusion2026)")
    parser.add_argument("--weights", type=Path, required=True, help="YOLO26n weights (best.pt)")
    parser.add_argument("--videos", type=Path, required=True, help="folder with MultiUAV-NNN.mp4")
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "hybrid_mot.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after this many frames (smoke test)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    new_tracker, detect = build_tracker(cfg, args.mot_repo, args.weights, resolve_device(args.device))
    for seq in args.sequences:
        run_sequence(cfg, new_tracker, detect, video_file(args.videos, seq), args.out_dir, seq, args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
