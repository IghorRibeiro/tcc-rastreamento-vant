"""Referencia visivel para H4: o hibrido do SIGE no DUT Anti-UAV, instrumentado.

run_hibrido_instrumented() e copia de publicar_github/.../src/04_rastreadores.py::
run_hibrido com UMA mudanca: alem das predicoes, registra os eventos de
ancoragem (inicializacao e reancoragem) e o evento de cada quadro, que o
original nao grava. detect_best() e make_local_tracker() sao os do proprio
SIGE, importados sem alteracao. O arquivo de predicoes sai no formato do SIGE
(frame,x,y,w,h,conf,tempo_ms), avaliavel pelo 05_avaliar.py original.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import write_csv


def run_hibrido_instrumented(sige04, uc, detector, image_paths, device: str, init_conf: float,
                             switch_conf: float, period: int, local_tracker: str,
                             pred_path: Path, anchors_path: Path, frames_path: Path) -> dict:
    import cv2

    rows = []
    times = []
    anchors = []
    frame_log = []
    detector_calls = 0
    tracker = None
    last_conf = 0.0

    for index, path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        start = time.perf_counter()
        current = None
        event = "none"
        detected = 0

        if tracker is None:
            detector_calls += 1
            detected = 1
            current = sige04.detect_best(detector, frame, device, init_conf)
            if current is not None:
                x, y, w, h, last_conf = current
                tracker = sige04.make_local_tracker(local_tracker)
                tracker.init(frame, (x, y, w, h))
                event = "init"
        else:
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = (int(round(v)) for v in box)
                w, h = max(1, w), max(1, h)
                current = (x, y, w, h, last_conf)
                event = "local"
                if index % period == 0:
                    detector_calls += 1
                    detected = 1
                    candidate = sige04.detect_best(detector, frame, device, switch_conf)
                    if candidate is not None and candidate[4] >= switch_conf:
                        x, y, w, h, last_conf = candidate
                        tracker = sige04.make_local_tracker(local_tracker)
                        tracker.init(frame, (x, y, w, h))
                        current = candidate
                        event = "reanchor"
            else:
                tracker = None
                event = "fail"

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)
        number = uc.frame_number(path)
        if current is not None:
            x, y, w, h, conf = current
            rows.append((number, x, y, w, h, f"{conf:.6f}", f"{elapsed_ms:.3f}"))
            if event in ("init", "reanchor"):
                anchors.append((1, number, x, y, w, h, float(conf), event))
        frame_log.append({"frame": number, "index": index, "time_ms": elapsed_ms, "event": event,
                          "detect": detected})

    uc.write_predictions(pred_path, rows)
    write_csv(pd.DataFrame(anchors, columns=["id", "frame", "x", "y", "w", "h", "conf", "kind"]), anchors_path)
    write_csv(pd.DataFrame(frame_log), frames_path)
    mean_ms = float(np.mean(times)) if times else float("nan")
    return {"frames": len(image_paths), "predictions": len(rows), "detector_calls": detector_calls,
            "mean_time_ms": mean_ms, "fps": 1000.0 / mean_ms if mean_ms > 0 else float("nan")}


def sige_traj_to_mot_df(pred_path: Path) -> pd.DataFrame:
    """Predicoes do SIGE (frame,x,y,w,h,...) como DataFrame no formato MOT (id = 1)."""
    rows = []
    text = Path(pred_path).read_text(encoding="utf-8") if Path(pred_path).exists() else ""
    for raw in text.splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 5:
            continue
        rows.append((int(float(parts[0])), 1, float(parts[1]), float(parts[2]), float(parts[3]),
                     float(parts[4])))
    return pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h"])


def find_dut_ground_truth(cfg: dict, uc):
    """Fonte do GT do DUT, na ordem de preferencia do SIGE (05_avaliar.py):

    1. ('oficial', {video: videoNN_gt.txt}) se houver GT oficial em dut.gt_dir (busca recursiva);
    2. ('rotulos_yolo', pasta) com os rotulos YOLO do PASSO 2 (videoNN_00001.txt), que o
       05_avaliar.py aceita como alternativa (--labels-dir); procura em dut.gt_dir, em
       dut.gt_dir/labels/test e em <images_dir>/../../labels/<nome de images_dir>;
    3. (None, motivo).
    """
    from .common import rpath

    gt_dir = rpath(cfg["dut"].get("gt_dir"))
    images = rpath(cfg["dut"].get("images_dir"))
    if gt_dir is not None and gt_dir.exists():
        files = uc.official_gt_files(gt_dir)
        if files:
            return "oficial", files
    candidates = []
    if gt_dir is not None:
        candidates += [gt_dir, gt_dir / "labels" / "test"]
    if images is not None:
        candidates.append(images.parent.parent / "labels" / images.name)
    for candidate in candidates:
        if candidate.is_dir() and next(candidate.glob("video*_*.txt"), None) is not None:
            return "rotulos_yolo", candidate
    return None, (f"nem GT oficial (videoNN_gt.txt) nem rotulos YOLO (videoNN_00001.txt) em "
                  f"{', '.join(str(c) for c in candidates) or 'dut.gt_dir vazio'}")


def dut_configured(cfg: dict) -> tuple[bool, str]:
    """(habilitado, motivo) para a referencia visivel de H4."""
    from .common import rpath

    images = rpath(cfg["dut"].get("images_dir"))
    gt_dir = rpath(cfg["dut"].get("gt_dir"))
    if images is None or gt_dir is None:
        return False, "dut.images_dir e/ou dut.gt_dir vazios no config.yaml"
    if not images.is_dir():
        return False, f"dut.images_dir nao existe: {images}"
    if not gt_dir.exists():
        return False, f"dut.gt_dir nao existe: {gt_dir}"
    return True, "ok"
