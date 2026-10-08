"""
Roda SAHI na detecção + BoT‑SORT (Ultralytics) usando a API oficial:
    OpenCV → frame → SAHI → Results → boxes.cpu().numpy() → tracker.update()

Toda a lógica de tracking é idêntica ao baseline (run_tracker_on_val.py):
- Mesmo tracker BoT‑SORT (YAML)
- Mesmos patches globais (FLIT, Geman‑McClure, NSA‑Kalman)
- Mesmo formato de saída MOTChallenge
- Medição de FPS

Uso:
    python -m src.inference.run_sahi_tracker --config 512 --device mps --conf 0.25 --tracker-cfg botsort.yaml
    python -m src.inference.run_sahi_tracker --config 384 --device cuda:0 --conf 0.25 --tracker-cfg ultra-botsort
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

# Ultralytics tracker (BoT‑SORT) e classe Results oficial
from ultralytics.trackers import BOTSORT
from ultralytics.trackers.bot_sort import STrack
from ultralytics.engine.results import Results


# Imports do projeto (comuns)
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(SRC_DIR))
from common import (
    MOT_EVAL_DIR,
    VIDEO_ROOT,
    get_val_sequences,
    seq_to_video_filename,
)


# ---------- Funções de suporte ----------
def resolve_tracker_cfg(arg: str) -> Path:
    """Resolve apelidos para caminhos de YAML do tracker."""
    from ultralytics import __file__ as ULT_FILE
    ultra_cfg = Path(ULT_FILE).parent / "cfg" / "trackers"

    p = Path(arg)
    if p.exists():
        return p

    candidate = ultra_cfg / arg
    if candidate.exists():
        return candidate

    if arg == "ultra-botsort":
        default = ultra_cfg / "botsort.yaml"
        if default.exists():
            return default

    raise FileNotFoundError(f"Arquivo de tracker não encontrado: {arg}")


def load_tracker_config(cfg_path: Path) -> SimpleNamespace:
    """Carrega o YAML do tracker e retorna um SimpleNamespace."""
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)


def build_sahi_detections(
    frame: np.ndarray,
    det_model: AutoDetectionModel,
    slice_h: int,
    slice_w: int,
    overlap: float,
    conf_thresh: float,
) -> np.ndarray:
    """
    Executa SAHI no frame e retorna detecções como array (N,6):
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

    H, W = frame.shape[:2]
    dets = []
    for obj in result.object_prediction_list:
        score = float(obj.score.value)
        if score < conf_thresh:
            continue
        x1, y1, x2, y2 = obj.bbox.to_xyxy()
        x1 = max(0.0, min(W - 1.0, x1))
        y1 = max(0.0, min(H - 1.0, y1))
        x2 = max(0.0, min(W - 1.0, x2))
        y2 = max(0.0, min(H - 1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cls_id = obj.category.id
        dets.append([x1, y1, x2, y2, score, float(cls_id)])

    if not dets:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(dets, dtype=np.float32)


def run_tracker_on_sequence(
    seq_name: str,
    det_model: AutoDetectionModel,
    tracker_cfg_path: Path,
    slice_h: int,
    slice_w: int,
    overlap: float,
    conf: float,
    device: str,
    suffix: str,
) -> tuple[int, float]:
    """Processa uma sequência inteira com SAHI + BoT‑SORT usando API oficial."""
    video_path = VIDEO_ROOT / seq_to_video_filename(seq_name)
    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    tracker_args = load_tracker_config(tracker_cfg_path)
    tracker = BOTSORT(args=tracker_args)

    names = {0: "drone"}

    out_lines: List[str] = []
    frame_idx = 0
    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1. SAHI no frame → array (N,6)
        dets = build_sahi_detections(
            frame, det_model, slice_h, slice_w, overlap, conf
        )

        # 2. Construir objeto Results oficial com boxes
        box_tensor = torch.from_numpy(dets).to(device=device, dtype=torch.float32)
        result = Results(
            orig_img=frame,
            path="frame.jpg",
            names=names,
            boxes=box_tensor,
        )

        # 3. Extrair o objeto Boxes exatamente como o callback oficial faz
        #    result.boxes.cpu().numpy() retorna um objeto Boxes (não um ndarray)
        det = result.boxes.cpu().numpy()

        # 4. Atualizar tracker passando o objeto Boxes e a imagem original
        tracks = tracker.update(det, frame)

        if len(tracks):

            for t in tracks:
                x1, y1, x2, y2 = t[:4]

                track_id = int(t[4])

                w = x2 - x1
                h = y2 - y1

                out_lines.append(
                    f"{frame_idx},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
                )

    cap.release()
    elapsed = time.perf_counter() - start_time

    out_path = MOT_EVAL_DIR / f"{seq_name}_{suffix}.txt"
    out_path.write_text("\n".join(out_lines))

    return frame_idx, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="SAHI -> BoT-SORT (validação)"
    )

    parser.add_argument(
        "--config",
        choices=["320", "384", "448", "512", "576"],
        required=True,
        help="Tamanho do slice SAHI"
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="Caminho do modelo YOLO (.pt)"
    )

    parser.add_argument(
        "--exp-name",
        required=True,
        help="Nome do experimento (baseline, hardmining, etc.)"
    )

    parser.add_argument(
        "--device",
        default="mps",
        help="mps | cpu | cuda:0"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--tracker-cfg",
        default="botsort.yaml",
        help="Arquivo YAML do tracker"
    )

    parser.add_argument(
        "--only-seq",
        nargs="*",
        default=None
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Configuração do SAHI
    # --------------------------------------------------

    CONFIGS = {
        "320": (320, 0.30),
        "384": (384, 0.25),
        "448": (448, 0.20),
        "512": (512, 0.20),
        "576": (576, 0.20),
    }

    slice_size, overlap = CONFIGS[args.config]

    slice_h = slice_size
    slice_w = slice_size

    suffix = f"{args.exp_name}_sahi{args.config}"

    # --------------------------------------------------

    MOT_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Modelo :", args.weights)
    print("Experimento :", args.exp_name)
    print("SAHI :", args.config)
    print("=" * 70)

    det_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
    )

    tracker_cfg_path = resolve_tracker_cfg(args.tracker_cfg)

    val_seqs = get_val_sequences()

    if args.only_seq:
        keep = set(args.only_seq)
        val_seqs = [s for s in val_seqs if s in keep]

    timing_rows = []

    total_frames = 0
    total_start = time.perf_counter()

    for seq_name in val_seqs:

        print(f"\nProcessando {seq_name}...")

        n_frames, elapsed = run_tracker_on_sequence(
            seq_name=seq_name,
            det_model=det_model,
            tracker_cfg_path=tracker_cfg_path,
            slice_h=slice_h,
            slice_w=slice_w,
            overlap=overlap,
            conf=args.conf,
            device=args.device,
            suffix=suffix,
        )

        fps = n_frames / elapsed if elapsed > 0 else 0

        total_frames += n_frames

        timing_rows.append(
            {
                "sequence": seq_name,
                "tracker": suffix,
                "n_frames": n_frames,
                "elapsed_seconds": round(elapsed, 3),
                "fps": round(fps, 3),
            }
        )

        print(
            f" -> {n_frames} frames | {elapsed:.2f}s | {fps:.2f} FPS"
        )

    total_elapsed = time.perf_counter() - total_start

    overall_fps = (
        total_frames / total_elapsed
        if total_elapsed > 0
        else 0
    )

    timing_rows.append(
        {
            "sequence": "OVERALL",
            "tracker": suffix,
            "n_frames": total_frames,
            "elapsed_seconds": round(total_elapsed, 3),
            "fps": round(overall_fps, 3),
        }
    )

    timing_csv = MOT_EVAL_DIR / f"timing_{suffix}.csv"

    with open(timing_csv, "w", newline="") as f:

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

    print("\n" + "=" * 70)
    print("Experimento finalizado.")
    print(f"Resultados : {suffix}")
    print(f"Timing     : {timing_csv}")
    print(f"FPS médio  : {overall_fps:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()