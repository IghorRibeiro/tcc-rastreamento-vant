"""
Roda detector + tracker nas sequências de VALIDAÇÃO (held-out, vídeo
completo, sem subamostragem) e salva os resultados em formato MOT
padrão, prontos para avaliação quantitativa contra o gt.txt original.

Também registra o tempo de processamento por sequência (e FPS
resultante), salvo em runs/mot_eval/timing_<tracker>.csv — usado para
a análise de trade-off velocidade x qualidade no artigo.
"""

import csv
import random
import re
import time
import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = PROJECT_ROOT / "dataset" / "MultiUAV_Train"
LABEL_ROOT = TRAIN_ROOT / "original_label_file"
VIDEO_ROOT = TRAIN_ROOT / "TrainVideos"

OUT_DIR = PROJECT_ROOT / "runs" / "mot_eval"
TRACKER_CFG = "botsort.yaml"

VAL_FRACTION = 0.15
SEED = 42


def seq_to_video_filename(seq_name: str) -> str:
    num = re.search(r"\d+", seq_name).group()
    return f"MultiUAV-{num}.mp4"


def get_val_sequences():
    """Recalcula a mesma lista de sequências de validação usada no
    split_train_val.py (mesma seed => mesmo resultado, sem depender
    de arquivos já processados)."""
    all_seq_dirs = sorted(p.name for p in LABEL_ROOT.iterdir() if p.is_dir())
    random.seed(SEED)
    shuffled = all_seq_dirs.copy()
    random.shuffle(shuffled)
    n_val = max(1, round(VAL_FRACTION * len(shuffled)))
    return sorted(shuffled[:n_val])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        default=str(PROJECT_ROOT / "weights" / "best.pt"),
        help="Modelo YOLO (.pt)"
    )

    parser.add_argument(
        "--exp-name",
        default="baseline",
        help="Nome do experimento"
    )

    parser.add_argument(
        "--device",
        default="mps"
    )

    parser.add_argument(
        "--tracker",
        default=TRACKER_CFG
    )

    args = parser.parse_args()
    val_seqs = get_val_sequences()
    print(f"Sequências de validação (held-out, gt completo): {val_seqs}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    tracker_name = args.exp_name
    timing_rows = []
    total_start = time.perf_counter()

    for seq_name in val_seqs:
        video_path = VIDEO_ROOT / seq_to_video_filename(seq_name)
        if not video_path.exists():
            print(f"[AVISO] vídeo não encontrado para {seq_name}, pulando")
            continue

        print(f"\nRastreando {seq_name} ({video_path.name})...")
        seq_start = time.perf_counter()

        results = model.track(
            source=str(video_path),
            conf=0.25,
            tracker=args.tracker,
            persist=True,
            stream=True,   # processa frame a frame, sem carregar tudo na memória
            device=args.device,
            verbose=False,
        )

        out_lines = []
        n_frames = 0
        for frame_idx, r in enumerate(results, start=1):  # 1-based, igual ao gt.txt
            n_frames = frame_idx
            if r.boxes.id is None:
                continue
            boxes = r.boxes.xywh.cpu().numpy()   # cx, cy, w, h
            ids = r.boxes.id.cpu().numpy().astype(int)
            for (cx, cy, w, h), tid in zip(boxes, ids):
                x = cx - w / 2
                y = cy - h / 2
                # formato MOTChallenge: frame,id,x,y,w,h,conf,-1,-1,-1
                out_lines.append(f"{frame_idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")

        seq_elapsed = time.perf_counter() - seq_start
        fps = n_frames / seq_elapsed if seq_elapsed > 0 else 0.0

        out_path = OUT_DIR / f"{seq_name}_{tracker_name}.txt"
        out_path.write_text("\n".join(out_lines))
        print(f"  -> {len(out_lines)} detecções salvas em {out_path}")
        print(f"  -> {n_frames} frames em {seq_elapsed:.2f}s ({fps:.2f} FPS)")

        timing_rows.append({
            "sequence": seq_name,
            "tracker": tracker_name,
            "n_frames": n_frames,
            "elapsed_seconds": round(seq_elapsed, 3),
            "fps": round(fps, 3),
        })

    total_elapsed = time.perf_counter() - total_start
    total_frames = sum(r["n_frames"] for r in timing_rows)
    overall_fps = total_frames / total_elapsed if total_elapsed > 0 else 0.0

    timing_rows.append({
        "sequence": "OVERALL",
        "tracker": tracker_name,
        "n_frames": total_frames,
        "elapsed_seconds": round(total_elapsed, 3),
        "fps": round(overall_fps, 3),
    })

    timing_csv = OUT_DIR / f"timing_{tracker_name}.csv"
    with open(timing_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "tracker", "n_frames", "elapsed_seconds", "fps"])
        writer.writeheader()
        writer.writerows(timing_rows)

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_frames} frames em {total_elapsed:.2f}s ({overall_fps:.2f} FPS médio)")
    print(f"Timing salvo em: {timing_csv}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()