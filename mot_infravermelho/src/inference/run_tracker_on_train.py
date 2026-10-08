"""
Roda detector + tracker nas sequências de treino (held-out, vídeo
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
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = PROJECT_ROOT / "dataset" / "MultiUAV_Train"
LABEL_ROOT = TRAIN_ROOT / "original_label_file"
VIDEO_ROOT = TRAIN_ROOT / "TrainVideos"
WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"

OUT_DIR = PROJECT_ROOT / "runs" / "train_predictions"
TRACKER_CFG = "botsort.yaml"  # troque para "botsort.yaml" conforme necessário

VAL_FRACTION = 0.15
SEED = 42


def get_train_sequences():
    """
    Reproduz EXATAMENTE o mesmo split usado no projeto.

    Validação:
        shuffled[:n_val]

    Treino:
        shuffled[n_val:]

    Dessa forma, nenhuma sequência de validação entra no hard mining.
    """
    all_seq_dirs = sorted(
        p.name for p in LABEL_ROOT.iterdir() if p.is_dir()
    )

    random.seed(SEED)

    shuffled = all_seq_dirs.copy()
    random.shuffle(shuffled)

    n_val = max(
        1,
        round(VAL_FRACTION * len(shuffled))
    )

    train_seqs = sorted(shuffled[n_val:])

    return train_seqs

def seq_to_video_filename(seq_name: str) -> str:
    num = re.search(r"\d+", seq_name).group()
    return f"MultiUAV-{num}.mp4"

def main():
    train_seqs = get_train_sequences()

    print(f"Sequências de TREINO: {len(train_seqs)}")
    print(train_seqs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS))

    tracker_name = TRACKER_CFG.split(".")[0]

    timing_rows = []
    total_start = time.perf_counter()

    for seq_name in train_seqs:

        video_path = VIDEO_ROOT / seq_to_video_filename(seq_name)

        if not video_path.exists():
            print(f"[AVISO] vídeo não encontrado para {seq_name}")
            continue

        print(f"\nRastreando {seq_name} ({video_path.name})...")

        seq_start = time.perf_counter()

        results = model.track(
            source=str(video_path),
            conf=0.25,
            tracker=TRACKER_CFG,
            persist=True,
            stream=True,
            device="mps",
            verbose=False,
        )

        out_lines = []
        n_frames = 0

        for frame_idx, r in enumerate(results, start=1):
            n_frames = frame_idx

            if r.boxes.id is None:
                continue

            boxes = r.boxes.xywh.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)

            for (cx, cy, w, h), tid in zip(boxes, ids):
                x = cx - w / 2
                y = cy - h / 2

                out_lines.append(
                    f"{frame_idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
                )

        seq_elapsed = time.perf_counter() - seq_start
        fps = n_frames / seq_elapsed if seq_elapsed else 0

        out_path = OUT_DIR / f"{seq_name}.txt"
        out_path.write_text("\n".join(out_lines))

        print(f"  -> {len(out_lines)} tracks")
        print(f"  -> {fps:.2f} FPS")

        timing_rows.append(
            {
                "sequence": seq_name,
                "tracker": tracker_name,
                "n_frames": n_frames,
                "elapsed_seconds": round(seq_elapsed, 3),
                "fps": round(fps, 3),
            }
        )

    total_elapsed = time.perf_counter() - total_start
    total_frames = sum(r["n_frames"] for r in timing_rows)

    overall_fps = total_frames / total_elapsed

    timing_rows.append(
        {
            "sequence": "OVERALL",
            "tracker": tracker_name,
            "n_frames": total_frames,
            "elapsed_seconds": round(total_elapsed, 3),
            "fps": round(overall_fps, 3),
        }
    )

    timing_csv = OUT_DIR / f"timing_{tracker_name}.csv"

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

    print("\nFinalizado.")


if __name__ == "__main__":
    main()