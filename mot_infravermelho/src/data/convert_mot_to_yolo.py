"""
Converte anotações MOTChallenge (gt.txt) + vídeos .mp4 do 4th Anti-UAV
Challenge (Track 3) para o formato de detecção do YOLO.

Saída: dataset/dataset_yolo/all/images/<SEQ>_<frame>.jpg
       dataset/dataset_yolo/all/labels/<SEQ>_<frame>.txt

Depois rode split_train_val.py para dividir em train/val por sequência.
"""

import csv
import re
from pathlib import Path

import cv2

# ---------- Caminhos base (ajuste só se mover o projeto) ----------
PROJECT_ROOT = Path("/Users/ighorribeiro/PycharmProjects/DroneTrackingMOT")
TRAIN_ROOT = PROJECT_ROOT / "dataset" / "MultiUAV_Train"
LABEL_ROOT = TRAIN_ROOT / "original_label_file"   # contém UAV002/gt/gt.txt etc.
VIDEO_ROOT = TRAIN_ROOT / "TrainVideos"            # contém MultiUAV-002.mp4 etc.

OUT_ROOT = PROJECT_ROOT / "dataset" / "dataset_yolo" / "all"
OUT_IMAGES = OUT_ROOT / "images"
OUT_LABELS = OUT_ROOT / "labels"


def seq_to_video_filename(seq_name: str) -> str:
    """UAV002 -> MultiUAV-002.mp4"""
    num = re.search(r"\d+", seq_name).group()
    return f"MultiUAV-{num}.mp4"


def convert_sequence(video_path: Path, gt_path: Path, seq_name: str) -> int:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[AVISO] não consegui abrir {video_path}, pulando {seq_name}")
        return 0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_annots = {}
    with open(gt_path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            frame, tid = int(row[0]), int(row[1])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            frame_annots.setdefault(frame, []).append((x, y, w, h))

    frame_idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frame_idx += 1  # gt.txt é 1-based

        img_name = f"{seq_name}_{frame_idx:06d}.jpg"
        label_name = f"{seq_name}_{frame_idx:06d}.txt"

        cv2.imwrite(str(OUT_IMAGES / img_name), img)

        lines = []
        for (x, y, w, h) in frame_annots.get(frame_idx, []):
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (OUT_LABELS / label_name).write_text("\n".join(lines))

    cap.release()
    return frame_idx


def main():
    seq_dirs = sorted(p for p in LABEL_ROOT.iterdir() if p.is_dir())
    if not seq_dirs:
        raise RuntimeError(f"Nenhuma sequência encontrada em {LABEL_ROOT}")

    print(f"Encontradas {len(seq_dirs)} sequências: {[p.name for p in seq_dirs]}")

    total_frames = 0
    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        gt_path = seq_dir / "gt" / "gt.txt"
        video_path = VIDEO_ROOT / seq_to_video_filename(seq_name)

        if not gt_path.exists():
            print(f"[AVISO] gt.txt não encontrado para {seq_name}, pulando")
            continue
        if not video_path.exists():
            print(f"[AVISO] vídeo não encontrado para {seq_name} ({video_path}), pulando")
            continue

        n = convert_sequence(video_path, gt_path, seq_name)
        print(f"{seq_name}: {n} frames processados")
        total_frames += n

    print(f"\nTotal geral: {total_frames} frames convertidos em {OUT_ROOT}")


if __name__ == "__main__":
    main()