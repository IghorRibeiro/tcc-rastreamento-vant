"""
Divide o dataset convertido (all/) em treino/validação, no nível de
SEQUÊNCIA (não de frame), evitando vazamento temporal entre splits.

Aplica também subamostragem temporal (FRAME_STRIDE) para reduzir
redundância entre frames consecutivos antes de gerar o split final.
"""

import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path("/Users/ighorribeiro/PycharmProjects/DroneTrackingMOT")
YOLO_ROOT = PROJECT_ROOT / "dataset" / "dataset_yolo"
ALL_IMAGES = YOLO_ROOT / "all" / "images"
ALL_LABELS = YOLO_ROOT / "all" / "labels"

VAL_FRACTION = 0.15
SEED = 42
FRAME_STRIDE = 4  # mantém 1 a cada N frames


def subsample_by_stride(label_files, stride):
    """Mantém apenas frames 1, 1+stride, 1+2*stride, ... de cada sequência."""
    kept = []
    for f in label_files:
        frame_num = int(f.stem.rsplit("_", 1)[1])
        if frame_num % stride == 1:
            kept.append(f)
    return kept


def main():
    all_label_files = sorted(ALL_LABELS.glob("*.txt"))
    if not all_label_files:
        raise RuntimeError(f"Nenhum label encontrado em {ALL_LABELS}. Rode convert_mot_to_yolo.py antes.")

    print(f"Total de frames disponíveis (all/): {len(all_label_files)}")

    label_files = subsample_by_stride(all_label_files, FRAME_STRIDE)
    print(f"Após subamostragem (stride={FRAME_STRIDE}): {len(label_files)} frames")

    sequences = sorted({f.stem.rsplit("_", 1)[0] for f in label_files})
    print(f"Sequências encontradas ({len(sequences)}): {sequences}")

    random.seed(SEED)
    random.shuffle(sequences)
    n_val = max(1, round(VAL_FRACTION * len(sequences)))
    val_seqs = set(sequences[:n_val])
    train_seqs = set(sequences[n_val:])

    print(f"Treino ({len(train_seqs)}): {sorted(train_seqs)}")
    print(f"Validação ({len(val_seqs)}): {sorted(val_seqs)}")

    for split_name in ("train", "val"):
        (YOLO_ROOT / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    moved = {"train": 0, "val": 0}
    for label_path in label_files:
        seq_name = label_path.stem.rsplit("_", 1)[0]
        split_name = "val" if seq_name in val_seqs else "train"

        img_path = ALL_IMAGES / f"{label_path.stem}.jpg"
        if not img_path.exists():
            print(f"[AVISO] imagem ausente para {label_path.stem}, pulando")
            continue

        shutil.copy2(img_path, YOLO_ROOT / "images" / split_name / img_path.name)
        shutil.copy2(label_path, YOLO_ROOT / "labels" / split_name / label_path.name)
        moved[split_name] += 1

    print(f"\nCopiados -> treino: {moved['train']} frames | validação: {moved['val']} frames")
    print("(arquivos originais mantidos em dataset_yolo/all/ como backup)")


if __name__ == "__main__":
    main()