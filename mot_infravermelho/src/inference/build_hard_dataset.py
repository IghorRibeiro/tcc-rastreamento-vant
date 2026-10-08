"""
build_hard_dataset.py

Cria um dataset YOLO contendo apenas as sequências mais difíceis
identificadas pelo evaluate_train.py.

Entrada:
runs/train_predictions/hard_examples.txt

Saída:

dataset/
    dataset_yolo_hard/
        images/
            train/
        labels/
            train/

"""

from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_IMAGES = PROJECT_ROOT / "dataset" / "dataset_yolo" / "images" / "train"
SOURCE_LABELS = PROJECT_ROOT / "dataset" / "dataset_yolo" / "labels" / "train"

HARD_LIST = PROJECT_ROOT / "runs" / "train_predictions" / "hard_examples.txt"

DEST_IMAGES = PROJECT_ROOT / "dataset" / "dataset_yolo_hard" / "images" / "train"
DEST_LABELS = PROJECT_ROOT / "dataset" / "dataset_yolo_hard" / "labels" / "train"

DEST_IMAGES.mkdir(parents=True, exist_ok=True)
DEST_LABELS.mkdir(parents=True, exist_ok=True)


def main():

    with open(HARD_LIST) as f:
        sequences = [x.strip() for x in f if x.strip()]

    copied = 0

    print("=" * 70)
    print(f"Copiando {len(sequences)} sequências...")
    print("=" * 70)

    for seq in sequences:

        img_files = sorted(SOURCE_IMAGES.glob(f"{seq}_*.jpg"))

        print(f"{seq:10s} -> {len(img_files)} imagens")

        for img in img_files:

            lbl = SOURCE_LABELS / (img.stem + ".txt")

            shutil.copy2(img, DEST_IMAGES / img.name)

            if lbl.exists():
                shutil.copy2(lbl, DEST_LABELS / lbl.name)

            copied += 1

    print("\n" + "=" * 70)
    print("Dataset Hard Mining criado.")
    print(f"Total de imagens : {copied}")
    print(f"Destino imagens  : {DEST_IMAGES}")
    print(f"Destino labels   : {DEST_LABELS}")
    print("=" * 70)


if __name__ == "__main__":
    main()