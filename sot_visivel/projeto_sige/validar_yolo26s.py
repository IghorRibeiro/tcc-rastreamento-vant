from __future__ import annotations

import csv
import time
from pathlib import Path

from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent
DATASET = PROJECT / "datasets" / "anti_uav_tracking_test_final" / "dataset.yaml"
MODEL = PROJECT / "pesos_treinados" / "yolo26s_dut_best.pt"
OUTPUT = PROJECT / "scripts" / "reviewer_validation" / "yolo26s_tabela1"
IMAGE_DIR = PROJECT / "datasets" / "anti_uav_tracking_test_final" / "images" / "test"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    n_images = len(list(IMAGE_DIR.glob("*.jpg")))
    model = YOLO(str(MODEL))
    start = time.perf_counter()
    metrics = model.val(data=str(DATASET), split="test", device="mps", batch=1, workers=0, plots=False, verbose=False, project=str(OUTPUT), name="val", exist_ok=True)
    elapsed = time.perf_counter() - start
    box = metrics.box
    speed = metrics.speed if isinstance(metrics.speed, dict) else {}
    row = {
        "model": "YOLO26s",
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "validation_seconds_wall": elapsed,
        "validation_images": n_images,
        "validation_images_per_second_wall": n_images / elapsed if elapsed > 0 else float("nan"),
        "preprocess_ms_per_image": float(speed.get("preprocess", float("nan"))),
        "inference_ms_per_image": float(speed.get("inference", float("nan"))),
        "postprocess_ms_per_image": float(speed.get("postprocess", float("nan"))),
    }
    with (OUTPUT / "yolo26s_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    for key, value in row.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
