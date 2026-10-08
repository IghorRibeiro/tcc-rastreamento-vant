"""
Roda detector + tracker (BoT-SORT) sobre as sequências de validação,
aplicando cada uma das 3 estratégias de pré-processamento temporal
(MOG2, EMA, Optical Flow) antes da detecção. O baseline "none" já foi
computado anteriormente (run_tracker_on_val.py) e é reaproveitado.
"""

import csv
import sys
import time
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from common import MOT_EVAL_DIR, VIDEO_ROOT, WEIGHTS, get_val_sequences, seq_to_video_filename  # noqa: E402
from background_masks import METHODS  # noqa: E402

TRACKER_CFG = "botsort.yaml"  # usa o melhor tracker já identificado
DEVICE = "mps"


def run_method(method_name: str, method_fn, model: YOLO):
    val_seqs = get_val_sequences()
    timing_rows = []
    total_start = time.perf_counter()

    for seq_name in val_seqs:
        video_path = VIDEO_ROOT / seq_to_video_filename(seq_name)
        if not video_path.exists():
            print(f"[AVISO] vídeo não encontrado para {seq_name}, pulando")
            continue

        print(f"[{method_name}] Processando {seq_name}...")
        seq_start = time.perf_counter()

        out_lines = []
        n_frames = 0
        for frame_idx, masked_frame in method_fn(str(video_path)):
            n_frames = frame_idx
            results = model.track(
                masked_frame,
                conf=0.25,
                tracker=TRACKER_CFG,
                persist=True,
                device=DEVICE,
                verbose=False,
            )
            r = results[0]
            if r.boxes.id is None:
                continue
            boxes = r.boxes.xywh.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            for (cx, cy, w, h), tid in zip(boxes, ids):
                x = cx - w / 2
                y = cy - h / 2
                out_lines.append(f"{frame_idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")

        seq_elapsed = time.perf_counter() - seq_start
        fps = n_frames / seq_elapsed if seq_elapsed > 0 else 0.0

        out_path = MOT_EVAL_DIR / f"{seq_name}_{method_name}.txt"
        out_path.write_text("\n".join(out_lines))
        print(f"  -> {n_frames} frames em {seq_elapsed:.2f}s ({fps:.2f} FPS)")

        timing_rows.append({
            "sequence": seq_name, "tracker": method_name,
            "n_frames": n_frames, "elapsed_seconds": round(seq_elapsed, 3),
            "fps": round(fps, 3),
        })

    total_elapsed = time.perf_counter() - total_start
    total_frames = sum(r["n_frames"] for r in timing_rows)
    overall_fps = total_frames / total_elapsed if total_elapsed > 0 else 0.0
    timing_rows.append({
        "sequence": "OVERALL", "tracker": method_name,
        "n_frames": total_frames, "elapsed_seconds": round(total_elapsed, 3),
        "fps": round(overall_fps, 3),
    })

    timing_csv = MOT_EVAL_DIR / f"timing_{method_name}.csv"
    with open(timing_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "tracker", "n_frames", "elapsed_seconds", "fps"])
        writer.writeheader()
        writer.writerows(timing_rows)

    print(f"[{method_name}] TOTAL: {total_frames} frames em {total_elapsed:.2f}s ({overall_fps:.2f} FPS médio)\n")


def main():
    MOT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))

    for method_name, method_fn in METHODS.items():
        run_method(method_name, method_fn, model)

    print("Ablação de pré-processamento completa. Rode evaluate_preprocessing_ablation.py.")


if __name__ == "__main__":
    main()