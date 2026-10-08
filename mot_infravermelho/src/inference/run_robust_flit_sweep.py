import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm
import pandas as pd
from ultralytics import YOLO

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(SRC_DIR / "postprocessing"))
from common import LABEL_ROOT, MOT_EVAL_DIR, VIDEO_ROOT, WEIGHTS, get_val_sequences, seq_to_video_filename, load_mot_file  # noqa: E402
from robust_flit_association import patch_flit, unpatch_flit
from nsa_kalman_patch import patch_nsa_kalman, unpatch_nsa_kalman  # noqa: E402

TRACKER_CFG = "botsort.yaml"
DEVICE = "mps"
FRAME_W, FRAME_H = 640, 512

GAMMAS = [0.15, 0.25, 0.35, 0.45, 0.55]
NSA_OPTIONS = [True]


def run_and_evaluate(gamma: float, use_nsa: bool, model: YOLO):
    patch_flit(gamma=gamma, frame_w=FRAME_W, frame_h=FRAME_H)
    if use_nsa:
        patch_nsa_kalman()

    try:
        val_seqs = get_val_sequences()
        acc_list, seq_names = [], []
        total_start = time.perf_counter()
        total_frames = 0

        for seq_name in val_seqs:
            video_path = VIDEO_ROOT / seq_to_video_filename(seq_name)
            if not video_path.exists():
                continue

            results_stream = model.track(
                source=str(video_path), conf=0.25, tracker=TRACKER_CFG,
                persist=True, stream=True, device=DEVICE, verbose=False,
            )

            rows = []
            n_frames = 0
            for frame_idx, r in enumerate(results_stream, start=1):
                n_frames = frame_idx
                if r.boxes.id is None:
                    continue
                boxes = r.boxes.xywh.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)
                for (cx, cy, w, h), tid in zip(boxes, ids):
                    rows.append([frame_idx, tid, cx - w / 2, cy - h / 2, w, h])
            total_frames += n_frames

            pred_df = pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h"])
            gt_path = LABEL_ROOT / seq_name / "gt" / "gt.txt"
            if not gt_path.exists():
                continue
            gt_df = load_mot_file(gt_path)

            acc = mm.MOTAccumulator(auto_id=True)
            all_frames = sorted(set(gt_df["frame"]) | set(pred_df["frame"]))
            for frame in all_frames:
                gt_frame = gt_df[gt_df["frame"] == frame]
                pred_frame = pred_df[pred_df["frame"] == frame]
                dist_matrix = mm.distances.iou_matrix(
                    gt_frame[["x", "y", "w", "h"]].values,
                    pred_frame[["x", "y", "w", "h"]].values,
                    max_iou=0.5,
                )
                acc.update(gt_frame["id"].values, pred_frame["id"].values, dist_matrix)

            acc_list.append(acc)
            seq_names.append(seq_name)

        total_elapsed = time.perf_counter() - total_start
    finally:
        unpatch_flit()
        if use_nsa:
            unpatch_nsa_kalman()

    mh = mm.metrics.create()
    summary = mh.compute_many(
        acc_list,
        metrics=["mota", "motp", "idf1", "num_switches", "num_false_positives", "num_misses", "num_objects"],
        names=seq_names,
        generate_overall=True,
    )
    overall = summary.loc["OVERALL"]
    fps = total_frames / total_elapsed if total_elapsed > 0 else 0.0

    return {
        "gamma": gamma,
        "nsa_kalman": use_nsa,
        "MOTA(%)": round(overall["mota"] * 100, 1),
        "IDF1(%)": round(overall["idf1"] * 100, 1),
        "IDs": int(overall["num_switches"]),
        "FP_rate": round(overall["num_false_positives"] / overall["num_objects"], 4),
        "FN_rate": round(overall["num_misses"] / overall["num_objects"], 4),
        "FPS": round(fps, 2),
    }


def main():
    model = YOLO(str(WEIGHTS))
    results = []

    print("Baseline de referência: MOTA=75.9  IDF1=71.2  IDs=2327  FPS=46.46 (sem patch)\n")

    for gamma, use_nsa in product(GAMMAS, NSA_OPTIONS):
        print(f"Rodando gamma={gamma}, nsa_kalman={use_nsa}...")
        metrics = run_and_evaluate(gamma, use_nsa, model)
        results.append(metrics)
        print(f"  -> MOTA={metrics['MOTA(%)']}  IDF1={metrics['IDF1(%)']}  "
              f"IDs={metrics['IDs']}  FPS={metrics['FPS']}\n")

    df = pd.DataFrame(results).sort_values("IDF1(%)", ascending=False)
    print("=" * 80)
    print("Resultado -- FLIT (L2-IoU) x NSA-Kalman, ordenado por IDF1")
    print("=" * 80)
    print(df.to_string(index=False))

    out_csv = MOT_EVAL_DIR / "flit_nsa_sweep_summary.csv"
    df.to_csv(out_csv, index=False)
    out_md = MOT_EVAL_DIR / "flit_nsa_sweep_summary.md"
    out_md.write_text(df.to_markdown(index=False))
    print(f"\nSalvo em: {out_csv} e {out_md}")


if __name__ == "__main__":
    main()