"""
Roda detector + tracker nas 30 sequências de validação para os 6
trackers nativos do Ultralytics (bytetrack, botsort, tracktrack,
fasttrack, ocsort, deepocsort), usando as configs padrão de cada um.
Gera um leaderboard único (MOTA/IDF1/IDs/FPS).
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm
import pandas as pd
from ultralytics import YOLO

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from common import LABEL_ROOT, MOT_EVAL_DIR, VIDEO_ROOT, WEIGHTS, get_val_sequences, seq_to_video_filename, load_mot_file  # noqa: E402

DEVICE = "mps"
TRACKERS = ["bytetrack", "botsort", "tracktrack", "fasttrack", "ocsort", "deepocsort"]


def run_tracker(tracker_name: str, model: YOLO):
    # Força reinicialização do tracker anexado ao model -- essencial
    # ao trocar de tipo de tracker entre chamadas; sem isso, persist=True
    # reaproveita silenciosamente o tracker da chamada anterior.
    if hasattr(model, "predictor") and model.predictor is not None:
        if hasattr(model.predictor, "trackers"):
            del model.predictor.trackers

    tracker_cfg = f"{tracker_name}.yaml"
    val_seqs = get_val_sequences()
    timing_rows = []
    total_start = time.perf_counter()

    for seq_name in val_seqs:
        video_path = VIDEO_ROOT / seq_to_video_filename(seq_name)
        if not video_path.exists():
            continue

        if hasattr(model, "predictor") and model.predictor is not None:
            if hasattr(model.predictor, "trackers"):
                del model.predictor.trackers

        try:
            results = model.track(
                source=str(video_path), conf=0.25, tracker=tracker_cfg,
                persist=True, stream=True, device=DEVICE, verbose=False,
            )
        except Exception as e:
            print(f"  [ERRO] {tracker_name} falhou em {seq_name}: {e}")
            continue

        seq_start = time.perf_counter()
        out_lines = []
        n_frames = 0
        try:
            for frame_idx, r in enumerate(results, start=1):
                n_frames = frame_idx
                if r.boxes.id is None:
                    continue
                boxes = r.boxes.xywh.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)
                for (cx, cy, w, h), tid in zip(boxes, ids):
                    x, y = cx - w / 2, cy - h / 2
                    out_lines.append(f"{frame_idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
        except Exception as e:
            print(f"  [ERRO] {tracker_name} quebrou durante {seq_name}: {e}")
            continue

        seq_elapsed = time.perf_counter() - seq_start
        fps = n_frames / seq_elapsed if seq_elapsed > 0 else 0.0

        out_path = MOT_EVAL_DIR / f"{seq_name}_{tracker_name}.txt"
        out_path.write_text("\n".join(out_lines))
        timing_rows.append({"sequence": seq_name, "n_frames": n_frames, "elapsed_seconds": seq_elapsed, "fps": fps})
        print(f"  [{tracker_name}] {seq_name}: {n_frames} frames, {fps:.1f} FPS")

    total_elapsed = time.perf_counter() - total_start
    total_frames = sum(r["n_frames"] for r in timing_rows)
    overall_fps = total_frames / total_elapsed if total_elapsed > 0 else 0.0

    timing_csv = MOT_EVAL_DIR / f"timing_{tracker_name}.csv"
    with open(timing_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "n_frames", "elapsed_seconds", "fps"])
        writer.writeheader()
        writer.writerows(timing_rows + [{"sequence": "OVERALL", "n_frames": total_frames,
                                          "elapsed_seconds": total_elapsed, "fps": overall_fps}])
    return len(timing_rows) > 0


def evaluate_tracker(tracker_name: str):
    result_files = sorted(MOT_EVAL_DIR.glob(f"*_{tracker_name}.txt"))
    if not result_files:
        return None

    acc_list, seq_names = [], []
    for result_path in result_files:
        seq_name = result_path.stem.replace(f"_{tracker_name}", "")
        gt_path = LABEL_ROOT / seq_name / "gt" / "gt.txt"
        if not gt_path.exists():
            continue
        gt_df = load_mot_file(gt_path)
        pred_df = load_mot_file(result_path)

        acc = mm.MOTAccumulator(auto_id=True)
        all_frames = sorted(set(gt_df["frame"]) | set(pred_df["frame"]))
        for frame in all_frames:
            gt_frame = gt_df[gt_df["frame"] == frame]
            pred_frame = pred_df[pred_df["frame"] == frame]
            dist_matrix = mm.distances.iou_matrix(
                gt_frame[["x", "y", "w", "h"]].values, pred_frame[["x", "y", "w", "h"]].values, max_iou=0.5,
            )
            acc.update(gt_frame["id"].values, pred_frame["id"].values, dist_matrix)
        acc_list.append(acc)
        seq_names.append(seq_name)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        acc_list, metrics=["mota", "motp", "idf1", "num_switches"], names=seq_names, generate_overall=True,
    )
    overall = summary.loc["OVERALL"]

    timing_path = MOT_EVAL_DIR / f"timing_{tracker_name}.csv"
    fps = float("nan")
    if timing_path.exists():
        timing_df = pd.read_csv(timing_path)
        overall_timing = timing_df[timing_df["sequence"] == "OVERALL"]
        if not overall_timing.empty:
            fps = overall_timing["fps"].values[0]

    return {
        "tracker": tracker_name,
        "MOTA(%)": round(overall["mota"] * 100, 1),
        "MOTP": round(overall["motp"], 3),
        "IDF1(%)": round(overall["idf1"] * 100, 1),
        "IDs": int(overall["num_switches"]),
        "FPS": round(fps, 2),
    }


def main():
    model = YOLO(str(WEIGHTS))
    rows = []

    for tracker_name in TRACKERS:
        existing = sorted(MOT_EVAL_DIR.glob(f"*_{tracker_name}.txt"))
        if len(existing) >= 25:  # já rodamos botsort/bytetrack antes -- reaproveita
            print(f"[{tracker_name}] resultados já existentes, pulando execução, só avaliando.")
        else:
            print(f"[{tracker_name}] rodando tracking em 30 sequências...")
            ok = run_tracker(tracker_name, model)
            if not ok:
                print(f"[{tracker_name}] falhou completamente, pulando avaliação.")
                continue

        metrics = evaluate_tracker(tracker_name)
        if metrics:
            rows.append(metrics)
            print(f"  -> {metrics}\n")

    df = pd.DataFrame(rows).sort_values("IDF1(%)", ascending=False)
    print("=" * 80)
    print("LEADERBOARD MULTI-TRACKER")
    print("=" * 80)
    print(df.to_string(index=False))

    out_csv = MOT_EVAL_DIR / "multitracker_leaderboard.csv"
    df.to_csv(out_csv, index=False)
    out_md = MOT_EVAL_DIR / "multitracker_leaderboard.md"
    out_md.write_text(df.to_markdown(index=False))
    print(f"\nSalvo em: {out_csv} e {out_md}")


if __name__ == "__main__":
    main()