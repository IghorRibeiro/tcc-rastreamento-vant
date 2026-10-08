"""
Avalia baseline (sem pré-processamento) vs. MOG2 vs. EMA vs. Optical
Flow, consolidando MOTA/IDF1/FP-rate/FN-rate/FPS numa tabela única.
"""

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm
import pandas as pd

from common import LABEL_ROOT, MOT_EVAL_DIR, load_mot_file

METHODS = ["botsort", "mog2", "ema", "optflow"]  # "botsort" = baseline sem pré-processamento


def evaluate_method(method_name: str):
    result_files = sorted(MOT_EVAL_DIR.glob(f"*_{method_name}.txt"))
    if not result_files:
        print(f"[AVISO] nenhum resultado para '{method_name}', pulando")
        return None

    acc_list, seq_names = [], []
    for result_path in result_files:
        seq_name = result_path.stem.replace(f"_{method_name}", "")
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
                gt_frame[["x", "y", "w", "h"]].values,
                pred_frame[["x", "y", "w", "h"]].values,
                max_iou=0.5,
            )
            acc.update(gt_frame["id"].values, pred_frame["id"].values, dist_matrix)

        acc_list.append(acc)
        seq_names.append(seq_name)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        acc_list,
        metrics=["mota", "motp", "idf1", "num_switches",
                 "num_false_positives", "num_misses", "num_objects"],
        names=seq_names,
        generate_overall=True,
    )
    overall = summary.loc["OVERALL"]

    timing_path = MOT_EVAL_DIR / f"timing_{method_name}.csv"
    fps = float("nan")
    if timing_path.exists():
        timing_df = pd.read_csv(timing_path)
        overall_timing = timing_df[timing_df["sequence"] == "OVERALL"]
        if not overall_timing.empty:
            fps = overall_timing["fps"].values[0]

    return {
        "method": method_name,
        "MOTA(%)": round(overall["mota"] * 100, 1),
        "IDF1(%)": round(overall["idf1"] * 100, 1),
        "IDs": int(overall["num_switches"]),
        "FP_rate": round(overall["num_false_positives"] / overall["num_objects"], 4),
        "FN_rate": round(overall["num_misses"] / overall["num_objects"], 4),
        "FPS": round(fps, 2),
    }


def main():
    rows = [r for m in METHODS if (r := evaluate_method(m)) is not None]
    df = pd.DataFrame(rows).set_index("method")

    print("\n" + "=" * 75)
    print("Ablação de Pré-processamento Temporal: baseline vs. MOG2 vs. EMA vs. Optical Flow")
    print("=" * 75)
    print(df.to_string())

    out_csv = MOT_EVAL_DIR / "preprocessing_ablation_summary.csv"
    df.to_csv(out_csv)
    out_md = MOT_EVAL_DIR / "preprocessing_ablation_summary.md"
    out_md.write_text(df.to_markdown())
    print(f"\nSalvo em: {out_csv} e {out_md}")


if __name__ == "__main__":
    main()