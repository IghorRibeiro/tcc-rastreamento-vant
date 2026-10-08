"""
Varredura de hiperparâmetros do filtro de recorrência espacial,
aplicado sobre os resultados já salvos do BoT-SORT (mesma lógica de
reuso do run_static_filter_sweep.py -- não requer re-rodar tracker).
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(SRC_DIR / "postprocessing"))
from common import LABEL_ROOT, MOT_EVAL_DIR, load_mot_file  # noqa: E402
from spatial_recurrence_filter import filter_by_spatial_recurrence  # noqa: E402

SOURCE_TRACKER = "botsort"
FRAME_W, FRAME_H = 640, 512  # confirmado na Fase 0 do projeto

CELL_SIZES = [10, 20, 40]
OCCUPANCY_THRESHOLDS = [0.2, 0.3, 0.5, 0.7]


def evaluate_config(all_kept_dfs: dict):
    acc_list, seq_names = [], []
    total_suppressed = total_original = 0

    for seq_name, (kept_df, n_original, n_suppressed) in all_kept_dfs.items():
        gt_path = LABEL_ROOT / seq_name / "gt" / "gt.txt"
        if not gt_path.exists():
            continue
        gt_df = load_mot_file(gt_path)

        acc = mm.MOTAccumulator(auto_id=True)
        all_frames = sorted(set(gt_df["frame"]) | set(kept_df["frame"]))
        for frame in all_frames:
            gt_frame = gt_df[gt_df["frame"] == frame]
            pred_frame = kept_df[kept_df["frame"] == frame]
            dist_matrix = mm.distances.iou_matrix(
                gt_frame[["x", "y", "w", "h"]].values,
                pred_frame[["x", "y", "w", "h"]].values,
                max_iou=0.5,
            )
            acc.update(gt_frame["id"].values, pred_frame["id"].values, dist_matrix)

        acc_list.append(acc)
        seq_names.append(seq_name)
        total_original += n_original
        total_suppressed += n_suppressed

    mh = mm.metrics.create()
    summary = mh.compute_many(
        acc_list,
        metrics=["mota", "motp", "idf1", "num_switches",
                 "num_false_positives", "num_misses", "num_objects"],
        names=seq_names,
        generate_overall=True,
    )
    overall = summary.loc["OVERALL"]
    suppression_rate = total_suppressed / total_original if total_original > 0 else 0.0

    return {
        "MOTA(%)": round(overall["mota"] * 100, 1),
        "IDF1(%)": round(overall["idf1"] * 100, 1),
        "IDs": int(overall["num_switches"]),
        "FP_rate": round(overall["num_false_positives"] / overall["num_objects"], 4),
        "FN_rate": round(overall["num_misses"] / overall["num_objects"], 4),
        "suppression_rate": round(suppression_rate, 4),
    }


def main():
    source_files = sorted(MOT_EVAL_DIR.glob(f"*_{SOURCE_TRACKER}.txt"))
    if not source_files:
        raise RuntimeError(f"Nenhum resultado '{SOURCE_TRACKER}' encontrado em {MOT_EVAL_DIR}.")

    raw_dfs = {}
    for f in source_files:
        seq_name = f.stem.replace(f"_{SOURCE_TRACKER}", "")
        raw_dfs[seq_name] = load_mot_file(f)
    print(f"Carregadas {len(raw_dfs)} sequências.\n")

    results = []
    for cell_size, occ_thresh in product(CELL_SIZES, OCCUPANCY_THRESHOLDS):
        print(f"Testando cell_size={cell_size}, occupancy_threshold={occ_thresh}...")
        per_seq = {}
        for seq_name, df in raw_dfs.items():
            kept_df, suppressed_df = filter_by_spatial_recurrence(
                df, FRAME_W, FRAME_H, cell_size=cell_size, occupancy_ratio_threshold=occ_thresh
            )
            per_seq[seq_name] = (kept_df, len(df), len(suppressed_df))

        metrics = evaluate_config(per_seq)
        metrics["cell_size"] = cell_size
        metrics["occupancy_threshold"] = occ_thresh
        results.append(metrics)
        print(f"  -> MOTA={metrics['MOTA(%)']}  IDF1={metrics['IDF1(%)']}  "
              f"IDs={metrics['IDs']}  suppressed={metrics['suppression_rate']:.1%}")

    baseline_per_seq = {name: (df, len(df), 0) for name, df in raw_dfs.items()}
    baseline_metrics = evaluate_config(baseline_per_seq)
    baseline_metrics["cell_size"] = "-"
    baseline_metrics["occupancy_threshold"] = "-"
    results.insert(0, baseline_metrics)

    df_results = pd.DataFrame(results)
    df_results = df_results[["cell_size", "occupancy_threshold", "MOTA(%)", "IDF1(%)",
                              "IDs", "FP_rate", "FN_rate", "suppression_rate"]]

    df_sorted = df_results.sort_values("IDF1(%)", ascending=False)
    print("\n" + "=" * 80)
    print("Resultado da varredura -- filtro de recorrência espacial (ordenado por IDF1)")
    print("=" * 80)
    print(df_sorted.to_string(index=False))

    out_csv = MOT_EVAL_DIR / "spatial_filter_sweep_summary.csv"
    df_results.to_csv(out_csv, index=False)
    out_md = MOT_EVAL_DIR / "spatial_filter_sweep_summary.md"
    out_md.write_text(df_sorted.to_markdown(index=False))
    print(f"\nSalvo em: {out_csv} e {out_md}")


if __name__ == "__main__":
    main()