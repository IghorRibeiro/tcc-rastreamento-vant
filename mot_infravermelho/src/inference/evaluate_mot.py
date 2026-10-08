"""
Avalia os resultados de tracking (formato MOT) contra o ground truth
original das sequências de validação, usando py-motmetrics.

Cruza automaticamente com o timing_<tracker>.csv gerado por
run_tracker_on_val.py, produzindo um resumo único com métricas de
qualidade (MOTA, IDF1, IDs) e de velocidade (FPS) lado a lado.
"""
import argparse
import numpy as np

# Compatibilidade: np.asfarray foi removido no NumPy 2.0, mas o
# motmetrics (biblioteca sem atualização recente) ainda depende dela.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

from pathlib import Path

import motmetrics as mm
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_ROOT = PROJECT_ROOT / "dataset" / "MultiUAV_Train" / "original_label_file"
RESULTS_DIR = PROJECT_ROOT / "runs" / "mot_eval"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tracker",
    default="botsort",
    help="Nome do tracker (botsort, sahi512, sahi384, sahi320...)"
)
args = parser.parse_args()

TRACKER_NAME = args.tracker


def load_mot_file(path: Path) -> pd.DataFrame:
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "a", "b", "c"]
    return pd.read_csv(path, header=None, names=cols)


def load_timing(tracker_name: str) -> pd.DataFrame:
    timing_path = RESULTS_DIR / f"timing_{tracker_name}.csv"
    if not timing_path.exists():
        print(f"[AVISO] {timing_path} não encontrado. FPS não será incluído no resumo.")
        return pd.DataFrame(columns=["sequence", "fps", "elapsed_seconds", "n_frames"])
    df = pd.read_csv(timing_path)
    return df[df["sequence"] != "OVERALL"]  # a linha OVERALL é tratada à parte


def main():
    result_files = sorted(RESULTS_DIR.glob(f"*_{TRACKER_NAME}.txt"))
    if not result_files:
        raise RuntimeError(f"Nenhum resultado encontrado em {RESULTS_DIR}. Rode run_tracker_on_val.py antes.")

    accumulator_summaries = []
    seq_names = []

    for result_path in result_files:
        seq_name = result_path.stem.replace(f"_{TRACKER_NAME}", "")
        gt_path = LABEL_ROOT / seq_name / "gt" / "gt.txt"
        if not gt_path.exists():
            print(f"[AVISO] gt.txt não encontrado para {seq_name}, pulando")
            continue

        gt_df = load_mot_file(gt_path)
        pred_df = load_mot_file(result_path)

        acc = mm.MOTAccumulator(auto_id=True)

        all_frames = sorted(set(gt_df["frame"]) | set(pred_df["frame"]))
        for frame in all_frames:
            gt_frame = gt_df[gt_df["frame"] == frame]
            pred_frame = pred_df[pred_df["frame"] == frame]

            gt_ids = gt_frame["id"].values
            pred_ids = pred_frame["id"].values

            gt_boxes = gt_frame[["x", "y", "w", "h"]].values
            pred_boxes = pred_frame[["x", "y", "w", "h"]].values

            dist_matrix = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
            acc.update(gt_ids, pred_ids, dist_matrix)

        accumulator_summaries.append(acc)
        seq_names.append(seq_name)
        print(f"{seq_name}: processado ({len(all_frames)} frames)")

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accumulator_summaries,
        metrics=["mota", "motp", "idf1", "num_switches", "num_false_positives", "num_misses", "num_objects"],
        names=seq_names,
        generate_overall=True,
    )

    print("\n" + "=" * 70)
    print(f"Resultados MOT (sem FPS) — tracker: {TRACKER_NAME}")
    print("=" * 70)
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    ))

    # ---------- Cruza com timing (FPS) ----------
    timing_df = load_timing(TRACKER_NAME).set_index("sequence")

    combined = summary.copy()
    combined["fps"] = combined.index.map(lambda s: timing_df["fps"].get(s, float("nan")))
    combined["elapsed_seconds"] = combined.index.map(lambda s: timing_df["elapsed_seconds"].get(s, float("nan")))

    # Para a linha OVERALL, usa o FPS médio agregado salvo pelo run_tracker_on_val.py
    overall_timing_path = RESULTS_DIR / f"timing_{TRACKER_NAME}.csv"
    if overall_timing_path.exists():
        full_timing = pd.read_csv(overall_timing_path)
        overall_row = full_timing[full_timing["sequence"] == "OVERALL"]
        if not overall_row.empty and "OVERALL" in combined.index:
            combined.loc["OVERALL", "fps"] = overall_row["fps"].values[0]
            combined.loc["OVERALL", "elapsed_seconds"] = overall_row["elapsed_seconds"].values[0]

    # Converte MOTA/MOTP/IDF1 (frações) para % legível na coluna combinada
    display_df = combined.copy()
    display_df["mota"] = (display_df["mota"] * 100).round(1)
    display_df["idf1"] = (display_df["idf1"] * 100).round(1)
    display_df["motp"] = display_df["motp"].round(3)
    display_df["fps"] = display_df["fps"].round(2)

    display_cols = ["mota", "motp", "idf1", "num_switches", "fps", "elapsed_seconds"]
    display_df = display_df[display_cols]
    display_df.columns = ["MOTA(%)", "MOTP", "IDF1(%)", "IDs", "FPS", "Time(s)"]

    print("\n" + "=" * 70)
    print(f"Resumo combinado (qualidade + velocidade) — tracker: {TRACKER_NAME}")
    print("=" * 70)
    print(display_df.to_string())

    # ---------- Salva tudo ----------
    csv_path = RESULTS_DIR / f"mot_summary_{TRACKER_NAME}.csv"
    combined.to_csv(csv_path)
    print(f"\nResumo completo (bruto) salvo em: {csv_path}")

    display_csv_path = RESULTS_DIR / f"mot_summary_{TRACKER_NAME}_readable.csv"
    display_df.to_csv(display_csv_path)
    print(f"Resumo legível salvo em: {display_csv_path}")

    txt_path = RESULTS_DIR / f"mot_summary_{TRACKER_NAME}.txt"
    txt_path.write_text(display_df.to_string())
    print(f"Resumo formatado (texto) salvo em: {txt_path}")


if __name__ == "__main__":
    main()