"""
Analisa a correlação entre densidade de enxame (drones simultâneos
por frame) e desempenho de tracking (MOTA/IDF1), por sequência,
cruzando com os resultados já salvos em runs/mot_eval/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from common import LABEL_ROOT, MOT_EVAL_DIR, get_val_sequences, load_mot_file  # noqa: E402

OUT_DIR = MOT_EVAL_DIR  # reaproveita a mesma pasta de resultados


def compute_density_stats(seq_name: str):
    gt_path = LABEL_ROOT / seq_name / "gt" / "gt.txt"
    if not gt_path.exists():
        return None

    df = load_mot_file(gt_path)
    counts_per_frame = df.groupby("frame")["id"].count()

    return {
        "sequence": seq_name,
        "mean_density": counts_per_frame.mean(),
        "max_density": counts_per_frame.max(),
        "min_density": counts_per_frame.min(),
        "std_density": counts_per_frame.std(),
    }


def load_tracker_results(tracker_name: str) -> pd.DataFrame:
    path = MOT_EVAL_DIR / f"mot_summary_{tracker_name}_readable.csv"
    if not path.exists():
        print(f"[AVISO] {path} não encontrado, pulando '{tracker_name}'")
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df = df.drop(index="OVERALL", errors="ignore")
    df.index.name = "sequence"
    return df.reset_index()


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def main():
    val_seqs = get_val_sequences()

    density_rows = [compute_density_stats(s) for s in val_seqs]
    density_df = pd.DataFrame([r for r in density_rows if r is not None])
    print(f"Densidade calculada para {len(density_df)} sequências.\n")

    trackers = {"botsort": "tab:blue", "bytetrack": "tab:orange"}
    merged_all = density_df.copy()

    correlations = {}
    for tracker_name in trackers:
        results_df = load_tracker_results(tracker_name)
        if results_df.empty:
            continue

        merged = density_df.merge(results_df, on="sequence", how="inner")
        merged_all = merged_all.merge(
            results_df.rename(columns={c: f"{c}_{tracker_name}" for c in results_df.columns if c != "sequence"}),
            on="sequence",
            how="left",
        )

        r_mota = pearson_r(merged["mean_density"].values, merged["MOTA(%)"].values)
        r_idf1 = pearson_r(merged["mean_density"].values, merged["IDF1(%)"].values)
        r_ids = pearson_r(merged["mean_density"].values, merged["IDs"].values)

        correlations[tracker_name] = {"r_MOTA": r_mota, "r_IDF1": r_idf1, "r_IDs": r_ids}

        print(f"[{tracker_name}] correlação (densidade média x métrica):")
        print(f"  MOTA:  r = {r_mota:+.3f}")
        print(f"  IDF1:  r = {r_idf1:+.3f}")
        print(f"  IDs:   r = {r_ids:+.3f}\n")

    # ---------- Salva CSV consolidado ----------
    out_csv = OUT_DIR / "density_vs_performance.csv"
    merged_all.to_csv(out_csv, index=False)
    print(f"CSV consolidado salvo em: {out_csv}")

    # ---------- Gráfico de dispersão: densidade x IDF1 ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for tracker_name, color in trackers.items():
        results_df = load_tracker_results(tracker_name)
        if results_df.empty:
            continue
        merged = density_df.merge(results_df, on="sequence", how="inner")

        axes[0].scatter(merged["mean_density"], merged["MOTA(%)"], label=tracker_name, color=color, alpha=0.7)
        axes[1].scatter(merged["mean_density"], merged["IDF1(%)"], label=tracker_name, color=color, alpha=0.7)

        if len(merged) >= 2:
            for ax, col in [(axes[0], "MOTA(%)"), (axes[1], "IDF1(%)")]:
                z = np.polyfit(merged["mean_density"], merged[col], 1)
                x_line = np.linspace(merged["mean_density"].min(), merged["mean_density"].max(), 50)
                ax.plot(x_line, np.polyval(z, x_line), color=color, linestyle="--", alpha=0.5)

    axes[0].set_xlabel("Densidade média (drones/frame)")
    axes[0].set_ylabel("MOTA (%)")
    axes[0].set_title("Densidade x MOTA")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Densidade média (drones/frame)")
    axes[1].set_ylabel("IDF1 (%)")
    axes[1].set_title("Densidade x IDF1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = OUT_DIR / "density_vs_performance.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Gráfico salvo em: {fig_path}")


if __name__ == "__main__":
    main()