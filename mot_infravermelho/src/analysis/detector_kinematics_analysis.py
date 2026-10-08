#!/usr/bin/env python3
"""
Analisa a estabilidade temporal das bounding boxes produzidas pelo
detector + BoT-SORT (pipeline HardMining + SAHI512), frame a frame
por track, para calibrar futuras restrições de escala no tracker.

Baseado em gt_kinematics_analysis.py – substitui GT por detecções.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from inference.common import MOT_EVAL_DIR, load_mot_file

# ----------------------------------------------------------------------
# Configuração do experimento (ajuste se necessário)
# Busca todos os arquivos do pipeline HardMining + SAHI512 em runs/mot_eval/
DET_FILES_PATTERN = "*_hardmining_sahi512.txt"
FRAME_W, FRAME_H = 640, 512                     # dimensões do frame
FRAME_DIAG = np.sqrt(FRAME_W**2 + FRAME_H**2)
# ----------------------------------------------------------------------

def main():
    # Lista os arquivos MOT do diretório de detecções com o padrão correto
    det_files = sorted(MOT_EVAL_DIR.glob(DET_FILES_PATTERN))
    if not det_files:
        raise FileNotFoundError(
            f"Nenhum arquivo com padrão '{DET_FILES_PATTERN}' encontrado em {MOT_EVAL_DIR}"
        )

    print(f"{len(det_files)} arquivos de detecção encontrados (HardMining+SAHI512).\n")

    # Coletores das métricas
    all_area_ratio = []
    all_width_ratio = []
    all_height_ratio = []
    all_center_shift_norm = []
    all_center_shift_px = []

    # Para classificação por tamanho do drone
    track_avg_areas = []          # área média da track
    track_metrics = []            # mesmo índice para associar depois

    for det_file in det_files:
        seq_name = det_file.stem  # ex.: "UAV005_hardmining_sahi512"
        df = load_mot_file(det_file)

        # Colunas do MOT: frame, id, x, y, w, h, conf, -1, -1
        df["area"] = df["w"] * df["h"]
        df["cx"] = df["x"] + df["w"] / 2
        df["cy"] = df["y"] + df["h"] / 2

        for tid, group in df.groupby("id"):
            group = group.sort_values("frame")
            # Ignora tracks com menos de 2 detecções
            if len(group) < 2:
                continue

            # Área média da track para classificação de tamanho
            avg_area = group["area"].mean()
            track_avg_areas.append(avg_area)

            areas = group["area"].values
            widths = group["w"].values
            heights = group["h"].values
            cxs = group["cx"].values
            cys = group["cy"].values
            frames = group["frame"].values

            # Coleta temporária para essa track
            t_area_ratio = []
            t_width_ratio = []
            t_height_ratio = []
            t_center_norm = []
            t_center_px = []

            for i in range(1, len(group)):
                # Apenas quadros consecutivos
                if frames[i] - frames[i - 1] != 1:
                    continue
                if areas[i - 1] <= 0 or widths[i - 1] <= 0 or heights[i - 1] <= 0:
                    continue

                area_r = areas[i] / areas[i - 1]
                width_r = widths[i] / widths[i - 1]
                height_r = heights[i] / heights[i - 1]
                shift_px = np.sqrt((cxs[i] - cxs[i - 1])**2 + (cys[i] - cys[i - 1])**2)
                shift_norm = shift_px / FRAME_DIAG

                t_area_ratio.append(area_r)
                t_width_ratio.append(width_r)
                t_height_ratio.append(height_r)
                t_center_norm.append(shift_norm)
                t_center_px.append(shift_px)

            if t_area_ratio:
                all_area_ratio.extend(t_area_ratio)
                all_width_ratio.extend(t_width_ratio)
                all_height_ratio.extend(t_height_ratio)
                all_center_shift_norm.extend(t_center_norm)
                all_center_shift_px.extend(t_center_px)
                # Guarda as métricas da track junto com a área média
                track_metrics.append({
                    "avg_area": avg_area,
                    "area_ratio": t_area_ratio,
                    "width_ratio": t_width_ratio,
                    "height_ratio": t_height_ratio,
                    "center_shift_norm": t_center_norm,
                    "center_shift_px": t_center_px,
                })

    # Converte para arrays numpy
    area_ratios = np.array(all_area_ratio)
    width_ratios = np.array(all_width_ratio)
    height_ratios = np.array(all_height_ratio)
    center_shifts_norm = np.array(all_center_shift_norm)
    center_shifts_px = np.array(all_center_shift_px)

    n = len(area_ratios)
    print(f"Total de transições frame‑a‑frame analisadas: {n}\n")

    # ------------------------------------------------------------------
    # Classificação por tamanho (small / medium / large)
    # Usa percentis da área média de todas as tracks
    if len(track_avg_areas) > 2:
        p33 = np.percentile(track_avg_areas, 33.33)
        p66 = np.percentile(track_avg_areas, 66.67)
    else:
        p33 = p66 = 0  # fallback

    def size_label(avg_area):
        if avg_area <= p33:
            return "small"
        elif avg_area <= p66:
            return "medium"
        else:
            return "large"

    # Constrói DataFrame final com todas as transições e rótulo de tamanho
    rows = []
    for tm in track_metrics:
        lbl = size_label(tm["avg_area"])
        for ar, wr, hr, cn, cp in zip(tm["area_ratio"], tm["width_ratio"],
                                       tm["height_ratio"], tm["center_shift_norm"],
                                       tm["center_shift_px"]):
            rows.append([ar, wr, hr, cn, cp, tm["avg_area"], lbl])
    df_all = pd.DataFrame(rows, columns=["area_ratio", "width_ratio", "height_ratio",
                                         "center_shift_norm", "center_shift_px",
                                         "avg_area", "size_group"])

    # Estatísticas desejadas
    percentiles = [50, 90, 95, 99, 99.5, 99.9]
    metrics = {
        "area_ratio": "area_ratio",
        "width_ratio": "width_ratio",
        "height_ratio": "height_ratio",
        "center_shift_norm": "center_shift_norm"
    }

    def compute_stats(series):
        s = {
            "count": len(series),
            "mean": series.mean(),
            "std": series.std(),
            "median": series.median()
        }
        for p in percentiles:
            s[f"p{p}"] = np.percentile(series, p)
        return s

    # Estatísticas gerais
    summary_lines = ["# Detector Kinematics Analysis\n"]
    summary_lines.append(f"Pipeline: HardMining + SAHI512 (BoT‑SORT)")
    summary_lines.append(f"Total de transições analisadas: {n}\n")

    summary_lines.append("## Estatísticas Gerais\n")
    summary_lines.append("| Métrica | count | mean | std | median | p90 | p95 | p99 | p99.5 | p99.9 |")
    summary_lines.append("|---------|-------|------|-----|--------|-----|-----|-----|-------|-------|")
    for col, label in metrics.items():
        stats = compute_stats(df_all[col])
        line = f"| {label} | {stats['count']} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['median']:.4f} | {stats['p90']:.4f} | {stats['p95']:.4f} | {stats['p99']:.4f} | {stats['p99.5']:.4f} | {stats['p99.9']:.4f} |"
        summary_lines.append(line)

    # Estatísticas por grupo de tamanho
    for grp in ["small", "medium", "large"]:
        grp_df = df_all[df_all["size_group"] == grp]
        if len(grp_df) == 0:
            continue
        summary_lines.append(f"\n## Grupo: {grp}\n")
        summary_lines.append("| Métrica | count | mean | std | median | p90 | p95 | p99 | p99.5 | p99.9 |")
        summary_lines.append("|---------|-------|------|-----|--------|-----|-----|-----|-------|-------|")
        for col, label in metrics.items():
            stats = compute_stats(grp_df[col])
            line = f"| {label} | {stats['count']} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['median']:.4f} | {stats['p90']:.4f} | {stats['p95']:.4f} | {stats['p99']:.4f} | {stats['p99.5']:.4f} | {stats['p99.9']:.4f} |"
            summary_lines.append(line)

    # Salva o Markdown
    md_path = THIS_DIR / "detector_kinematics_summary.md"
    md_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Sumário salvo em {md_path}")

    # Salva CSV completo
    csv_path = THIS_DIR / "detector_kinematics.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"CSV salvo em {csv_path}")

    # ------------------------------------------------------------------
    # Histogramas
    def save_hist(data, title, xlabel, filename, bins=100):
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=bins, alpha=0.75, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequência")
        plt.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        plt.savefig(THIS_DIR / filename, dpi=150)
        plt.close()
        print(f"Histograma salvo: {filename}")

    save_hist(df_all["area_ratio"], "Distribuição da Razão de Área (detecções)", "area_ratio", "hist_area_ratio.png")
    save_hist(df_all["width_ratio"], "Distribuição da Razão de Largura", "width_ratio", "hist_width_ratio.png")
    save_hist(df_all["height_ratio"], "Distribuição da Razão de Altura", "height_ratio", "hist_height_ratio.png")
    save_hist(df_all["center_shift_px"], "Deslocamento do Centro (pixels)", "center_shift (px)", "hist_center_shift.png")

    print("\nAnálise concluída.")

if __name__ == "__main__":
    main()