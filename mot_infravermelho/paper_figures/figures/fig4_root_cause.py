import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/inference"))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from common import MOT_EVAL_DIR
from metrics import (
    get_density_dataframe, get_camera_motion_dataframe,
    ieee_style, save_figure, pearson_r, confidence_interval
)


def generate_figure4(output_dir, csv_dir):
    ieee_style()

    # 1. Carrega métricas de densidade e movimento (já existentes)
    df_density = get_density_dataframe()
    df_motion = get_camera_motion_dataframe()

    # 2. Carrega métricas de qualidade de imagem a partir do CSV gerado pelo seu script
    quality_csv = MOT_EVAL_DIR / "frame_quality_metrics.csv"
    if not quality_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {quality_csv}. Execute frame_quality_metrics.py primeiro.")
    df_quality = pd.read_csv(quality_csv)

    # 3. Carrega resultados do tracker (BoT-SORT) – usa o arquivo com IDF1
    tracker_csv = MOT_EVAL_DIR / "mot_summary_botsort_readable.csv"
    if not tracker_csv.exists():
        raise FileNotFoundError(f"Tracker results not found: {tracker_csv}")
    df_results = pd.read_csv(tracker_csv, index_col=0)
    df_results = df_results.drop(index="OVERALL", errors='ignore').reset_index().rename(columns={'index': 'sequence'})
    df_results = df_results[['sequence', 'IDF1(%)']]  # mantém só IDF1

    # 4. Faz merge de todas as métricas
    df = df_density.merge(df_motion, on='sequence', how='inner')
    df = df.merge(df_quality, on='sequence', how='inner')
    df = df.merge(df_results, on='sequence', how='inner')

    # Salva CSV consolidado para depuração
    df.to_csv(csv_dir / 'fig4_data.csv', index=False)

    # 5. Configuração dos subplots (1x4)
    fig, axes = plt.subplots(1, 4, figsize=(9, 3.2), sharey=True)
    plt.subplots_adjust(wspace=0.08)

    # As colunas de interesse – note que usamos as colunas exatas do seu CSV
    configs = [
        (df, 'mean_density', 'Swarm density\n(drones/frame)', '(a)'),
        (df, 'mean_motion', 'Camera motion\n(px)', '(b)'),
        (df, 'mean_edge_density', 'Edge density', '(c)'),
        (df, 'mean_local_contrast', 'Local contrast', '(d)'),   # <-- corrigido!
    ]

    for ax, (data, x_col, xlabel, label) in zip(axes, configs):
        mask = data[x_col].notna() & data['IDF1(%)'].notna()
        x = data.loc[mask, x_col].values
        y = data.loc[mask, 'IDF1(%)'].values

        # Scatter
        ax.scatter(x, y, color='black', marker='o', alpha=0.6, s=20)

        # Regressão e IC
        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x), max(x), 50)
            ax.plot(x_line, p(x_line), color='black', linestyle='--', linewidth=1)

            if len(x) >= 3:
                ci_func = confidence_interval(x, y)
                if ci_func is not None:
                    lower, upper = ci_func(x_line)
                    ax.fill_between(x_line, lower, upper, color='black', alpha=0.12)

            # Pearson r no canto superior direito
            r, _ = pearson_r(x, y)
            ax.text(0.95, 0.92, f'r = {r:.2f}', transform=ax.transAxes,
                    fontsize=8, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7))

        # Rótulos
        ax.set_xlabel(xlabel, fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel('IDF1 (%)', fontsize=8)
        else:
            ax.set_ylabel('')

        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top')

    plt.tight_layout()
    save_figure(fig, output_dir / 'fig4_root_cause')