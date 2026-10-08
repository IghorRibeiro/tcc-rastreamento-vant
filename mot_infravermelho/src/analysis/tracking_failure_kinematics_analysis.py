#!/usr/bin/env python3
"""
Analisa as transições cinemáticas das detecções do pipeline
HardMining+SAHI512, classificadas por evento de tracking (TP, FP,
ID_SWITCH). Verifica se erros de associação possuem assinatura
cinemática diferente dos acertos.

Baseado em evaluate_mot.py (via motmetrics) e detector_kinematics_analysis.py.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.optimize import linear_sum_assignment
import importlib.util

# Ajuste de caminhos: script está em src/analysis/
THIS_DIR = Path(__file__).resolve().parent            # .../src/analysis
SRC_DIR = THIS_DIR.parent                             # .../src
INFERENCE_DIR = SRC_DIR / "inference"                 # .../src/inference

# Carrega o módulo common diretamente do caminho, evitando conflito com site-packages
common_path = INFERENCE_DIR / "common.py"
spec = importlib.util.spec_from_file_location("inference_common", common_path)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

LABEL_ROOT = common.LABEL_ROOT
MOT_EVAL_DIR = common.MOT_EVAL_DIR
load_mot_file = common.load_mot_file
get_val_sequences = common.get_val_sequences

# ----------------------------------------------------------------------
# Configurações do matching (idênticas ao evaluate_mot.py)
IOU_THRESHOLD = 0.5
FRAME_W, FRAME_H = 640, 512
FRAME_DIAG = np.sqrt(FRAME_W**2 + FRAME_H**2)
# ----------------------------------------------------------------------

def iou(bb1, bb2):
    """Calcula IoU entre duas bounding boxes [x,y,w,h]."""
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def match_frame(det_df, gt_df, threshold=IOU_THRESHOLD):
    """
    Associa detecções a ground truths usando IoU e algoritmo Húngaro.
    Retorna: matches (lista de tuplas (idx_det, idx_gt)),
             det_unmatched (índices), gt_unmatched (índices).
    """
    if det_df.empty or gt_df.empty:
        return [], list(range(len(det_df))), list(range(len(gt_df)))

    det_boxes = det_df[['x', 'y', 'w', 'h']].values
    gt_boxes = gt_df[['x', 'y', 'w', 'h']].values
    cost = np.zeros((len(det_boxes), len(gt_boxes)))
    for i, db in enumerate(det_boxes):
        for j, gb in enumerate(gt_boxes):
            cost[i, j] = 1 - iou(db, gb)

    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    det_matched = set()
    gt_matched = set()
    for r, c in zip(row_ind, col_ind):
        if 1 - cost[r, c] >= threshold:
            matches.append((r, c))
            det_matched.add(r)
            gt_matched.add(c)

    det_unmatched = [i for i in range(len(det_boxes)) if i not in det_matched]
    gt_unmatched = [j for j in range(len(gt_boxes)) if j not in gt_matched]
    return matches, det_unmatched, gt_unmatched

def process_sequence(det_path, gt_path):
    """
    Processa uma sequência: associa frame‑a‑frame, classifica eventos
    e retorna DataFrame de detecções com colunas:
    frame, track_id, x, y, w, h, event, gt_id (opcional).
    """
    det_all = load_mot_file(det_path)
    gt_all = load_mot_file(gt_path)

    # Mapa global: GT ID -> último track_id associado (para detectar ID switches)
    gt_last_track = {}
    events = []  # lista de dicts: {frame, track_id, x, y, w, h, event}

    for frame_num in sorted(det_all['frame'].unique()):
        det_frame = det_all[det_all['frame'] == frame_num]
        gt_frame = gt_all[gt_all['frame'] == frame_num]

        # Associações
        matches, det_unmatched, gt_unmatched = match_frame(det_frame, gt_frame)

        # TP e possíveis ID switches
        for i_det, i_gt in matches:
            det_row = det_frame.iloc[i_det]
            gt_row = gt_frame.iloc[i_gt]
            track_id = int(det_row['id'])
            gt_id = int(gt_row['id'])
            # Verifica ID switch
            if gt_id in gt_last_track and gt_last_track[gt_id] != track_id:
                event = 'ID_SWITCH'
            else:
                event = 'TP'
            events.append({
                'frame': frame_num,
                'track_id': track_id,
                'x': det_row['x'], 'y': det_row['y'],
                'w': det_row['w'], 'h': det_row['h'],
                'event': event,
                'gt_id': gt_id
            })
            gt_last_track[gt_id] = track_id

        # FP
        for i_det in det_unmatched:
            det_row = det_frame.iloc[i_det]
            events.append({
                'frame': frame_num,
                'track_id': int(det_row['id']),
                'x': det_row['x'], 'y': det_row['y'],
                'w': det_row['w'], 'h': det_row['h'],
                'event': 'FP',
                'gt_id': None
            })

        # FN (apenas registramos, mas não afeta as detecções)
        # Não serão adicionados a events.

    # Ordena e retorna DataFrame
    df_events = pd.DataFrame(events)
    if df_events.empty:
        return df_events
    df_events = df_events.sort_values(['track_id', 'frame']).reset_index(drop=True)
    return df_events

def main():
    val_seqs = get_val_sequences()
    print(f"Processando {len(val_seqs)} sequências de validação...\n")

    all_transitions = []

    for seq in val_seqs:
        det_path = MOT_EVAL_DIR / f"{seq}_hardmining_sahi512.txt"
        gt_path = LABEL_ROOT / seq / "gt" / "gt.txt"
        if not det_path.exists() or not gt_path.exists():
            print(f"  Pulando {seq}: arquivos não encontrados.")
            continue

        print(f"  {seq}...")
        df_events = process_sequence(det_path, gt_path)
        if df_events.empty:
            continue

        # Calcula métricas de transição para cada track
        for tid, track_df in df_events.groupby('track_id'):
            track_df = track_df.sort_values('frame')
            if len(track_df) < 2:
                continue
            frames = track_df['frame'].values
            xs = track_df['x'].values
            ys = track_df['y'].values
            ws = track_df['w'].values
            hs = track_df['h'].values
            events = track_df['event'].values
            areas = ws * hs
            cxs = xs + ws / 2
            cys = ys + hs / 2

            for i in range(1, len(track_df)):
                if frames[i] - frames[i-1] != 1:
                    continue
                if areas[i-1] <= 0:
                    continue
                area_r = areas[i] / areas[i-1]
                width_r = ws[i] / ws[i-1]
                height_r = hs[i] / hs[i-1]
                aspect_prev = ws[i-1] / hs[i-1] if hs[i-1] > 0 else 0
                aspect_curr = ws[i] / hs[i] if hs[i] > 0 else 0
                aspect_r = aspect_curr / aspect_prev if aspect_prev > 0 else np.nan
                shift_px = np.sqrt((cxs[i] - cxs[i-1])**2 + (cys[i] - cys[i-1])**2)
                shift_norm = shift_px / FRAME_DIAG

                all_transitions.append({
                    'track_id': tid,
                    'frame': frames[i],
                    'event': events[i],   # evento da detecção atual
                    'area_ratio': area_r,
                    'width_ratio': width_r,
                    'height_ratio': height_r,
                    'center_shift': shift_px,
                    'center_shift_normalized': shift_norm,
                    'aspect_ratio_ratio': aspect_r
                })

    # DataFrame final
    if not all_transitions:
        print("Nenhuma transição encontrada. Encerrando.")
        return

    trans_df = pd.DataFrame(all_transitions)
    # Remove linhas com aspect_ratio_ratio NaN (se houver)
    trans_df = trans_df.dropna(subset=['aspect_ratio_ratio'])

    # Salva CSV
    csv_path = THIS_DIR / "tracking_failure_kinematics.csv"
    trans_df.to_csv(csv_path, index=False)
    print(f"\nCSV salvo em {csv_path} ({len(trans_df)} transições)")

    # --------------------------------------------------------------
    # Estatísticas descritivas
    metrics = ['area_ratio', 'width_ratio', 'height_ratio', 'center_shift_normalized', 'aspect_ratio_ratio']
    events_of_interest = ['TP', 'FP', 'ID_SWITCH']
    percentiles = [50, 90, 95, 99, 99.5, 99.9]

    def compute_stats(series):
        s = {
            'count': len(series),
            'mean': series.mean(),
            'std': series.std(),
            'median': series.median()
        }
        for p in percentiles:
            s[f'p{p}'] = np.percentile(series, p)
        return s

    summary = ["# Cinemática por Evento de Tracking\n"]
    summary.append(f"Pipeline: HardMining + SAHI512 (BoT‑SORT)")
    summary.append(f"Total de transições analisadas: {len(trans_df)}\n")

    # Estatísticas por evento
    for ev in events_of_interest + ['FN']:
        if ev == 'FN':
            summary.append("## FN (não gera transições, count=0)\n")
            continue
        subset = trans_df[trans_df['event'] == ev]
        if subset.empty:
            summary.append(f"## {ev} (sem transições)\n")
            continue
        summary.append(f"## {ev}\n")
        summary.append("| Métrica | count | mean | std | median | p90 | p95 | p99 | p99.5 | p99.9 |")
        summary.append("|---------|-------|------|-----|--------|-----|-----|-----|-------|-------|")
        for m in metrics:
            stats = compute_stats(subset[m])
            line = f"| {m} | {stats['count']} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['median']:.4f} | {stats['p90']:.4f} | {stats['p95']:.4f} | {stats['p99']:.4f} | {stats['p99.5']:.4f} | {stats['p99.9']:.4f} |"
            summary.append(line)
        summary.append("")

    # --------------------------------------------------------------
    # Testes Mann-Whitney (TP vs FP, TP vs ID_SWITCH)
    summary.append("## Comparação Estatística (Mann‑Whitney U)\n")
    tp = trans_df[trans_df['event'] == 'TP']
    for other_ev in ['FP', 'ID_SWITCH']:
        other = trans_df[trans_df['event'] == other_ev]
        if tp.empty or other.empty:
            summary.append(f"**TP vs {other_ev}**: dados insuficientes.\n")
            continue
        summary.append(f"### TP vs {other_ev}\n")
        summary.append("| Métrica | p-valor | Mediana TP | Mediana {0} | Média TP | Média {0} |".format(other_ev))
        summary.append("|---------|---------|------------|--------------|----------|------------|")
        for m in metrics:
            u_stat, p_val = mannwhitneyu(tp[m], other[m], alternative='two-sided')
            med_tp = tp[m].median()
            med_other = other[m].median()
            mean_tp = tp[m].mean()
            mean_other = other[m].mean()
            summary.append(f"| {m} | {p_val:.2e} | {med_tp:.4f} | {med_other:.4f} | {mean_tp:.4f} | {mean_other:.4f} |")
        summary.append("")

    # Conclusão automática
    summary.append("## Conclusão Preliminar\n")
    # Analisa p-valores para dar uma resposta automática
    significant = []
    for other_ev in ['FP', 'ID_SWITCH']:
        other = trans_df[trans_df['event'] == other_ev]
        if tp.empty or other.empty:
            continue
        for m in metrics:
            _, p = mannwhitneyu(tp[m], other[m], alternative='two-sided')
            if p < 0.001:
                significant.append(f"{m} (TP vs {other_ev})")
    if significant:
        summary.append("**Variáveis com diferença altamente significativa (p<0.001):**")
        for s in significant:
            summary.append(f"- {s}")
        summary.append("\nEssas variáveis são promissoras para incluir no custo de associação do BoT‑SORT.\n")
    else:
        summary.append("Não foram encontradas diferenças estatisticamente significativas (p<0.001) entre eventos.\n")

    # Salva markdown
    md_path = THIS_DIR / "tracking_failure_summary.md"
    md_path.write_text('\n'.join(summary), encoding='utf-8')
    print(f"Sumário salvo em {md_path}")

    # --------------------------------------------------------------
    # Gráficos
    sns.set_style("whitegrid")
    plot_events = ['TP', 'FP', 'ID_SWITCH']
    # Garante ordem
    trans_df['event'] = pd.Categorical(trans_df['event'], categories=plot_events, ordered=True)
    plot_vars = {
        'area_ratio': 'Area Ratio',
        'center_shift': 'Center Shift (pixels)',
        'width_ratio': 'Width Ratio',
        'height_ratio': 'Height Ratio'
    }

    for var, title in plot_vars.items():
        plt.figure(figsize=(8,5))
        sns.boxplot(x='event', y=var, data=trans_df, order=plot_events)
        plt.title(f'{title} por Evento')
        plt.tight_layout()
        plt.savefig(THIS_DIR / f'boxplot_{var}.png', dpi=150)
        plt.close()
        print(f'boxplot_{var}.png salvo')

    # Violin plot apenas para area_ratio
    plt.figure(figsize=(8,5))
    sns.violinplot(x='event', y='area_ratio', data=trans_df, order=plot_events, cut=0)
    plt.title('Area Ratio por Evento (Violin)')
    plt.tight_layout()
    plt.savefig(THIS_DIR / 'violin_area_ratio.png', dpi=150)
    plt.close()
    print('violin_area_ratio.png salvo')

    print("\nAnálise concluída.")

if __name__ == "__main__":
    main()