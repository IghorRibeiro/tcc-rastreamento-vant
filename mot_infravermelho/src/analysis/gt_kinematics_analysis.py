"""
Analisa a distribuição real de variação de área (e posição) do
ground truth, frame a frame por track, no SPLIT DE TREINO -- para
calibrar um gate de plausibilidade cinemática por tamanho, a ser
usado como termo de custo adicional no GM-FLIT (não testado ainda).
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = THIS_DIR.parent / "inference"

sys.path.insert(0, str(INFERENCE_DIR))

from common import (
    LABEL_ROOT,
    MOT_EVAL_DIR,
    load_mot_file,
    get_val_sequences,
)

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from inference.common import LABEL_ROOT, MOT_EVAL_DIR, load_mot_file


def get_train_sequences():
    all_seqs = sorted(p.name for p in LABEL_ROOT.iterdir() if p.is_dir())
    val_seqs = set(get_val_sequences())
    return [s for s in all_seqs if s not in val_seqs]


def main():
    train_seqs = get_train_sequences()
    print(f"{len(train_seqs)} sequências de treino encontradas.\n")

    area_ratios = []
    center_shifts = []  # em px, normalizado pela diagonal do frame (640x512)
    frame_diag = np.sqrt(640**2 + 512**2)

    for seq_name in train_seqs:
        gt_path = LABEL_ROOT / seq_name / "gt" / "gt.txt"
        if not gt_path.exists():
            continue
        df = load_mot_file(gt_path)
        df["area"] = df["w"] * df["h"]
        df["cx"] = df["x"] + df["w"] / 2
        df["cy"] = df["y"] + df["h"] / 2

        for tid, group in df.groupby("id"):
            group = group.sort_values("frame")
            areas = group["area"].values
            cxs, cys = group["cx"].values, group["cy"].values
            frames = group["frame"].values

            for i in range(1, len(group)):
                if frames[i] - frames[i - 1] != 1:  # só passos consecutivos
                    continue
                if areas[i - 1] <= 0:
                    continue
                ratio = areas[i] / areas[i - 1]
                area_ratios.append(ratio)

                shift = np.sqrt((cxs[i] - cxs[i - 1])**2 + (cys[i] - cys[i - 1])**2) / frame_diag
                center_shifts.append(shift)

    area_ratios = np.array(area_ratios)
    center_shifts = np.array(center_shifts)

    print(f"Total de transições frame-a-frame analisadas: {len(area_ratios)}\n")
    print("Distribuição de RAZÃO DE ÁREA (area_t / area_{t-1}) -- ground truth real:")
    for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
        print(f"  p{p:>3}: {np.percentile(area_ratios, p):.3f}")

    print(f"\n  % de transições com razão fora de [0.5, 2.0]: {((area_ratios < 0.5) | (area_ratios > 2.0)).mean():.2%}")
    print(f"  % de transições com razão fora de [0.6, 1.67]: {((area_ratios < 0.6) | (area_ratios > 1.67)).mean():.2%}")

    print("\nDistribuição de DESLOCAMENTO NORMALIZADO (|center_t - center_{t-1}| / diag):")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  p{p:>3}: {np.percentile(center_shifts, p):.4f}")

    out_csv = MOT_EVAL_DIR / "gt_kinematics_train_split.csv"
    pd.DataFrame({"area_ratio": area_ratios[:len(center_shifts)], "center_shift_norm": center_shifts}).to_csv(out_csv, index=False)
    print(f"\nSalvo em: {out_csv}")


if __name__ == "__main__":
    main()