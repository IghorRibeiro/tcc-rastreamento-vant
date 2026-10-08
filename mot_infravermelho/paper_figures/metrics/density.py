import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/inference"))

import pandas as pd
from common import LABEL_ROOT, get_val_sequences, load_mot_file


def compute_density_stats(seq_name: str):
    """Calcula densidade média, max, min, std por frame."""
    gt_path = LABEL_ROOT / seq_name / "gt" / "gt.txt"
    if not gt_path.exists():
        return None
    df = load_mot_file(gt_path)
    counts = df.groupby("frame")["id"].count()
    return {
        "sequence": seq_name,
        "mean_density": counts.mean(),
        "max_density": counts.max(),
        "min_density": counts.min(),
        "std_density": counts.std(),
    }


def get_density_dataframe(sequences=None):
    if sequences is None:
        sequences = get_val_sequences()
    rows = [compute_density_stats(s) for s in sequences]
    df = pd.DataFrame([r for r in rows if r is not None])
    return df