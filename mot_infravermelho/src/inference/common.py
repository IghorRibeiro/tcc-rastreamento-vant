"""
Funções compartilhadas entre os scripts de inferência/avaliação MOT.
"""

import random
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = PROJECT_ROOT / "dataset" / "MultiUAV_Train"
LABEL_ROOT = TRAIN_ROOT / "original_label_file"
VIDEO_ROOT = TRAIN_ROOT / "TrainVideos"
WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"
MOT_EVAL_DIR = PROJECT_ROOT / "runs" / "mot_eval"
TRACKER_VARIANTS_DIR = PROJECT_ROOT / "configs" / "tracker_variants"

VAL_FRACTION = 0.15
SEED = 42


def seq_to_video_filename(seq_name: str) -> str:
    num = re.search(r"\d+", seq_name).group()
    return f"MultiUAV-{num}.mp4"


def get_val_sequences():
    """Recalcula a mesma lista de sequências de validação usada no
    split_train_val.py (mesma seed => mesmo resultado)."""
    all_seq_dirs = sorted(p.name for p in LABEL_ROOT.iterdir() if p.is_dir())
    random.seed(SEED)
    shuffled = all_seq_dirs.copy()
    random.shuffle(shuffled)
    n_val = max(1, round(VAL_FRACTION * len(shuffled)))
    return sorted(shuffled[:n_val])


def load_mot_file(path: Path) -> pd.DataFrame:
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "a", "b", "c"]
    return pd.read_csv(path, header=None, names=cols)