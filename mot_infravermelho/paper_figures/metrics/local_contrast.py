import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/inference"))

import cv2
import numpy as np
import pandas as pd
from common import VIDEO_ROOT, get_val_sequences, seq_to_video_filename


def estimate_local_contrast(video_path: Path):
    """Calcula o desvio padrão médio dos pixels (RMS contrast) por frame."""
    cap = cv2.VideoCapture(str(video_path))
    contrasts = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrasts.append(float(np.std(gray)))
    cap.release()
    if not contrasts:
        return None
    return {
        "mean_contrast": float(np.mean(contrasts)),
        "median_contrast": float(np.median(contrasts)),
        "max_contrast": float(np.max(contrasts)),
        "std_contrast": float(np.std(contrasts)),
    }


def get_contrast_dataframe(sequences=None):
    if sequences is None:
        sequences = get_val_sequences()
    rows = []
    for seq in sequences:
        video_path = VIDEO_ROOT / seq_to_video_filename(seq)
        if not video_path.exists():
            continue
        stats = estimate_local_contrast(video_path)
        if stats:
            stats["sequence"] = seq
            rows.append(stats)
    return pd.DataFrame(rows)