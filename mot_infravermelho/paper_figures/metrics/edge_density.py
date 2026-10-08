import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/inference"))

import cv2
import numpy as np
import pandas as pd
from common import VIDEO_ROOT, get_val_sequences, seq_to_video_filename


def estimate_edge_density(video_path: Path):
    """Calcula a proporção média de pixels de borda (Canny) por frame."""
    cap = cv2.VideoCapture(str(video_path))
    edge_ratios = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        ratio = np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1])
        edge_ratios.append(ratio)
    cap.release()
    if not edge_ratios:
        return None
    return {
        "mean_edge_density": float(np.mean(edge_ratios)),
        "median_edge_density": float(np.median(edge_ratios)),
        "max_edge_density": float(np.max(edge_ratios)),
        "std_edge_density": float(np.std(edge_ratios)),
    }


def get_edge_density_dataframe(sequences=None):
    if sequences is None:
        sequences = get_val_sequences()
    rows = []
    for seq in sequences:
        video_path = VIDEO_ROOT / seq_to_video_filename(seq)
        if not video_path.exists():
            continue
        stats = estimate_edge_density(video_path)
        if stats:
            stats["sequence"] = seq
            rows.append(stats)
    return pd.DataFrame(rows)