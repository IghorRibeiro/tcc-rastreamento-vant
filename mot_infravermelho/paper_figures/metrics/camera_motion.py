import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/inference"))

import cv2
import numpy as np
import pandas as pd
from common import VIDEO_ROOT, get_val_sequences, seq_to_video_filename

FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)


def estimate_camera_motion(video_path: Path):
    """Estima movimento médio da câmera (mediana dos deslocamentos ópticos)."""
    cap = cv2.VideoCapture(str(video_path))
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return None
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_medians = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
        if p0 is None or len(p0) < 10:
            prev_gray = gray
            continue

        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
        if p1 is None:
            prev_gray = gray
            continue

        good_new = p1[st == 1]
        good_old = p0[st == 1]
        if len(good_new) < 10:
            prev_gray = gray
            continue

        disp = good_new - good_old
        mag = np.linalg.norm(disp, axis=1)
        frame_medians.append(float(np.median(mag)))
        prev_gray = gray

    cap.release()
    if not frame_medians:
        return None

    return {
        "mean_motion": float(np.mean(frame_medians)),
        "median_motion": float(np.median(frame_medians)),
        "max_motion": float(np.max(frame_medians)),
        "std_motion": float(np.std(frame_medians)),
    }


def get_camera_motion_dataframe(sequences=None):
    if sequences is None:
        sequences = get_val_sequences()
    rows = []
    for seq in sequences:
        video_path = VIDEO_ROOT / seq_to_video_filename(seq)
        if not video_path.exists():
            continue
        stats = estimate_camera_motion(video_path)
        if stats:
            stats["sequence"] = seq
            rows.append(stats)
    return pd.DataFrame(rows)