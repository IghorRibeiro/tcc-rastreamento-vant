"""
Estima o movimento de câmera por sequência via optical flow esparso
(Lucas-Kanade), usando a mediana do deslocamento dos pontos rastreados
como proxy do movimento de fundo/câmera (robusta à presença de poucos
pontos sobre os drones, que se movem de forma diferente do fundo).

Cruza o resultado com o desempenho de tracking já calculado, testando
se o movimento de câmera explica a queda de desempenho -- e se essa
correlação é mais fraca quando o GMC está ativo (BoT-SORT) do que
quando não está (ByteTrack), o que confirmaria que o GMC está de fato
compensando o efeito.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from common import MOT_EVAL_DIR, VIDEO_ROOT, get_val_sequences, seq_to_video_filename  # noqa: E402

OUT_DIR = MOT_EVAL_DIR

FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def estimate_camera_motion(video_path: Path):
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

        p1, st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
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
        frame_medians.append(float(np.median(mag)))  # mediana = robusta ao pequeno nº de pontos sobre drones

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

    print("Estimando movimento de câmera por sequência (pode levar alguns minutos)...\n")
    motion_rows = []
    for seq_name in val_seqs:
        video_path = VIDEO_ROOT / seq_to_video_filename(seq_name)
        if not video_path.exists():
            print(f"[AVISO] vídeo não encontrado para {seq_name}, pulando")
            continue
        stats = estimate_camera_motion(video_path)
        if stats is None:
            continue
        stats["sequence"] = seq_name
        motion_rows.append(stats)
        print(f"{seq_name}: motion_mean={stats['mean_motion']:.3f}px  motion_median={stats['median_motion']:.3f}px")

    motion_df = pd.DataFrame(motion_rows)
    motion_csv = OUT_DIR / "camera_motion_estimates.csv"
    motion_df.to_csv(motion_csv, index=False)
    print(f"\nEstimativas salvas em: {motion_csv}\n")

    # Ranking: mais parado -> mais movido
    ranked = motion_df.sort_values("mean_motion")
    print("Sequências mais ESTÁTICAS (menor movimento estimado):")
    print(ranked[["sequence", "mean_motion"]].head(5).to_string(index=False))
    print("\nSequências mais MOVIDAS (maior movimento estimado):")
    print(ranked[["sequence", "mean_motion"]].tail(5).to_string(index=False))

    # Correlação: movimento de câmera x desempenho, COM GMC (botsort) e SEM GMC (bytetrack)
    print("\n" + "=" * 70)
    print("Correlação: movimento de câmera x desempenho")
    print("=" * 70)

    for tracker_name, label in [("botsort", "BoT-SORT (GMC ativo)"), ("bytetrack", "ByteTrack (sem GMC)")]:
        results_df = load_tracker_results(tracker_name)
        if results_df.empty:
            continue
        merged = motion_df.merge(results_df, on="sequence", how="inner")

        r_mota = pearson_r(merged["mean_motion"].values, merged["MOTA(%)"].values)
        r_idf1 = pearson_r(merged["mean_motion"].values, merged["IDF1(%)"].values)
        r_ids = pearson_r(merged["mean_motion"].values, merged["IDs"].values)

        print(f"\n[{label}]")
        print(f"  MOTA:  r = {r_mota:+.3f}")
        print(f"  IDF1:  r = {r_idf1:+.3f}")
        print(f"  IDs:   r = {r_ids:+.3f}")

    out_all = OUT_DIR / "camera_motion_vs_performance.csv"
    combined = motion_df.copy()
    for tracker_name in ["botsort", "bytetrack"]:
        results_df = load_tracker_results(tracker_name)
        if results_df.empty:
            continue
        combined = combined.merge(
            results_df.rename(columns={c: f"{c}_{tracker_name}" for c in results_df.columns if c != "sequence"}),
            on="sequence",
            how="left",
        )
    combined.to_csv(out_all, index=False)
    print(f"\nCSV consolidado salvo em: {out_all}")


if __name__ == "__main__":
    main()