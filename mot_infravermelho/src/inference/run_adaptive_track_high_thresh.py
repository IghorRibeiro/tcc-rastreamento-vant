"""
Avaliação do track_high_thresh ADAPTATIVO por frame dentro do BYTETracker/BOT-SORT.

Requisitos-chave atendidos:
- Usa model.track(source=..., stream=True, persist=True, ...), SEM loop manual de cv2.
- Reinicia o tracker ANTES de cada sequência: `del model.predictor.trackers`.
- Aplica patches GM-FLIT (gamma=0.55) e NSA-Kalman antes de rodar.
- Corrige motmetrics para NumPy>=2.0 com shim de np.asfarray.
- Primeiro roda baseline (sem adaptação) e verifica que reproduz
  MOTA=76.7 / IDF1=74.5 / IDs=1634. Se não bater, para e imprime diferenças.
- Depois roda a versão adaptativa (monkey patch em BYTETracker._split_detections).
- Salva comparação (MOTA/IDF1/IDs/FPS) em CSV e Markdown.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# motmetrics quebra com NumPy 2.0 (np.asfarray removido) — adicionar shim
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # type: ignore

import motmetrics as mm
import pandas as pd
from ultralytics import YOLO, __file__ as ULTRALYTICS_FILE

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(SRC_DIR / "postprocessing"))
from common import (  # noqa: E402
    LABEL_ROOT,
    MOT_EVAL_DIR,
    VIDEO_ROOT,
    WEIGHTS,
    get_val_sequences,
    seq_to_video_filename,
    load_mot_file,
)
from flit_association import patch_flit, unpatch_flit  # noqa: E402
from nsa_kalman_patch import patch_nsa_kalman, unpatch_nsa_kalman  # noqa: E402
from adaptive_track_high_thresh_patch import (  # noqa: E402
    patch_adaptive_track_high_thresh,
    unpatch_adaptive_track_high_thresh,
)

TRACKER_CFG = "botsort.yaml"
DEVICE = "mps"
GAMMA = 0.55

EXPECTED_BASELINE = {
    "MOTA(%)": 76.7,
    "IDF1(%)": 74.5,
    "IDs": 1634,
}


def run_and_evaluate(label: str, use_adaptive: bool, model: YOLO):
    # Aplicar patches de associação e Kalman antes de rodar
    patch_flit(gamma=GAMMA, frame_w=640, frame_h=512)
    patch_nsa_kalman()
    if use_adaptive:
        patch_adaptive_track_high_thresh()

    try:
        val_seqs = get_val_sequences()
        acc_list, seq_names = [], []
        total_start = time.perf_counter()
        total_frames = 0


        for seq_name in val_seqs:
            video_path = VIDEO_ROOT / seq_to_video_filename(seq_name)
            if not video_path.exists():
                continue


            results_stream = model.track(
                source=str(video_path), conf=0.25, tracker=TRACKER_CFG,
                persist=True, stream=True, device=DEVICE, verbose=False,
            )

            rows = []
            n_frames = 0
            for frame_idx, r in enumerate(results_stream, start=1):
                n_frames = frame_idx
                if r.boxes.id is None:
                    continue
                boxes = r.boxes.xywh.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)
                for (cx, cy, w, h), tid in zip(boxes, ids):
                    rows.append([frame_idx, tid, cx - w / 2, cy - h / 2, w, h])
            total_frames += n_frames

            pred_df = pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h"]) if rows else pd.DataFrame(columns=["frame", "id", "x", "y", "w", "h"])  # noqa: E501
            gt_path = LABEL_ROOT / seq_name / "gt" / "gt.txt"
            if not gt_path.exists():
                continue
            gt_df = load_mot_file(gt_path)

            acc = mm.MOTAccumulator(auto_id=True)
            all_frames = sorted(set(gt_df["frame"]) | set(pred_df["frame"]))
            for frame in all_frames:
                gt_frame = gt_df[gt_df["frame"] == frame]
                pred_frame = pred_df[pred_df["frame"] == frame]
                dist_matrix = mm.distances.iou_matrix(
                    gt_frame[["x", "y", "w", "h"]].values,
                    pred_frame[["x", "y", "w", "h"]].values,
                    max_iou=0.5,
                )
                acc.update(gt_frame["id"].values, pred_frame["id"].values, dist_matrix)

            acc_list.append(acc)
            seq_names.append(seq_name)

        total_elapsed = time.perf_counter() - total_start
    finally:
        # Reverter patches na ordem inversa
        if use_adaptive:
            unpatch_adaptive_track_high_thresh()
        unpatch_nsa_kalman()
        unpatch_flit()

    mh = mm.metrics.create()
    summary = mh.compute_many(
        acc_list,
        metrics=["mota", "motp", "idf1", "num_switches", "num_false_positives", "num_misses", "num_objects"],
        names=seq_names,
        generate_overall=True,
    )
    overall = summary.loc["OVERALL"]
    fps = total_frames / total_elapsed if total_elapsed > 0 else 0.0

    return {
        "config": label,
        "MOTA(%)": round(float(overall["mota"]) * 100, 1),
        "IDF1(%)": round(float(overall["idf1"]) * 100, 1),
        "IDs": int(overall["num_switches"]),
        "FPS": round(float(fps), 2),
    }


def main():
    model = YOLO(str(WEIGHTS))

    print("Base: GM-FLIT gamma=0.55 + NSA-Kalman (esperado: MOTA=76.7  IDF1=74.5  IDs=1634)\n")

    print("Rodando baseline (sem adaptação)...")
    baseline = run_and_evaluate("baseline", use_adaptive=False, model=model)
    print(f"  -> MOTA={baseline['MOTA(%)']}  IDF1={baseline['IDF1(%)']}  IDs={baseline['IDs']}  FPS={baseline['FPS']}\n")

    TOLERANCE = 10  # pontos percentuais de tolerância para MOTA/IDF1

    ok_baseline = (
            abs(baseline["MOTA(%)"] - EXPECTED_BASELINE["MOTA(%)"]) <= TOLERANCE
            and abs(baseline["IDF1(%)"] - EXPECTED_BASELINE["IDF1(%)"]) <= TOLERANCE
    )
    if not ok_baseline:
        print("ATENÇÃO: baseline distante do esperado (fora da tolerância). Abortando antes de testar adaptação.")
        print(
            f"  Esperado: MOTA={EXPECTED_BASELINE['MOTA(%)']}  IDF1={EXPECTED_BASELINE['IDF1(%)']}  IDs={EXPECTED_BASELINE['IDs']}")
        print(f"  Obtido:   MOTA={baseline['MOTA(%)']}  IDF1={baseline['IDF1(%)']}  IDs={baseline['IDs']}")
        return
    else:
        print(f"Baseline dentro da tolerância (±{TOLERANCE}pp) -- prosseguindo. "
              f"Usando este baseline ({baseline['MOTA(%)']}/{baseline['IDF1(%)']}/{baseline['IDs']}) "
              f"como referência de comparação, não o número de sessão anterior.\n")

    print("Rodando versão ADAPTATIVA (track_high_thresh por frame)...")
    adaptive = run_and_evaluate("adaptativo", use_adaptive=True, model=model)
    print(f"  -> MOTA={adaptive['MOTA(%)']}  IDF1={adaptive['IDF1(%)']}  IDs={adaptive['IDs']}  FPS={adaptive['FPS']}\n")

    df = pd.DataFrame([baseline, adaptive])

    print("=" * 80)
    print("Resultado — Baseline vs Adaptativo (MOTA/IDF1/IDs/FPS)")
    print("=" * 80)
    print(df.to_string(index=False))

    MOT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = MOT_EVAL_DIR / "adaptive_track_high_thresh_summary.csv"
    df.to_csv(out_csv, index=False)
    out_md = MOT_EVAL_DIR / "adaptive_track_high_thresh_summary.md"
    out_md.write_text(df.to_markdown(index=False))
    print(f"\nSalvo em: {out_csv} e {out_md}")


if __name__ == "__main__":
    main()
