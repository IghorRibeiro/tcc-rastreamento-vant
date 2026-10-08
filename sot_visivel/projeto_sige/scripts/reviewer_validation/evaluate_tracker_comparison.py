#!/usr/bin/env python3
"""Evaluate local trackers and merge per-sequence wall-clock FPS.

This wrapper uses the project's evaluate_tracking.py for SR, IoU, coverage and
prediction errors, then merges run_summary.csv produced by run_opencv_trackers.py.
It writes per-video and aggregate CSVs with both FPS_from_log and FPS_wall.

Example:
  python scripts/reviewer_validation/evaluate_tracker_comparison.py \
    --methods CSRT=avaliacao_tracking/predicoes/csrt_hibrido \
    KCF=avaliacao_tracking/predicoes/kcf_hibrido \
    MOSSE=avaliacao_tracking/predicoes/mosse_hibrido \
    MedianFlow=avaliacao_tracking/predicoes/medianflow_hibrido \
    --output-dir scripts/reviewer_validation/results_local_trackers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate_tracking import main as evaluate_tracking_main
from evaluate_tracking import parse_method_args, evaluate_video, load_sequence_gt
import evaluate_tracking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--methods", nargs="+", required=True, help="NAME=PREDICTION_DIR pairs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=None, help="Pasta contendo video01_*.jpg; default: datasets/anti_uav_tracking_test_final/images/test")
    parser.add_argument("--labels-dir", type=Path, default=None, help="Pasta contendo video01_*.txt; default: datasets/anti_uav_tracking_test_final/labels/test")
    parser.add_argument("--img-width", type=int, default=1920)
    parser.add_argument("--img-height", type=int, default=1080)
    parser.add_argument(
        "--damping",
        type=float,
        default=1.0,
        help="Mesmo fator de damping do evaluate_tracking.py; use 0.95 para "
             "reproduzir o Kalman Damped oficial em cada rastreador local, "
             "com o codigo identico ao usado no Hibrido.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = (args.images_dir or root / "datasets" / "anti_uav_tracking_test_final" / "images" / "test").expanduser().resolve()
    labels_dir = (args.labels_dir or root / "datasets" / "anti_uav_tracking_test_final" / "labels" / "test").expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"--images-dir não existe ou não é uma pasta: {images_dir}. Substitua o caminho de exemplo pelo caminho real das imagens.")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"--labels-dir não existe ou não é uma pasta: {labels_dir}. Substitua o caminho de exemplo pelo caminho real dos labels.")
    methods = parse_method_args(root, args.methods)
    video_ids = sorted({path.name.split("_")[0] for path in images_dir.glob("video*_*.jpg")})
    if not video_ids:
        raise RuntimeError(f"Nenhuma imagem videoXX_*.jpg encontrada em {images_dir}")
    all_rows = []

    damping_tag = "undamped" if args.damping == 1.0 else f"damped{args.damping:.2f}".replace(".", "")

    for method, prediction_dir in methods.items():
        prediction_dir = prediction_dir.expanduser().resolve()
        if not prediction_dir.is_dir():
            raise FileNotFoundError(f"Diretório de predições ausente para {method}: {prediction_dir}. Execute primeiro o runner correspondente.")
        method_rows = []
        missing_predictions = []
        for video_id in video_ids:
            sample = next(iter(images_dir.glob(f"{video_id}_*.jpg")))
            import cv2
            image = cv2.imread(str(sample))
            if image is None:
                raise RuntimeError(f"Imagem inválida: {sample}")
            height, width = image.shape[:2]
            prediction_path = prediction_dir / f"{video_id}.txt"
            if not prediction_path.exists():
                missing_predictions.append(video_id)
            row = evaluate_video(images_dir, labels_dir, prediction_path, video_id, width, height, damping=args.damping)
            row["method"] = method
            method_rows.append(row)
        if missing_predictions:
            print(f"AVISO {method}: {len(missing_predictions)} arquivos de predição ausentes; eles serão avaliados como sem predição.")
        frame_df = pd.DataFrame(method_rows)
        run_summary_path = prediction_dir / "run_summary.csv"
        if run_summary_path.exists():
            run_df = pd.read_csv(run_summary_path)
            keep = [column for column in ["video", "time_ms_total", "time_ms_mean_all_frames", "fps"] if column in run_df.columns]
            frame_df = frame_df.merge(run_df[keep], on="video", how="left")
            frame_df = frame_df.rename(columns={"fps": "FPS_wall", "time_ms_mean_all_frames": "time_ms_mean_all_frames_wall"})
        else:
            frame_df["FPS_wall"] = pd.NA
            frame_df["time_ms_mean_all_frames_wall"] = pd.NA
        frame_df.to_csv(output_dir / f"per_video_{method.replace(' ', '_').replace('/', '_')}_{damping_tag}.csv", index=False, float_format="%.8f")
        all_rows.append(frame_df)
        print(f"{method} [{damping_tag}]: SR@0.5={pd.to_numeric(frame_df.get('SR@0.5'), errors='coerce').mean():.4f}; IoU={pd.to_numeric(frame_df.get('mean_IoU_present'), errors='coerce').mean():.4f}; FPS_log={pd.to_numeric(frame_df.get('FPS_from_log'), errors='coerce').mean():.2f}; FPS_wall={pd.to_numeric(frame_df.get('FPS_wall'), errors='coerce').mean():.2f}")

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(output_dir / f"all_methods_per_video_with_fps_{damping_tag}.csv", index=False, float_format="%.8f")
    metrics = ["coverage", "SA_legacy_IoU50", "SR@0.5", "mean_IoU_present", "success_AUC", "FPS_from_log", "FPS_wall", "time_ms_mean", "pred_error_H5_mean", "pred_error_H10_mean", "pred_error_H15_mean", "pred_error_H30_mean"]
    summary_rows = []
    for (method, metric) in [(m, metric) for m in methods for metric in metrics if metric in all_df.columns]:
        values = pd.to_numeric(all_df.loc[all_df.method == method, metric], errors="coerce").dropna()
        if values.empty:
            continue
        summary_rows.append({"method": method, "metric": metric, "damping": args.damping, "n_sequences": len(values), "mean": values.mean(), "std_between_sequences": values.std(ddof=1), "median": values.median(), "min": values.min(), "max": values.max()})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / f"summary_local_trackers_{damping_tag}.csv", index=False, float_format="%.8f")
    summary.pivot(index="metric", columns="method", values="mean").to_csv(output_dir / f"summary_local_trackers_pivot_{damping_tag}.csv", float_format="%.8f")
    print(f"Resultados consolidados ({damping_tag}) salvos em {output_dir}")


if __name__ == "__main__":
    main()