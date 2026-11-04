"""Identity-aware trajectory-prediction error on the Track 3 sequences (multiple targets).

For every sequence it evaluates the tracker output against the ground truth (CLEAR MOT, IDF1), applies the
external Kalman filter (persistent and damped) to each track and measures the error at H5, H10, H15 and H30 against
the target the track was following at the moment of the prediction.

    python etapa3_predicao/predict_mot.py --variant baseline \\
        --trajectories <runs>/baseline --gt-root <data-root>/MultiUAV_Train/original_label_file \\
        --out-dir results_mot/baseline

Inputs: ``<trajectories>/<seq>.txt`` in MOTChallenge format and ``<gt-root>/<seq>/gt/gt.txt``.
Outputs (columns as in results/): ``tracking_per_seq.csv`` and ``prediction_per_seq.csv``, plus ``pairs/<seq>.csv.gz``
when ``--keep-pairs`` is given (needed for the common-support comparison between two variants).
The data root can also be set with the environment variable TCC_DATA_ROOT.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trajpred.errors import GroundTruth, aggregate_sequence, pair_errors  # noqa: E402
from trajpred.matching import evaluate_sequence, load_mot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def evaluate_variant(variant: str, trajectories: Path, gt_root: Path, sequences: list[str], horizons, dampings,
                     keep_pairs_dir: Path | None = None):
    tracking_rows, prediction_rows = [], []
    for seq in sequences:
        gt_df = load_mot(gt_root / seq / "gt" / "gt.txt")
        pred_df = load_mot(trajectories / f"{seq}.txt")
        metrics, corr = evaluate_sequence(gt_df, pred_df, variant)
        gt = GroundTruth(gt_df)
        pairs, counts = pair_errors(pred_df, gt, corr, horizons, dampings)
        table = aggregate_sequence(pairs, counts)
        table.insert(0, "seq", seq)
        table.insert(0, "variant", variant)
        prediction_rows.append(table)
        tracking_rows.append({
            "variant": variant, "seq": seq, "mota": metrics["mota"], "idf1": metrics["idf1"],
            "idtp": metrics["idtp"], "id_switches": metrics["num_switches"], "fp": metrics["num_false_positives"],
            "fn": metrics["num_misses"], "gt_objects": metrics["num_objects"],
            "predictions": metrics["num_predictions"], "mean_density": gt.mean_density,
            "idf1_matches_library": metrics["idf1_matches"]})
        if keep_pairs_dir is not None:
            keep_pairs_dir.mkdir(parents=True, exist_ok=True)
            pairs.to_csv(keep_pairs_dir / f"{seq}.csv.gz", index=False)
        print(f"  {seq}: MOTA {metrics['mota']:.4f}  IDF1 {metrics['idf1']:.4f}  IDs {int(metrics['num_switches'])}")
    return pd.DataFrame(tracking_rows), pd.concat(prediction_rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, help="label written to the output (baseline, hit_tracker, ...)")
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, default=None,
                        help="folder with <seq>/gt/gt.txt (default: $TCC_DATA_ROOT/MultiUAV_Train/original_label_file)")
    parser.add_argument("--sequences", nargs="*", default=None, help="default: every <seq>.txt in --trajectories")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "prediction.yaml")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--keep-pairs", action="store_true")
    args = parser.parse_args()

    gt_root = args.gt_root
    if gt_root is None:
        data_root = os.environ.get("TCC_DATA_ROOT")
        if not data_root:
            raise SystemExit("pass --gt-root or set TCC_DATA_ROOT")
        gt_root = Path(data_root) / "MultiUAV_Train" / "original_label_file"
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    horizons = [int(h) for h in cfg["horizons"]]
    dampings = [float(cfg["damping"]["persistent"]), float(cfg["damping"]["damped"])]
    sequences = args.sequences or sorted(p.stem for p in args.trajectories.glob("*.txt"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracking, prediction = evaluate_variant(args.variant, args.trajectories, gt_root, sequences, horizons, dampings,
                                            args.out_dir / "pairs" if args.keep_pairs else None)
    tracking.to_csv(args.out_dir / "tracking_per_seq.csv", index=False, float_format="%.10g")
    prediction.to_csv(args.out_dir / "prediction_per_seq.csv", index=False, float_format="%.10g")
    print(f"written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
