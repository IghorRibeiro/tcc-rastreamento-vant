"""Trajectory-prediction error on the DUT Anti-UAV tracking sequences (single target).

Applies the external constant-velocity Kalman filter, persistent (gamma = 1) and damped (gamma = 0.95), to the
trajectories saved by one tracker and writes the error at H5, H10, H15 and H30 for every video.

    python etapa3_predicao/predict_sot.py --method CSRT \\
        --predictions <sige>/avaliacao_tracking/predicoes/hibrido_csrt \\
        --gt-dir <sige>/hitratepos/Anti-UAV-Tracking-V0GT --out results_csrt.csv

Trajectories come from the SIGE 2026 code (etapa1_alvo_unico/README.md). The columns match
results/sot_prediction_per_video.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trajpred.sot import per_video_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", required=True, help="label written to the output (e.g. CSRT, BoT-SORT)")
    parser.add_argument("--predictions", type=Path, required=True, help="folder with videoNN.txt prediction files")
    parser.add_argument("--gt-dir", type=Path, required=True, help="folder with videoNN_gt.txt official annotations")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    videos = {}
    for pred in sorted(args.predictions.glob("video*.txt")):
        gt = args.gt_dir / f"{pred.stem}_gt.txt"
        if gt.exists():
            videos[pred.stem] = (pred, gt)
    if not videos:
        raise SystemExit(f"no videoNN.txt with a matching ground truth in {args.predictions}")
    table = per_video_table(args.method, videos)
    table.to_csv(args.out, index=False, float_format="%.10g")
    print(f"{len(videos)} videos, {len(table)} rows written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
