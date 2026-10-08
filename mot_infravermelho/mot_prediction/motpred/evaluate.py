"""Avaliacao de uma trajetoria MOT: metricas MOT + erro de predicao com identidade.

Grava, por (rotulo, sequencia), apenas arquivos pequenos:
    <dir>/<seq>_mot.json      metricas MOT, densidade, lacunas, checagem do IDF1
    <dir>/<seq>_errors.csv    por (modo, gamma, H): n, media/dp/mediana/p95 (px e normalizado), descartes
    <dir>/<seq>_tracks.csv    por (modo, gamma, H, trilha): n, media, dp
    <dir>/<seq>_common.csv    (opcional) suporte comum com um rastreador de referencia
"""

from __future__ import annotations

import time
from pathlib import Path

from .common import sha256_file, uav_common, write_csv, write_json
from .data import load_mot
from .mot_eval import evaluate_mot_sequence
from .prediction import aggregate_sequence, aggregate_tracks, common_support, pair_errors


def evaluate_trajectory(cfg: dict, seq: str, traj_path: Path, gt_df, gt_index, out_dir: Path,
                        label: str, reference_pairs=None, keep_pairs: bool = False) -> dict:
    """Avalia e grava; devolve dict com metricas e (se keep_pairs) os pares."""
    t0 = time.perf_counter()
    uc = uav_common(cfg)
    horizons = tuple(int(h) for h in cfg["prediction"]["horizons"])
    dampings = tuple(float(d) for d in cfg["prediction"]["dampings"])
    pred_df = load_mot(cfg, traj_path)

    metrics, corr = evaluate_mot_sequence(gt_df, pred_df, seq, max_iou=float(cfg["track3"]["iou_threshold"]))
    pairs, counts, gaps = pair_errors(uc, pred_df, gt_index, corr, horizons, dampings)
    per_seq = aggregate_sequence(pairs, counts)
    per_track = aggregate_tracks(pairs)

    out_dir = Path(out_dir)
    per_seq.insert(0, "seq", seq)
    per_seq.insert(0, "rotulo", label)
    write_csv(per_seq, out_dir / f"{seq}_errors.csv")
    per_track.insert(0, "seq", seq)
    per_track.insert(0, "rotulo", label)
    write_csv(per_track, out_dir / f"{seq}_tracks.csv")

    if reference_pairs is not None:
        common = common_support(reference_pairs, pairs, mode="clear")
        common.insert(0, "seq", seq)
        common.insert(0, "rotulo", label)
        write_csv(common, out_dir / f"{seq}_common.csv")

    record = {
        "rotulo": label,
        "seq": seq,
        "trajetoria": str(traj_path),
        "trajetoria_sha256": sha256_file(traj_path),
        "densidade_media": gt_index.mean_density,
        "densidade_max": gt_index.max_density,
        "quadros_anotados": gt_index.n_frames_annotated,
        **{f"mot_{k}": v for k, v in metrics.items()},
        **{f"lacunas_{k}": v for k, v in gaps.items()},
        "tempo_avaliacao_s": time.perf_counter() - t0,
    }
    write_json(out_dir / f"{seq}_mot.json", record)
    result = {"record": record, "per_seq": per_seq}
    if keep_pairs:
        result["pairs"] = pairs
    return result


def reference_pairs_for(cfg: dict, seq: str, traj_path: Path, gt_df, gt_index):
    """Pares da linha de base (para suporte comum), sem gravar nada."""
    uc = uav_common(cfg)
    horizons = tuple(int(h) for h in cfg["prediction"]["horizons"])
    dampings = tuple(float(d) for d in cfg["prediction"]["dampings"])
    pred_df = load_mot(cfg, traj_path)
    _, corr = evaluate_mot_sequence(gt_df, pred_df, seq, max_iou=float(cfg["track3"]["iou_threshold"]))
    pairs, _, _ = pair_errors(uc, pred_df, gt_index, corr, horizons, dampings)
    return pairs
