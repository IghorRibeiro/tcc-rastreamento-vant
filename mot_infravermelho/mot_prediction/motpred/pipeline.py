"""Processamento por sequencia, com retomada (um marcador .json por sequencia concluida).

Usado pelas etapas 1 (teste rapido), 2 (Experimento A), 3 (Experimento B
principal + referencia visivel de H4) e 4 (varreduras).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .common import (is_done, log, mark_done, read_json, rpath, sige_rastreadores, uav_common, weights_path,
                     write_csv, write_json)
from .data import load_gt, load_mot, trajectory_path
from .drift import drift_analysis, gt_structures_from_sige
from .evaluate import evaluate_trajectory, reference_pairs_for
from .runners import (HitReference, HybridFactory, run_baseline_ref_sequence, run_hybrid_sequence,
                      variant_paths)


# ---------------------------------------------------------------------------
# Trajetorias do Experimento A (salvas ou, se ausentes, geradas)
# ---------------------------------------------------------------------------

def published_traj(cfg: dict, selection: dict, tracker: str, seq: str, base_a: Path) -> Path:
    suffix = selection["rastreadores"][tracker]["selecionado"]
    if suffix:
        return trajectory_path(cfg, suffix, seq)
    return base_a / "generated" / tracker / f"{seq}.txt"


def ensure_generated(cfg: dict, selection: dict, tracker: str, seq: str, base_a: Path, gt_index) -> None:
    """So quando o check_setup nao achou trajetorias completas (criterio 'gerar')."""
    if selection["rastreadores"][tracker]["selecionado"]:
        return
    target = base_a / "generated" / tracker / f"{seq}.txt"
    if target.exists():
        return
    log(f"  gerando trajetoria {tracker}/{seq} com a configuracao do artigo (nao havia arquivo salvo)")
    paths = variant_paths(base_a / "generated_runs", tracker, seq)
    if tracker == "baseline":
        run_baseline_ref_sequence(cfg, cfg["reference"]["baseline"], seq, gt_index, paths)
    else:
        with HitReference(cfg, cfg["reference"]["hit"]) as hit:
            hit.run_sequence(seq, gt_index, paths)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(paths["traj"].read_text(encoding="utf-8"), encoding="utf-8")


def process_expa_sequence(cfg: dict, selection: dict, seq: str, base_a: Path) -> dict | None:
    done = base_a / "done" / f"{seq}.json"
    if is_done(done):
        return None
    t0 = time.perf_counter()
    gt_df, gt_index = load_gt(cfg, seq)
    for tracker in ("baseline", "hit"):
        ensure_generated(cfg, selection, tracker, seq, base_a, gt_index)
    path_base = published_traj(cfg, selection, "baseline", seq, base_a)
    path_hit = published_traj(cfg, selection, "hit", seq, base_a)
    res_base = evaluate_trajectory(cfg, seq, path_base, gt_df, gt_index, base_a / "eval" / "baseline",
                                   "baseline", keep_pairs=True)
    res_hit = evaluate_trajectory(cfg, seq, path_hit, gt_df, gt_index, base_a / "eval" / "hit", "hit",
                                  reference_pairs=res_base["pairs"])
    info = {"seq": seq, "tempo_s": time.perf_counter() - t0}
    for label, res in (("baseline", res_base), ("hit", res_hit)):
        table = res["per_seq"]
        sel = table[(table["mode"] == "clear") & (table["damping"] == 1.0) & (table["H"] == 30)]
        h30 = float(sel["mean_px"].iloc[0]) if len(sel) else float("nan")
        rec = res["record"]
        info[label] = {"H30_px_persistente": h30, "IDF1": rec["mot_idf1"], "MOTA": rec["mot_mota"],
                       "IDs": rec["mot_num_switches"]}
    mark_done(done, info)
    return info


# ---------------------------------------------------------------------------
# Experimento B
# ---------------------------------------------------------------------------

def process_baseline_ref(cfg: dict, seq: str, base_b: Path, max_frames: int | None = None,
                         evaluate: bool = True) -> dict | None:
    rcfg = cfg["reference"]["baseline"]
    paths = variant_paths(base_b, rcfg["name"], seq)
    if is_done(paths["done"]):
        return None
    gt_df, gt_index = load_gt(cfg, seq)
    info = run_baseline_ref_sequence(cfg, rcfg, seq, gt_index, paths, max_frames=max_frames)
    if evaluate:
        t0 = time.perf_counter()
        evaluate_trajectory(cfg, seq, paths["traj"], gt_df, gt_index, paths["eval_dir"], rcfg["name"])
        info["tempo_avaliacao_s"] = time.perf_counter() - t0
    mark_done(paths["done"], info)
    return info


def process_hybrid(cfg: dict, factory: HybridFactory, seq: str, base_b: Path, reference_traj: Path | None,
                   max_frames: int | None = None, evaluate: bool = True) -> dict | None:
    hcfg = factory.hcfg
    paths = variant_paths(base_b, hcfg["name"], seq)
    if is_done(paths["done"]):
        return None
    gt_df, gt_index = load_gt(cfg, seq)
    info = run_hybrid_sequence(cfg, factory, seq, gt_index, paths, max_frames=max_frames)
    if evaluate:
        t0 = time.perf_counter()
        ref_pairs = None
        if reference_traj is not None and Path(reference_traj).exists():
            ref_pairs = reference_pairs_for(cfg, seq, reference_traj, gt_df, gt_index)
        evaluate_trajectory(cfg, seq, paths["traj"], gt_df, gt_index, paths["eval_dir"], hcfg["name"],
                            reference_pairs=ref_pairs)
        anchors = pd.read_csv(paths["anchors"]) if paths["anchors"].exists() else pd.DataFrame()
        curve, summary = drift_analysis(load_mot(cfg, paths["traj"]), anchors, gt_index.by_frame,
                                        gt_index.boxes_by_id, int(hcfg["n_red"]), int(cfg["drift"]["max_age"]),
                                        float(cfg["drift"]["min_anchor_iou"]), last_frame=info["quadros"])
        curve.insert(0, "seq", seq)
        write_csv(curve, paths["eval_dir"] / f"{seq}_drift_curve.csv")
        write_json(paths["eval_dir"] / f"{seq}_drift.json", {"seq": seq, "dominio": "track3_ir", **summary})
        info["tempo_avaliacao_s"] = time.perf_counter() - t0
    mark_done(paths["done"], info)
    return info


def process_hit_ref(cfg: dict, hit: HitReference, seq: str, base_b: Path, max_frames: int | None = None,
                    evaluate: bool = True) -> dict | None:
    rcfg = cfg["reference"]["hit"]
    paths = variant_paths(base_b, rcfg["name"], seq)
    if is_done(paths["done"]):
        return None
    gt_df, gt_index = load_gt(cfg, seq)
    info = hit.run_sequence(seq, gt_index, paths, max_frames=max_frames)
    if evaluate:
        evaluate_trajectory(cfg, seq, paths["traj"], gt_df, gt_index, paths["eval_dir"], rcfg["name"])
    mark_done(paths["done"], info)
    return info


# ---------------------------------------------------------------------------
# H4: referencia visivel (DUT Anti-UAV, hibrido do SIGE instrumentado)
# ---------------------------------------------------------------------------

class DutContext:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO

        from .common import resolve_device

        self.cfg = cfg
        from .dut_sige import find_dut_ground_truth

        self.uc = uav_common(cfg)
        self.sige04 = sige_rastreadores(cfg)
        self.images = rpath(cfg["dut"]["images_dir"])
        self.gt_kind, self.gt_source = find_dut_ground_truth(cfg, self.uc)
        if self.gt_kind is None:
            raise SystemExit(f"DUT: {self.gt_source}")
        self.device = resolve_device(cfg.get("device", "auto"))
        self.detector = YOLO(str(weights_path(cfg, "sige_dut")))

    def videos(self) -> list[str]:
        return self.uc.list_videos(self.images)

    def ground_truth(self, video: str) -> dict:
        """{quadro: caixa ou None}. Rotulos YOLO: convertidos com a resolucao medida no 1o quadro."""
        if self.gt_kind == "oficial":
            return self.uc.read_official_gt(self.gt_source[video]) if video in self.gt_source else {}
        frames = self.uc.frames_of_video(self.images, video)
        size = self.uc.jpeg_size(frames[0]) if frames else None
        return self.uc.read_yolo_labels(self.gt_source, video, size=size)


def process_dut_video(cfg: dict, ctx: DutContext, video: str, base_h4: Path) -> dict | None:
    from .dut_sige import run_hibrido_instrumented, sige_traj_to_mot_df

    uc = ctx.uc
    dcfg = cfg["dut"]
    done = base_h4 / "done" / f"{video}.json"
    if is_done(done):
        return None
    pred_path = base_h4 / "predictions" / f"{video}.txt"
    anchors_path = base_h4 / "anchors" / f"{video}.csv"
    frames_path = base_h4 / "frames" / f"{video}.csv"
    image_paths = uc.frames_of_video(ctx.images, video)
    stats = run_hibrido_instrumented(ctx.sige04, uc, ctx.detector, image_paths, ctx.device,
                                     float(dcfg["init_conf"]), float(dcfg["switch_conf"]), int(dcfg["n_red"]),
                                     str(dcfg["local_tracker"]), pred_path, anchors_path, frames_path)
    gt_frames = ctx.ground_truth(video)
    record = {"video": video, "fonte_gt": ctx.gt_kind, **stats}
    if gt_frames:
        predictions = uc.read_predictions(pred_path)
        record.update({f"sige_{k}": v for k, v in uc.tracking_metrics(gt_frames, predictions).items()})
        for damping in cfg["prediction"]["dampings"]:
            errors = uc.summarize_errors(uc.prediction_errors(uc.centers_from_predictions(predictions),
                                                              uc.centers_from_gt(gt_frames), float(damping)))
            record.update({f"g{float(damping):.2f}_{k}": v for k, v in errors.items()})
        by_frame, by_id, last = gt_structures_from_sige(gt_frames)
        anchors = pd.read_csv(anchors_path)
        curve, summary = drift_analysis(sige_traj_to_mot_df(pred_path), anchors, by_frame, by_id,
                                        int(dcfg["n_red"]), int(cfg["drift"]["max_age"]),
                                        float(cfg["drift"]["min_anchor_iou"]), last_frame=last)
        curve.insert(0, "seq", video)
        write_csv(curve, base_h4 / "eval" / f"{video}_drift_curve.csv")
        write_json(base_h4 / "eval" / f"{video}_drift.json", {"seq": video, "dominio": "dut_visivel", **summary})
    write_json(base_h4 / "eval" / f"{video}_sige.json", record)
    mark_done(done, {"fps": stats["fps"]})
    return record


def stage_marker(base: Path, name: str) -> Path:
    return base / f"_DONE_{name}.json"


def read_done_info(path: Path) -> dict:
    return read_json(path) if Path(path).exists() else {}
