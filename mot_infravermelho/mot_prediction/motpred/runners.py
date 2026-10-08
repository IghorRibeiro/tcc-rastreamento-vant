"""Execucao por sequencia: hibrido MOT e rodadas de referencia com tempo por quadro.

Todas as rodadas gravam o MESMO formato:
    traj/<variante>/<seq>.txt     MOTChallenge: frame,id,x,y,w,h,conf,-1,-1,-1
    frames/<variante>/<seq>.csv   um registro por quadro (tempo, n_gt, ativos, eventos)
    anchors/<variante>/<seq>.csv  eventos de ancoragem (so o hibrido; usado em H4)

Tempo por quadro: time.perf_counter() em volta do processamento do quadro
(deteccao + rastreamento/associacao), sem decodificacao do video e sem E/S,
depois de aquecer o detector (HYPOTHESES.md, 2.9).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (atomic_write_text, lafusion_flit, lafusion_nsa, lafusion_sahi_runner, log,
                     resolve_device, sige_rastreadores, weights_path, write_csv)
from .data import iter_video_frames, video_path
from .hybrid import HybridMOT, detections_from_sahi, detections_from_yolo

FRAME_COLUMNS = ["frame", "time_ms", "n_gt", "n_active", "n_lost", "detect", "n_dets", "births",
                 "reanchors", "reactivations", "failures", "exceptions", "init_failures",
                 "deaths_lost", "deaths_unconfirmed"]


def variant_paths(base: Path, variant: str, seq: str) -> dict:
    return {
        "traj": base / "traj" / variant / f"{seq}.txt",
        "frames": base / "frames" / variant / f"{seq}.csv",
        "anchors": base / "anchors" / variant / f"{seq}.csv",
        "eval_dir": base / "eval" / variant,
        "done": base / "done" / variant / f"{seq}.json",
    }


def _write_outputs(paths: dict, lines: list[str], frame_rows: list[dict], anchor_rows: list | None) -> None:
    atomic_write_text(paths["traj"], "\n".join(lines))
    write_csv(pd.DataFrame(frame_rows, columns=FRAME_COLUMNS), paths["frames"])
    if anchor_rows is not None:
        write_csv(pd.DataFrame(anchor_rows, columns=["id", "frame", "x", "y", "w", "h", "conf", "kind"]),
                  paths["anchors"])


def _summary(frame_rows: list[dict], elapsed: float) -> dict:
    times = np.asarray([r["time_ms"] for r in frame_rows], dtype=float)
    mean_ms = float(times.mean()) if times.size else float("nan")
    return {
        "quadros": int(times.size),
        "tempo_medio_ms": mean_ms,
        "fps": 1000.0 / mean_ms if mean_ms and mean_ms > 0 else float("nan"),
        "tempo_total_s": float(elapsed),
        "chamadas_detector": int(sum(r.get("detect", 0) for r in frame_rows)),
    }


# ---------------------------------------------------------------------------
# Hibrido MOT
# ---------------------------------------------------------------------------

class HybridFactory:
    """Carrega detector, custo GM e rastreador local UMA vez por variante."""

    def __init__(self, cfg: dict, hcfg: dict):
        from ultralytics import YOLO
        from ultralytics.trackers.utils import matching

        self.cfg = cfg
        self.hcfg = hcfg
        self.device = resolve_device(cfg.get("device", "auto"))
        self.weights = weights_path(cfg, hcfg["detector_weights"])
        if not self.weights.exists():
            raise SystemExit(f"Pesos do detector nao encontrados: {self.weights}")
        self.sahi = bool(hcfg.get("sahi", {}).get("enabled", False))
        conf = float(hcfg["det_conf"])
        if self.sahi:
            runner = lafusion_sahi_runner(cfg)
            from sahi import AutoDetectionModel

            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=str(self.weights), confidence_threshold=conf,
                device=f"cuda:{self.device}" if self.device.isdigit() else self.device)
            size = int(hcfg["sahi"]["slice"])
            overlap = float(hcfg["sahi"]["overlap"])
            build = runner.build_sahi_detections
            self.detect_fn = lambda frame: detections_from_sahi(build, self._sahi_model, frame, size, overlap, conf)
        else:
            self._yolo = YOLO(str(self.weights))
            self.detect_fn = lambda frame: detections_from_yolo(self._yolo, frame, self.device, conf)
        track3 = cfg["track3"]
        self.get_dists = lafusion_flit(cfg).make_flit_get_dists(
            gamma=float(hcfg["association"]["gamma"]), frame_w=int(track3["frame_w"]),
            frame_h=int(track3["frame_h"]))
        self.linear_assignment = matching.linear_assignment
        self.make_local_tracker = sige_rastreadores(cfg).make_local_tracker
        # falha cedo se o OpenCV nao tiver o rastreador local pedido
        self.make_local_tracker(str(hcfg["local_tracker"]))

    def warmup(self) -> None:
        h, w = int(self.cfg["track3"]["frame_h"]), int(self.cfg["track3"]["frame_w"])
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(int(self.hcfg.get("warmup_calls", 2))):
            self.detect_fn(blank)

    def new_tracker(self) -> HybridMOT:
        return HybridMOT(self.hcfg, self.detect_fn, self.make_local_tracker, self.get_dists,
                         self.linear_assignment)


def run_hybrid_sequence(cfg: dict, factory: HybridFactory, seq: str, gt_index, paths: dict,
                        max_frames: int | None = None) -> dict:
    tracker = factory.new_tracker()
    factory.warmup()
    lines, frame_rows, anchor_rows = [], [], []
    start_all = time.perf_counter()
    first = True
    for index, frame in iter_video_frames(video_path(cfg, seq), max_frames=max_frames):
        if first:
            h, w = frame.shape[:2]
            if (w, h) != (int(cfg["track3"]["frame_w"]), int(cfg["track3"]["frame_h"])):
                log(f"AVISO {seq}: quadro {w}x{h} difere de track3.frame_w/h do config "
                    "(normalizacao do custo GM usa o config, como o HIT-Tracker)")
            first = False
        t0 = time.perf_counter()
        outputs, stats, anchors = tracker.step(index, frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        for tid, x, y, w, h, conf in outputs:
            lines.append(f"{index},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.6f},-1,-1,-1")
        frame_rows.append({"frame": index, "time_ms": elapsed_ms, "n_gt": gt_index.n_gt(index), **stats})
        anchor_rows.extend(anchors)
    info = _summary(frame_rows, time.perf_counter() - start_all)
    info["trilhas_criadas"] = tracker.next_id - 1
    _write_outputs(paths, lines, frame_rows, anchor_rows)
    return info


# ---------------------------------------------------------------------------
# Referencia 1: linha de base (YOLO26n + BoT-SORT nativo, deteccao em todo quadro)
# Origem: src/inference/run_tracker_on_val.py (conf 0,25, botsort.yaml, persist=True)
# ---------------------------------------------------------------------------

def run_baseline_ref_sequence(cfg: dict, rcfg: dict, seq: str, gt_index, paths: dict,
                              max_frames: int | None = None) -> dict:
    from ultralytics import YOLO

    device = resolve_device(cfg.get("device", "auto"))
    model = YOLO(str(weights_path(cfg, rcfg["weights"])))   # modelo novo por sequencia = estado limpo
    conf = float(rcfg["conf"])
    tracker_cfg = str(rcfg["tracker_cfg"])
    h, w = int(cfg["track3"]["frame_h"]), int(cfg["track3"]["frame_w"])
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(2):                                        # aquecimento, fora do cronometro
        model.track(blank, persist=True, tracker=tracker_cfg, conf=conf, device=device, verbose=False)
    for trk in getattr(model.predictor, "trackers", None) or []:
        if hasattr(trk, "reset"):
            trk.reset()                                       # zera estado, GMC e contador de ids

    lines, frame_rows = [], []
    start_all = time.perf_counter()
    for index, frame in iter_video_frames(video_path(cfg, seq), max_frames=max_frames):
        t0 = time.perf_counter()
        result = model.track(frame, persist=True, tracker=tracker_cfg, conf=conf, device=device,
                             verbose=False)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_out = 0
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu().numpy()            # cx, cy, w, h (como run_tracker_on_val)
            ids = result.boxes.id.cpu().numpy().astype(int)
            for (cx, cy, bw, bh), tid in zip(boxes, ids):
                x = cx - bw / 2
                y = cy - bh / 2
                lines.append(f"{index},{tid},{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},1,-1,-1,-1")
                n_out += 1
        frame_rows.append({"frame": index, "time_ms": elapsed_ms, "n_gt": gt_index.n_gt(index),
                           "n_active": n_out, "n_lost": 0, "detect": 1})
    info = _summary(frame_rows, time.perf_counter() - start_all)
    _write_outputs(paths, lines, frame_rows, None)
    return info


# ---------------------------------------------------------------------------
# Referencia 2: HIT-Tracker (HardMining + SAHI512 + BoT-SORT + GM-FLIT + NSA-Kalman)
# Origem: src/inference/run_sahi_flit_nsa.py::run_tracker_on_sequence, copiado com
# tempo por quadro; patches aplicados antes de instanciar o BOTSORT e removidos no fim.
# ---------------------------------------------------------------------------

class HitReference:
    def __init__(self, cfg: dict, rcfg: dict):
        self.cfg = cfg
        self.rcfg = rcfg
        self.runner = lafusion_sahi_runner(cfg)
        self.flit = lafusion_flit(cfg)
        self.nsa = lafusion_nsa(cfg)
        self.device = resolve_device(cfg.get("device", "auto"))
        self.torch_device = f"cuda:{self.device}" if self.device.isdigit() else self.device
        from sahi import AutoDetectionModel

        self.det_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(weights_path(cfg, rcfg["weights"])),
            confidence_threshold=float(rcfg["conf"]), device=self.torch_device)
        self.tracker_cfg_path = self.runner.resolve_tracker_cfg(str(rcfg["tracker_cfg"]))
        self.patched = False

    def __enter__(self):
        self.flit.patch_flit(gamma=float(self.rcfg["gamma"]), frame_w=int(self.cfg["track3"]["frame_w"]),
                             frame_h=int(self.cfg["track3"]["frame_h"]))
        self.nsa.patch_nsa_kalman()
        self.patched = True
        return self

    def __exit__(self, *exc):
        if self.patched:
            self.nsa.unpatch_nsa_kalman()
            self.flit.unpatch_flit()
            self.patched = False
        return False

    def run_sequence(self, seq: str, gt_index, paths: dict, max_frames: int | None = None) -> dict:
        import torch
        from ultralytics.engine.results import Results
        from ultralytics.trackers import BOTSORT

        rcfg = self.rcfg
        size, overlap, conf = int(rcfg["sahi_slice"]), float(rcfg["sahi_overlap"]), float(rcfg["conf"])
        h, w = int(self.cfg["track3"]["frame_h"]), int(self.cfg["track3"]["frame_w"])
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(2):
            self.runner.build_sahi_detections(frame=blank, det_model=self.det_model, slice_h=size,
                                              slice_w=size, overlap=overlap, conf_thresh=conf)
        tracker = BOTSORT(args=self.runner.load_tracker_config(self.tracker_cfg_path))
        names = {0: "drone"}
        lines, frame_rows = [], []
        start_all = time.perf_counter()
        for index, frame in iter_video_frames(video_path(self.cfg, seq), max_frames=max_frames):
            t0 = time.perf_counter()
            detections = self.runner.build_sahi_detections(frame=frame, det_model=self.det_model,
                                                           slice_h=size, slice_w=size, overlap=overlap,
                                                           conf_thresh=conf)
            box_tensor = torch.from_numpy(detections).to(device=self.torch_device, dtype=torch.float32)
            result = Results(orig_img=frame, path="frame.jpg", names=names, boxes=box_tensor)
            det = result.boxes.cpu().numpy()
            tracks = tracker.update(det, frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            n_out = 0
            for track in tracks if len(tracks) else []:
                x1, y1, x2, y2 = track[:4]
                lines.append(f"{index},{int(track[4])},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1")
                n_out += 1
            frame_rows.append({"frame": index, "time_ms": elapsed_ms, "n_gt": gt_index.n_gt(index),
                               "n_active": n_out, "n_lost": 0, "detect": 1})
        info = _summary(frame_rows, time.perf_counter() - start_all)
        _write_outputs(paths, lines, frame_rows, None)
        return info
