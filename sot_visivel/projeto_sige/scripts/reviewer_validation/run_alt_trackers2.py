#!/usr/bin/env python3
"""Run alternative trackers on the DUT Anti-UAV image sequences.

Trackers:
  - bytetrack: Ultralytics ByteTrack with a YOLO detector.
  - deepsort: deep-sort-realtime with the MobileNet embedder.
  - ostrack: Single Object Tracker (ViT-based) using the first GT frame.

Examples:
  python scripts/reviewer_validation/run_alt_trackers2.py --tracker ostrack --ostrack-repo OSTrack --ostrack-weights vitb_384_mae_ce_32x4_ep300 --device mps
  python scripts/reviewer_validation/run_alt_trackers2.py \ --tracker odtrack \--odtrack-repo ODTrack \ --odtrack-weights baseline \ --video video01 \ --device mps
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

# Correção de compatibilidade para torch._six em versões novas do PyTorch
import torch
if not hasattr(torch, '_six'):
    class MockSix:
        string_classes = (str, bytes)
    torch._six = MockSix()
    sys.modules['torch._six'] = torch._six

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_RE = "video*_*.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Root of the TesteYolo26 project.",
    )
    parser.add_argument(
        "--tracker",
        choices=("bytetrack", "deepsort", "ostrack", "odtrack", "all"),
        default="all",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Only process one sequence, e.g. video01. Default: all 20 videos.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="YOLO weights (for bytetrack/deepsort).",
    )
    parser.add_argument(
        "--ostrack-repo",
        type=Path,
        default=None,
        help="Path to the cloned OSTrack repository.",
    )
    parser.add_argument(
        "--ostrack-weights",
        type=str,
        default="vitb_384_mae_ce_32x4_ep300",
        help="OSTrack parameter name (weights file name without extension).",
    )
    parser.add_argument(
        "--odtrack-repo",
        type=Path,
        default=None,
        help="Path to the cloned ODTrack repository (github.com/GXNU-ZhongLab/ODTrack).",
    )
    parser.add_argument(
        "--odtrack-weights",
        type=str,
        default="baseline",
        help="Nome do yaml em experiments/odtrack/ (sem extensão), ex.: baseline, baseline_large.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Device to run on, e.g. mps, cuda, cpu.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: avaliacao_tracking/predicoes/reviewer_validation.",
    )
    return parser.parse_args()


def video_ids(images_dir: Path, only: str | None) -> list[str]:
    found = sorted({p.name.split("_")[0] for p in images_dir.glob(VIDEO_RE)})
    if only is not None:
        if only not in found:
            raise FileNotFoundError(f"Sequence {only!r} not found under {images_dir}")
        return [only]
    return found


def frame_paths(images_dir: Path, video_id: str) -> list[Path]:
    paths = sorted(images_dir.glob(f"{video_id}_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No frames found for {video_id} under {images_dir}")
    return paths


def frame_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def xyxy_to_xywh(xyxy: Iterable[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def get_initial_bbox(project_root: Path, video_id: str, first_frame_path: Path) -> list[float]:
    """Lê o primeiro bounding box do frame inicial e devolve em pixels, formato [x, y, w, h]
    (canto superior-esquerdo), a partir de um label YOLO normalizado
    (classe, x_center, y_center, w, h — todos em [0,1])."""
    labels_dir = project_root / "datasets" / "anti_uav_tracking_test_final" / "labels" / "test"

    gt_files = sorted(labels_dir.glob(f"{video_id}_*.txt"))
    if not gt_files:
        gt_path = project_root / "datasets" / "anti_uav_tracking_test_final" / "annotations" / f"{video_id}.txt"
        if not gt_path.is_file():
            raise FileNotFoundError(f"Ground truth files not found for {video_id} under {labels_dir} or annotations/")
        with open(gt_path, "r") as f:
            first_line = f.readline().strip()
    else:
        gt_path = gt_files[0]
        with open(gt_path, "r") as f:
            first_line = f.readline().strip()

    val_strs = first_line.replace('\t', ',').replace(' ', ',').split(',')
    vals = [float(v) for v in val_strs if v.strip()]
    if len(vals) < 5:
        raise ValueError(f"Linha de label inesperada em {gt_path}: {first_line!r}")

    # Descarta o índice de classe (vals[0]); os 4 seguintes são normalizados (YOLO: classe xc yc w h)
    _cls, xc_n, yc_n, w_n, h_n = vals[:5]

    frame = cv2.imread(str(first_frame_path))
    if frame is None:
        raise RuntimeError(f"Could not decode {first_frame_path} to read image dimensions")
    img_h, img_w = frame.shape[:2]

    w_px = w_n * img_w
    h_px = h_n * img_h
    x_px = xc_n * img_w - w_px / 2.0
    y_px = yc_n * img_h - h_px / 2.0
    return [x_px, y_px, w_px, h_px]


def write_prediction(handle, frame: int, bbox: tuple[float, float, float, float], conf: float,
                     elapsed_ms: float) -> None:
    x, y, w, h = bbox
    handle.write(f"{frame},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{conf:.6f},{elapsed_ms:.3f}\n")


def run_bytetrack(model_path: Path, images_dir: Path, video_id: str, output_file: Path, device: str,
                  conf: float) -> None:
    pass


def run_deepsort(model_path: Path, images_dir: Path, video_id: str, output_file: Path, device: str,
                 conf: float) -> None:
    pass


def run_vit_tracker(
    tracker_repo: Path,
    param_name: str,
    project_root: Path,
    images_dir: Path,
    video_id: str,
    output_file: Path,
    device_str: str,
    tracker_module: str,
    tracker_class: str,
    config_subdir: str,
    checkpoint_subdir: str,
    constructor_needs_dataset_name: bool,
    dataset_name: str = "trackingnet",
) -> None:
    """Roda um tracker estilo OSTrack/ODTrack (mesma família de código lib.test.tracker.*)
    usando o primeiro frame do Ground Truth para inicialização.

    Nomes confirmados por inspeção direta dos repositórios oficiais:
      - OSTrack (github.com/botaoye/OSTrack): módulo lib.test.tracker.ostrack, classe OSTrack,
        construtor (params, dataset_name).
      - ODTrack (github.com/GXNU-ZhongLab/ODTrack): módulo lib.test.tracker.odtrack, classe ODTrack,
        construtor (params) -- SEM dataset_name.
    Se você usar outro tracker dessa família, confirme esses 5 valores no repo antes de rodar.
    """

    # --- MOCKS DE COMPATIBILIDADE ---
    import types
    import sys
    import importlib

    if 'jpeg4py' not in sys.modules:
        class MockJpeg4Py:
            def __init__(self, filename):
                self.filename = filename
            def read(self):
                img = cv2.imread(self.filename)
                if img is None:
                    raise FileNotFoundError(f"Não foi possível ler a imagem: {self.filename}")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sys.modules['jpeg4py'] = MockJpeg4Py

    if 'visdom' not in sys.modules:
        visdom_mock = types.ModuleType('visdom')
        visdom_mock.server = types.ModuleType('visdom.server')
        class MockVisdom:
            def __init__(self, *args, **kwargs):
                pass
            def check_connection(self):
                return True
            def line(self, *args, **kwargs):
                pass
            def text(self, *args, **kwargs):
                pass
            def close(self, *args, **kwargs):
                pass
        visdom_mock.Visdom = MockVisdom
        sys.modules['visdom'] = visdom_mock
        sys.modules['visdom.server'] = visdom_mock.server
    # --------------------------------

    if str(tracker_repo) not in sys.path:
        sys.path.insert(0, str(tracker_repo))

    try:
        tracker_mod = importlib.import_module(tracker_module)
        TrackerClass = getattr(tracker_mod, tracker_class)
        config_mod = importlib.import_module(f"lib.config.{config_subdir}.config")
        cfg = config_mod.cfg
        update_config_from_file = config_mod.update_config_from_file
        from lib.test.tracker.data_utils import Preprocessor
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Não foi possível importar {tracker_class} de {tracker_repo} "
            f"(module={tracker_module}). Detalhes do erro: {e}"
        )

    # --- DEFINIÇÃO PRÉVIA DO DISPOSITIVO ---
    device = torch.device(device_str)

    # --- PATCH GLOBAL PARA REDIRECIONAR QUALQUER .cuda() PARA O MAC ---
    # (cobre tanto Tensor.cuda() quanto nn.Module.cuda(), que internamente chama Tensor.cuda()
    #  em cada parâmetro — é assim que o ODTrack faz "network.cuda()" no __init__ dele)
    old_cuda = torch.Tensor.cuda
    def new_cuda(self, *args, **kwargs):
        return self.to(device)
    torch.Tensor.cuda = new_cuda

    def new_init(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view((1, 3, 1, 1))
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view((1, 3, 1, 1))
    Preprocessor.__init__ = new_init
    # -----------------------------------------------------------------

    config_file = tracker_repo / "experiments" / config_subdir / f"{param_name}.yaml"
    if config_file.is_file():
        update_config_from_file(str(config_file))
    else:
        print(f"[aviso] config {config_file} não encontrado — usando defaults do cfg. "
              f"Se os resultados vierem ruins, confira este caminho primeiro.")

    class TrackerParams:
        def __init__(self):
            self.cfg = cfg
            self.checkpoint = str(
                tracker_repo / "output" / "checkpoints" / "train" / checkpoint_subdir / param_name / f"{param_name}.pth.tar")
            self.device = device
            self.debug = 0
            self.run_id = None
            self.save_all_boxes = False

            self.template_factor = cfg.TEST.TEMPLATE_FACTOR if hasattr(cfg, 'TEST') and hasattr(cfg.TEST, 'TEMPLATE_FACTOR') else 2.0
            self.search_factor = cfg.TEST.SEARCH_FACTOR if hasattr(cfg, 'TEST') and hasattr(cfg.TEST, 'SEARCH_FACTOR') else 4.0
            self.template_size = cfg.TEST.TEMPLATE_SIZE if hasattr(cfg, 'TEST') and hasattr(cfg.TEST, 'TEMPLATE_SIZE') else 192
            self.search_size = cfg.TEST.SEARCH_SIZE if hasattr(cfg, 'TEST') and hasattr(cfg.TEST, 'SEARCH_SIZE') else 384

    params = TrackerParams()
    if not Path(params.checkpoint).is_file():
        raise FileNotFoundError(
            f"Checkpoint não encontrado em {params.checkpoint}. Confira --{checkpoint_subdir}-weights "
            f"e a estrutura de pastas do repositório clonado (deve ter sido baixado dos links de "
            f"'Models' do README e colocado em output/checkpoints/train/{checkpoint_subdir}/<param_name>/)."
        )

    if constructor_needs_dataset_name:
        tracker = TrackerClass(params, dataset_name)
    else:
        tracker = TrackerClass(params)

    paths = frame_paths(images_dir, video_id)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    init_bbox = get_initial_bbox(project_root, video_id, paths[0])
    init_info = {'init_bbox': init_bbox}

    predictions = 0
    with output_file.open("w", encoding="utf-8") as handle:
        for i, path in enumerate(paths):
            frame_bgr = cv2.imread(str(path))
            if frame_bgr is None:
                raise RuntimeError(f"Could not decode {path}")
            # O pipeline oficial (lib/test/evaluation/tracker.py) lê com cv2.imread e converte
            # para RGB antes de repassar ao tracker -- reproduzimos isso aqui.
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            start = time.perf_counter()

            if i == 0:
                tracker.initialize(frame, init_info)
                bbox = init_bbox
                conf = 1.0
            else:
                out = tracker.track(frame)
                bbox = out['target_bbox']
                conf = out.get('conf_score', 1.0)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            write_prediction(handle, frame_number(path), bbox, conf, elapsed_ms)
            predictions += 1

    print(f"{tracker_class} {video_id}: {len(paths)} frames, {predictions} predictions -> {output_file}")


def run_ostrack(ostrack_repo: Path, param_name: str, project_root: Path, images_dir: Path, video_id: str,
                output_file: Path, device_str: str) -> None:
    run_vit_tracker(
        tracker_repo=ostrack_repo,
        param_name=param_name,
        project_root=project_root,
        images_dir=images_dir,
        video_id=video_id,
        output_file=output_file,
        device_str=device_str,
        tracker_module="lib.test.tracker.ostrack",
        tracker_class="OSTrack",
        config_subdir="ostrack",
        checkpoint_subdir="ostrack",
        constructor_needs_dataset_name=True,
    )


def run_odtrack(odtrack_repo: Path, param_name: str, project_root: Path, images_dir: Path, video_id: str,
                output_file: Path, device_str: str) -> None:
    run_vit_tracker(
        tracker_repo=odtrack_repo,
        param_name=param_name,
        project_root=project_root,
        images_dir=images_dir,
        video_id=video_id,
        output_file=output_file,
        device_str=device_str,
        tracker_module="lib.test.tracker.odtrack",
        tracker_class="ODTrack",
        config_subdir="odtrack",
        checkpoint_subdir="odtrack",
        constructor_needs_dataset_name=False,
    )


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    images_dir = project_root / "datasets" / "anti_uav_tracking_test_final" / "images" / "test"
    output_root = (args.output_root or (project_root / "avaliacao_tracking" / "predicoes" / "reviewer_validation")).resolve()

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    ids = video_ids(images_dir, args.video)
    requested = ("bytetrack", "deepsort", "ostrack", "odtrack") if args.tracker == "all" else (args.tracker,)

    for tracker_name in requested:
        for video_id in ids:
            output_file = output_root / tracker_name / f"{video_id}.txt"

            if tracker_name == "bytetrack":
                model_path = (args.model or (project_root / "pesos_treinados" / "yolo26n_dut_best.pt")).resolve()
                run_bytetrack(model_path, images_dir, video_id, output_file, args.device, args.conf)

            elif tracker_name == "deepsort":
                model_path = (args.model or (project_root / "pesos_treinados" / "yolo26n_dut_best.pt")).resolve()
                run_deepsort(model_path, images_dir, video_id, output_file, args.device, args.conf)

            elif tracker_name == "ostrack":
                if args.ostrack_repo is None:
                    raise ValueError("Você deve fornecer --ostrack-repo apontando para o diretório clonado do OSTrack.")
                ostrack_repo = args.ostrack_repo.resolve()
                run_ostrack(ostrack_repo, args.ostrack_weights, project_root, images_dir, video_id, output_file, args.device)

            elif tracker_name == "odtrack":
                if args.odtrack_repo is None:
                    raise ValueError("Você deve fornecer --odtrack-repo apontando para o diretório clonado do ODTrack.")
                odtrack_repo = args.odtrack_repo.resolve()
                run_odtrack(odtrack_repo, args.odtrack_weights, project_root, images_dir, video_id, output_file, args.device)


if __name__ == "__main__":
    main()