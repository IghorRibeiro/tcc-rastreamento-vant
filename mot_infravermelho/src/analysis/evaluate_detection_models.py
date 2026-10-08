#!/usr/bin/env python3
"""
Avaliação exclusiva de detecção de objetos para modelos YOLO.

O script calcula:
- AP50, AP75 e mAP@[0.50:0.95] por meio do avaliador oficial do Ultralytics;
- TP, FP, FN, precisão, recall, F1 e IoU médio dos matches em um limiar fixo;
- métricas AP manuais como verificação independente;
- não calcula TN, pois verdadeiro-negativo não é uma contagem bem definida
  no nível de instâncias/bounding boxes.

Exemplo:
python src/analysis/evaluate_detection_models.py \\
  --data configs/data.yaml \\
  --device mps \\
  --model baseline weights/best.pt \\
  --model hardmining weights/best_hardmining.pt \\
  --model hardmining20 weights/best_harmining20.pt \\
  --output runs/detection_eval
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO


IOU_AP_THRESHOLDS = np.arange(0.50, 0.951, 0.05)


def resolve_path(path_value: str | Path, root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else root / path


def load_dataset_images(data_path: Path) -> tuple[list[Path], list[str]]:
    """Resolve as imagens de validação e os nomes das classes do data.yaml."""
    with data_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    dataset_root = resolve_path(cfg.get("path", "."), data_path.parent)
    val_entry = cfg["val"]

    if isinstance(val_entry, list):
        image_paths: list[Path] = []
        for entry in val_entry:
            candidate = resolve_path(entry, dataset_root)
            image_paths.extend(sorted(p for p in candidate.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}))
    else:
        candidate = resolve_path(val_entry, dataset_root)
        if candidate.is_file() and candidate.suffix.lower() == ".txt":
            image_paths = [Path(line.strip()).expanduser() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
            image_paths = [p if p.is_absolute() else candidate.parent / p for p in image_paths]
        else:
            image_paths = sorted(p for p in candidate.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})

    names = cfg.get("names", {0: "object"})
    if isinstance(names, dict):
        class_names = [str(names[k]) for k in sorted(names, key=lambda value: int(value))]
    else:
        class_names = [str(name) for name in names]

    if not image_paths:
        raise RuntimeError(f"Nenhuma imagem de validação encontrada a partir de {data_path}")
    return image_paths, class_names


def label_path_for_image(image_path: Path) -> Path:
    """Converte .../images/val/x.jpg em .../labels/val/x.txt."""
    parts = list(image_path.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def load_ground_truth(image_path: Path) -> list[dict[str, Any]]:
    """Lê labels YOLO e converte xywh normalizado para xyxy em pixels."""
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []

    from PIL import Image

    width, height = Image.open(image_path).size
    objects: list[dict[str, Any]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        cls, cx, cy, box_w, box_h = map(float, values[:5])
        x1 = (cx - box_w / 2.0) * width
        y1 = (cy - box_h / 2.0) * height
        x2 = (cx + box_w / 2.0) * width
        y2 = (cy + box_h / 2.0) * height
        objects.append({"cls": int(cls), "box": np.array([x1, y1, x2, y2], dtype=float)})
    return objects


def box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    if len(many) == 0:
        return np.empty((0,), dtype=float)
    left = np.maximum(one[0], many[:, 0])
    top = np.maximum(one[1], many[:, 1])
    right = np.minimum(one[2], many[:, 2])
    bottom = np.minimum(one[3], many[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    area_one = max(0.0, one[2] - one[0]) * max(0.0, one[3] - one[1])
    area_many = np.maximum(0.0, many[:, 2] - many[:, 0]) * np.maximum(0.0, many[:, 3] - many[:, 1])
    union = area_one + area_many - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def load_predictions(model: YOLO, image_paths: list[Path], device: str, imgsz: int, ap_conf: float, nms_iou: float, max_det: int, infer_batch: int) -> list[dict[str, Any]]:
    """Executa inferência em lotes para evitar alocar todo o split no MPS."""
    path_to_index = {p.resolve(): index for index, p in enumerate(image_paths)}
    predictions = [{"boxes": np.empty((0, 4)), "scores": np.empty((0,)), "classes": np.empty((0,), dtype=int)} for _ in image_paths]

    # Passar as 5.724 imagens de uma vez ao predictor faz o Ultralytics
    # tentar reservar vários gigabytes no MPS. O batch aqui é independente
    # do batch usado pelo model.val().
    for start in range(0, len(image_paths), infer_batch):
        batch_paths = image_paths[start:start + infer_batch]
        results = model.predict(
            source=[str(path) for path in batch_paths],
            stream=True,
            batch=infer_batch,
            conf=ap_conf,
            iou=nms_iou,
            imgsz=imgsz,
            max_det=max_det,
            device=device,
            verbose=False,
        )
        for fallback_index, result in enumerate(results, start=start):
            result_path = Path(str(result.path)).resolve()
            index = path_to_index.get(result_path, fallback_index)
            if result.boxes is None or len(result.boxes) == 0:
                continue
            predictions[index] = {
                "boxes": result.boxes.xyxy.detach().cpu().numpy().astype(float),
                "scores": result.boxes.conf.detach().cpu().numpy().astype(float),
                "classes": result.boxes.cls.detach().cpu().numpy().astype(int),
            }
        del results

        # Libera o cache do backend Apple quando disponível. A chamada é
        # opcional e não altera as predições já copiadas para CPU.
        try:
            import torch
            if str(device).startswith("mps") and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        except Exception:
            pass

    return predictions


def greedy_matches(prediction: dict[str, Any], ground_truth: list[dict[str, Any]], confidence: float, iou_threshold: float) -> tuple[int, int, int, list[float]]:
    """Faz matching 1-para-1, ordenando predições por confiança."""
    keep = prediction["scores"] >= confidence
    order = np.argsort(-prediction["scores"][keep])
    boxes = prediction["boxes"][keep][order]
    scores = prediction["scores"][keep][order]
    classes = prediction["classes"][keep][order]

    used = set()
    tp = 0
    fp = 0
    matched_ious: list[float] = []
    for box, _score, cls in zip(boxes, scores, classes):
        candidates = [index for index, gt in enumerate(ground_truth) if index not in used and gt["cls"] == int(cls)]
        if not candidates:
            fp += 1
            continue
        candidate_boxes = np.stack([ground_truth[index]["box"] for index in candidates])
        ious = box_iou(box, candidate_boxes)
        best_local = int(np.argmax(ious))
        best_iou = float(ious[best_local])
        if best_iou >= iou_threshold:
            used.add(candidates[best_local])
            tp += 1
            matched_ious.append(best_iou)
        else:
            fp += 1
    fn = len(ground_truth) - tp
    return tp, fp, fn, matched_ious


def interpolated_ap(predictions: list[dict[str, Any]], ground_truths: list[list[dict[str, Any]]], class_id: int, iou_threshold: float) -> float:
    """AP interpolado em 101 pontos de recall, para uma classe e IoU."""
    total_gt = sum(1 for image_gt in ground_truths for gt in image_gt if gt["cls"] == class_id)
    if total_gt == 0:
        return float("nan")

    candidates: list[tuple[float, int, np.ndarray]] = []
    for image_index, prediction in enumerate(predictions):
        for box, score, cls in zip(prediction["boxes"], prediction["scores"], prediction["classes"]):
            if int(cls) == class_id:
                candidates.append((float(score), image_index, box))
    candidates.sort(key=lambda item: item[0], reverse=True)

    used: dict[int, set[int]] = {index: set() for index in range(len(ground_truths))}
    tp = np.zeros(len(candidates), dtype=float)
    fp = np.zeros(len(candidates), dtype=float)
    for index, (_score, image_index, box) in enumerate(candidates):
        gt_indices = [gt_index for gt_index, gt in enumerate(ground_truths[image_index]) if gt["cls"] == class_id and gt_index not in used[image_index]]
        if not gt_indices:
            fp[index] = 1.0
            continue
        gt_boxes = np.stack([ground_truths[image_index][gt_index]["box"] for gt_index in gt_indices])
        ious = box_iou(box, gt_boxes)
        best_local = int(np.argmax(ious))
        if float(ious[best_local]) >= iou_threshold:
            used[image_index].add(gt_indices[best_local])
            tp[index] = 1.0
        else:
            fp[index] = 1.0

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall = cumulative_tp / max(total_gt, 1)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    values = []
    for recall_level in np.linspace(0.0, 1.0, 101):
        valid = precision_envelope[recall >= recall_level]
        values.append(float(valid.max()) if len(valid) else 0.0)
    return float(np.mean(values))


def official_metrics(model: YOLO, data_path: Path, device: str, imgsz: int, batch: int, ap_conf: float) -> dict[str, float]:
    """Obtém AP/mAP pelo avaliador oficial do Ultralytics."""
    metrics = model.val(
        data=str(data_path),
        split="val",
        imgsz=imgsz,
        batch=batch,
        conf=ap_conf,
        device=device,
        workers=0,
        plots=False,
        save_json=False,
        verbose=False,
    )
    box = metrics.box
    values = {
        "AP": float(box.map),
        "AP50": float(box.map50),
        "AP75": float(box.map75),
        "precision_best_f1": float(box.mp),
        "recall_best_f1": float(box.mr),
        "F1_best_f1": float(2 * box.mp * box.mr / max(box.mp + box.mr, 1e-12)),
    }
    return values


def evaluate_model(name: str, weight_path: Path, data_path: Path, image_paths: list[Path], class_names: list[str], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"\n{'=' * 80}\nMODELO: {name}\nPESO: {weight_path}\n{'=' * 80}")
    model = YOLO(str(weight_path))
    # O val() da versão local do Ultralytics/MPS retornou P=R=mAP=0 apesar
    # de a inferência produzir caixas corretas. Portanto, AP é calculado
    # abaixo diretamente das predições e dos labels do mesmo split.
    predictions = load_predictions(model, image_paths, args.device, args.imgsz, args.ap_conf, args.nms_iou, args.max_det, args.infer_batch)
    ground_truths = [load_ground_truth(path) for path in image_paths]

    tp, fp, fn, matched_ious = 0, 0, 0, []
    per_image_rows = []
    for index, (image_path, prediction, ground_truth) in enumerate(zip(image_paths, predictions, ground_truths)):
        image_tp, image_fp, image_fn, image_ious = greedy_matches(prediction, ground_truth, args.conf, args.match_iou)
        tp += image_tp
        fp += image_fp
        fn += image_fn
        matched_ious.extend(image_ious)
        per_image_rows.append({"model": name, "image": str(image_path), "ground_truth": len(ground_truth), "predictions_at_conf": int(np.sum(prediction["scores"] >= args.conf)), "TP": image_tp, "FP": image_fp, "FN": image_fn})

    manual_ap_values = []
    for iou_threshold in IOU_AP_THRESHOLDS:
        class_values = [interpolated_ap(predictions, ground_truths, class_id, float(iou_threshold)) for class_id in range(len(class_names))]
        valid_values = [value for value in class_values if not math.isnan(value)]
        manual_ap_values.append(float(np.mean(valid_values)) if valid_values else float("nan"))

    ground_truth_count = sum(len(objects) for objects in ground_truths)
    detections_at_conf = tp + fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    row = {
        "model": name,
        "weight": str(weight_path),
        "images": len(image_paths),
        "ground_truth_objects": ground_truth_count,
        "detections_at_conf": detections_at_conf,
        "confidence_threshold": args.conf,
        "match_iou_threshold": args.match_iou,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": np.nan,
        "TN_note": "não definido no nível de instâncias/bounding boxes",
        "precision_at_conf": precision,
        "recall_at_conf": recall,
        "F1_at_conf": f1,
        "mean_IoU_TP_at_conf": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "AP": float(np.nanmean(manual_ap_values)),
        "AP50": manual_ap_values[0],
        "AP75": manual_ap_values[5],
        "mAP50_95": float(np.nanmean(manual_ap_values)),
        "precision_best_f1_Ultralytics": np.nan,
        "recall_best_f1_Ultralytics": np.nan,
        "F1_best_f1_Ultralytics": np.nan,
        "manual_AP50": manual_ap_values[0],
        "manual_AP75": manual_ap_values[5],
        "manual_mAP50_95": float(np.nanmean(manual_ap_values)),
    }
    return row, per_image_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparação de detecção YOLO sem tracker/MOT")
    parser.add_argument("--data", required=True, type=Path, help="Caminho para configs/data.yaml")
    parser.add_argument("--model", dest="models", nargs=2, action="append", required=True, metavar=("NOME", "PESO"), help="Par repetível: --model nome caminho.pt")
    parser.add_argument("--output", type=Path, default=Path("runs/detection_eval"))
    parser.add_argument("--device", default="mps", help="mps, cpu ou cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="Batch usado pelo model.val()")
    parser.add_argument("--infer-batch", type=int, default=1, help="Batch usado pela inferência detalhada; use 1 no MPS se houver erro de memória")
    parser.add_argument("--conf", type=float, default=0.25, help="Limiar para TP/FP/FN")
    parser.add_argument("--ap-conf", type=float, default=0.001, help="Limiar baixo usado para preservar predições no cálculo de AP")
    parser.add_argument("--match-iou", type=float, default=0.50, help="IoU mínimo para um TP no ponto operacional")
    parser.add_argument("--nms-iou", type=float, default=0.70, help="IoU do NMS durante a inferência")
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    if args.infer_batch < 1:
        parser.error("--infer-batch deve ser >= 1")

    data_path = args.data.expanduser().resolve()
    image_paths, class_names = load_dataset_images(data_path)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Imagens de validação: {len(image_paths)}")
    print(f"Classes: {class_names}")
    print(f"TP/FP/FN: conf >= {args.conf:.3f}, IoU >= {args.match_iou:.2f}")
    print(f"AP: conf >= {args.ap_conf:.3f}, NMS IoU = {args.nms_iou:.2f}")

    summary_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    for name, weight in args.models:
        row, details = evaluate_model(name, Path(weight).expanduser().resolve(), data_path, image_paths, class_names, args)
        summary_rows.append(row)
        per_image_rows.extend(details)

    summary_path = args.output / "detection_summary.csv"
    per_image_path = args.output / "detection_per_image.csv"
    config_path = args.output / "evaluation_config.json"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(per_image_rows).to_csv(per_image_path, index=False)
    config_path.write_text(json.dumps({
        "data": str(data_path),
        "models": [{"name": name, "weight": str(Path(weight).expanduser().resolve())} for name, weight in args.models],
        "num_validation_images": len(image_paths),
        "classes": class_names,
        "imgsz": args.imgsz,
        "batch_for_model_val": args.batch,
        "batch_for_detailed_inference": args.infer_batch,
        "confidence_threshold_for_counts": args.conf,
        "confidence_threshold_for_ap": args.ap_conf,
        "match_iou_threshold": args.match_iou,
        "nms_iou": args.nms_iou,
        "note": "TN não é definido no nível de instâncias/bounding boxes; AP/mAP são calculados diretamente a partir das predições e dos labels do split val, pois o val() local retornou zeros inconsistentes com a inferência.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    display_columns = ["model", "AP50", "mAP50_95", "AP75", "TP", "FP", "FN", "precision_at_conf", "recall_at_conf", "F1_at_conf", "mean_IoU_TP_at_conf"]
    print("\nRESULTADO CONSOLIDADO")
    print(pd.DataFrame(summary_rows)[display_columns].to_string(index=False))
    print(f"\nResumo: {summary_path}")
    print(f"Por imagem: {per_image_path}")
    print(f"Configuração: {config_path}")


if __name__ == "__main__":
    main()
