import os, cv2, time, csv, json
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==================== CONFIGURAÇÕES ====================
EXPERIMENT_NAME = "Hibrido_YOLO26n_CSRT_CV_Damped_0.95"
DATASET_NAME = "Anti-UAV-Tracking-V0"
MODEL_PATH = '/Users/ighorribeiro/PycharmProjects/TesteYolo26/pesos_treinados/yolo26n_dut_best.pt'
DATASET_ROOT = '/Volumes/1T Ighor/Drone_Tracking/Datasets/Anti-UAV-Tracking/Anti-UAV-Tracking-V0'
GT_DIR = DATASET_ROOT
OUTPUT_DIR = 'avaliacao_avancada/hibrido_cv_damped'
DEVICE = 'mps'

INIT_CONF = 0.6
SWITCH_CONF = 0.6
YOLO_PERIOD = 15
PREDICTION_HORIZONS = [5, 10, 15, 30]
DAMPING = 0.95

DRIFT_THRESHOLDS = [100, 250, 500]
MIN_EPISODE_LEN = 6

# ========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/tmp', exist_ok=True)
detector = YOLO(MODEL_PATH)
VIDEO_IDS = [f'video{i:02d}' for i in range(1, 21)]


def compute_iou(boxA, boxB):
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    xi1 = max(xA, xB);
    yi1 = max(yA, yB)
    xi2 = min(xA + wA, xB + wB);
    yi2 = min(yA + hA, yB + hB)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    areaA = wA * hA;
    areaB = wB * hB
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0


def calc_ap_coco(gt_frames, pred_dict, frames, video_id, tmp_dir):
    coco_gt = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "drone"}]}
    coco_dt = []
    ann_id = 0
    for f_idx, frame in enumerate(frames, start=1):
        gt_bbox = gt_frames.get(frame)
        pred = pred_dict.get(frame)
        coco_gt["images"].append({"id": f_idx})
        if gt_bbox is not None:
            coco_gt["annotations"].append({
                "id": ann_id, "image_id": f_idx, "category_id": 1,
                "bbox": [float(v) for v in gt_bbox],
                "area": float(gt_bbox[2] * gt_bbox[3]),
                "iscrowd": 0, "ignore": 0
            })
            ann_id += 1
        if pred is not None:
            pred_bbox = pred[1:5]
            score = pred[5]
            coco_dt.append({
                "image_id": f_idx, "category_id": 1,
                "bbox": [float(v) for v in pred_bbox],
                "score": float(score)
            })
    ap50 = ap = 0.0
    if len(coco_gt["annotations"]) > 0 and len(coco_dt) > 0:
        gt_json = f'{tmp_dir}/gt_{video_id}.json'
        dt_json = f'{tmp_dir}/dt_{video_id}.json'
        with open(gt_json, 'w') as f: json.dump(coco_gt, f)
        with open(dt_json, 'w') as f: json.dump(coco_dt, f)
        cocoGt = COCO(gt_json);
        cocoDt = cocoGt.loadRes(dt_json)
        evaluator = COCOeval(cocoGt, cocoDt, iouType='bbox')
        evaluator.evaluate();
        evaluator.accumulate();
        evaluator.summarize()
        ap50 = evaluator.stats[1];
        ap = evaluator.stats[0]
        os.remove(gt_json);
        os.remove(dt_json)
    return ap50, ap


def calc_pred_errors_frame(gt_centers, measurements, horizon, damping):
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    initialized = False
    errors = {}
    for frame in sorted(measurements.keys()):
        cx, cy = measurements[frame]
        if not initialized:
            kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
            kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
            initialized = True
            continue
        kalman.predict()
        kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        state = kalman.statePost.copy()
        # Aplica amortecimento na velocidade do estado corrigido
        state[2, 0] *= damping
        state[3, 0] *= damping
        kalman.statePost = state

        future_frame = frame + horizon
        if future_frame in gt_centers and gt_centers[future_frame] is not None:
            state_pred = state
            for _ in range(horizon):
                state_pred = kalman.transitionMatrix @ state_pred
                # Aplica amortecimento a cada passo da previsão
                state_pred[2, 0] *= damping
                state_pred[3, 0] *= damping
            pred_x, pred_y = state_pred[0, 0], state_pred[1, 0]
            gt_cx, gt_cy = gt_centers[future_frame]
            err = np.sqrt((pred_x - gt_cx) ** 2 + (pred_y - gt_cy) ** 2)
            errors[frame] = err
    return errors


def find_episodes(errors, threshold, min_len):
    episodes = []
    cur = 0
    for e in errors:
        if e > threshold:
            cur += 1
        else:
            if cur >= min_len: episodes.append(cur)
            cur = 0
    if cur >= min_len: episodes.append(cur)
    return episodes


def process_hybrid_damped(video_id):
    video_path = os.path.join(DATASET_ROOT, video_id, f'{video_id}.mp4')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None

    gt_file = os.path.join(GT_DIR, video_id, f'{video_id}_gt.txt')
    gt_frames = {};
    gt_centers = {};
    gt_bbox_areas = []
    with open(gt_file, 'r') as f:
        for idx, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) != 4: continue
            x, y, w, h = map(float, parts)
            if x < 0 or y < 0:
                gt_frames[idx] = None;
                gt_centers[idx] = None
            else:
                gt_frames[idx] = [x, y, w, h]
                gt_centers[idx] = (x + w / 2, y + h / 2)
                gt_bbox_areas.append(w * h)

    tracker = None;
    last_det_conf = 0.0
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kalman_initialized = False

    frame_count = 0;
    yolo_calls = 0
    preds = [];
    times_per_frame = [];
    center_errors = []
    measurements = {}

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        t0 = time.time()

        if tracker is None:
            yolo_calls += 1
            results = detector(frame, device=DEVICE, conf=INIT_CONF, verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().numpy()
                best_idx = confs.argmax()
                x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
                conf = float(confs[best_idx])
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x1, y1, w, h))
                last_det_conf = conf
                cx, cy = x1 + w / 2, y1 + h / 2
                kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
                kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
                kalman_initialized = True
                t1 = time.time()
                preds.append((frame_count, x1, y1, w, h, conf, (t1 - t0) * 1000))
                measurements[frame_count] = (cx, cy)
                # Erro de centro
                gt_bbox = gt_frames.get(frame_count)
                if gt_bbox is not None:
                    gt_cx, gt_cy = gt_bbox[0] + gt_bbox[2] / 2, gt_bbox[1] + gt_bbox[3] / 2
                    center_errors.append(np.sqrt((cx - gt_cx) ** 2 + (cy - gt_cy) ** 2))
                else:
                    center_errors.append(0.0)
            else:
                t1 = time.time()
        else:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                conf = last_det_conf
                t1 = time.time()
                preds.append((frame_count, x, y, w, h, conf, (t1 - t0) * 1000))
                cx, cy = x + w / 2, y + h / 2
                measurements[frame_count] = (cx, cy)
                if kalman_initialized:
                    kalman.predict()
                    kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
                    # Aplica amortecimento na velocidade
                    state = kalman.statePost.copy()
                    state[2, 0] *= DAMPING
                    state[3, 0] *= DAMPING
                    kalman.statePost = state

                if frame_count % YOLO_PERIOD == 0:
                    yolo_calls += 1
                    results = detector(frame, device=DEVICE, conf=SWITCH_CONF, verbose=False)
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        confs = boxes.conf.cpu().numpy()
                        best_conf = float(confs.max())
                        if best_conf > SWITCH_CONF:
                            best_idx = confs.argmax()
                            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            w_new, h_new = x2 - x1, y2 - y1
                            tracker = cv2.TrackerCSRT_create()
                            tracker.init(frame, (x1, y1, w_new, h_new))
                            last_det_conf = best_conf
                            if preds: preds[-1] = (frame_count, x1, y1, w_new, h_new, best_conf, preds[-1][6])
                            cx_n, cy_n = x1 + w_new / 2, y1 + h_new / 2
                            kalman.statePre = np.array([[cx_n], [cy_n], [0], [0]], np.float32)
                            kalman.statePost = np.array([[cx_n], [cy_n], [0], [0]], np.float32)
                            kalman_initialized = True
                # Erro de centro
                gt_bbox = gt_frames.get(frame_count)
                if gt_bbox is not None:
                    gt_cx, gt_cy = gt_bbox[0] + gt_bbox[2] / 2, gt_bbox[1] + gt_bbox[3] / 2
                    center_errors.append(np.sqrt((cx - gt_cx) ** 2 + (cy - gt_cy) ** 2))
                else:
                    center_errors.append(0.0)
            else:
                tracker = None;
                kalman_initialized = False
                t1 = time.time()

        times_per_frame.append((t1 - t0) * 1000)

    cap.release()

    pred_dict = {}
    for p in preds:
        if p[0] not in pred_dict: pred_dict[p[0]] = p

    frames = sorted(set(list(gt_frames.keys()) + list(pred_dict.keys())))
    total_frames = len(frames)
    frames_com_drone = sum(1 for f in frames if gt_frames.get(f) is not None)
    sa_correct = 0;
    sr_count_05 = 0
    for frame in frames:
        gt_bbox = gt_frames.get(frame)
        pred = pred_dict.get(frame)
        pred_bbox = None if pred is None else pred[1:5]
        conf = 0 if pred is None else pred[5]
        if gt_bbox is not None:
            if pred_bbox is not None:
                iou = compute_iou(gt_bbox, pred_bbox)
                if iou >= 0.5: sr_count_05 += 1; sa_correct += 1
        else:
            if pred is None or conf < 0.25: sa_correct += 1

    sa = sa_correct / total_frames if total_frames else 0
    sr05 = sr_count_05 / frames_com_drone if frames_com_drone else 0
    avg_bbox_area = np.mean(gt_bbox_areas) if gt_bbox_areas else 0
    avg_bbox_size_px = np.sqrt(avg_bbox_area) if avg_bbox_area > 0 else 0
    tempo_medio = np.mean(times_per_frame) if times_per_frame else 0
    fps = 1000 / tempo_medio if tempo_medio > 0 else 0
    yolo_calls_pct = (yolo_calls / frame_count) * 100 if frame_count else 0

    ap50, ap = calc_ap_coco(gt_frames, pred_dict, frames, video_id, f'{OUTPUT_DIR}/tmp')

    # Erros de predição (com damping)
    error_stats = {}
    for h in PREDICTION_HORIZONS:
        errs = calc_pred_errors_frame(gt_centers, measurements, h, DAMPING)
        vals = list(errs.values())
        if vals:
            error_stats[f'erro_medio_h{h}'] = np.mean(vals)
            error_stats[f'erro_mediana_h{h}'] = np.median(vals)
            error_stats[f'erro_std_h{h}'] = np.std(vals)
        else:
            for suf in ['medio', 'mediana', 'std']:
                error_stats[f'erro_{suf}_h{h}'] = 0.0

    # Episódios de deriva
    drift_stats = {}
    for th in DRIFT_THRESHOLDS:
        ep = find_episodes(center_errors, th, MIN_EPISODE_LEN)
        drift_stats[f'drift_ep_{th}_count'] = len(ep)
        drift_stats[f'drift_ep_{th}_mean_len'] = np.mean(ep) if ep else 0
        drift_stats[f'drift_ep_{th}_max_len'] = np.max(ep) if ep else 0

    return {
        'video': video_id,
        'SA': sa, 'SR@0.5': sr05, 'AP@0.5': ap50, 'AP@0.5:0.95': ap,
        'FPS': fps, 'Tempo_medio_ms': tempo_medio,
        'YOLO_calls_pct': yolo_calls_pct,
        **error_stats, **drift_stats
    }, preds, gt_frames, gt_centers, measurements, center_errors


# ==================== EXECUÇÃO ====================
if __name__ == '__main__':
    print(f"Processando Híbrido com Kalman CV Damped ({DAMPING})...")
    all_metrics = []

    for vid in VIDEO_IDS:
        res = process_hybrid_damped(vid)
        if res is None: continue
        metrics, _, _, _, _, _ = res
        all_metrics.append(metrics)
        print(f"  {vid}: SA={metrics['SA']:.4f}, AP@0.5={metrics['AP@0.5']:.4f}, FPS={metrics['FPS']:.1f}")

    df = pd.DataFrame(all_metrics)
    df.to_csv(f'{OUTPUT_DIR}/metricas_hibrido_cv_damped.csv', index=False)

    # Carrega baseline (Híbrido CV original)
    baseline_path = 'avaliacao_avancada/metricas_hibrido_COMPLETO.csv'
    if os.path.exists(baseline_path):
        df_base = pd.read_csv(baseline_path)

        # Comparação
        print("\n=== COMPARAÇÃO HÍBRIDO CV ORIGINAL vs CV DAMPED 0.95 ===")
        metrics_compare = ['SA', 'AP@0.5', 'AP@0.5:0.95', 'FPS']
        for h in PREDICTION_HORIZONS:
            metrics_compare.extend([f'erro_medio_h{h}', f'erro_mediana_h{h}'])

        for m in metrics_compare:
            if m not in df.columns or m not in df_base.columns: continue
            base_mean = df_base[m].mean()
            damped_mean = df[m].mean()
            gain = (base_mean - damped_mean) / base_mean * 100 if base_mean != 0 else 0
            print(f"  {m:<20}: CV={base_mean:.2f}  Damped={damped_mean:.2f}  Ganho={gain:+.1f}%")

        # Tabela detalhada por vídeo
        comp = pd.merge(df_base[['video'] + metrics_compare],
                        df[['video'] + metrics_compare],
                        on='video', suffixes=('_cv', '_damped'))
        comp.to_csv(f'{OUTPUT_DIR}/comparacao_damped_por_video.csv', index=False)
    else:
        print(f"Baseline não encontrado em {baseline_path}")

    print(f"\n✅ Resultados salvos em '{OUTPUT_DIR}'")