import os, cv2, time, csv, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==================== CONFIGURAÇÕES ====================
MODEL_PATH = '/Users/ighorribeiro/PycharmProjects/TesteYolo26/pesos_treinados/yolo26n_dut_best.pt'
DATASET_ROOT = '/Volumes/1T Ighor/Drone_Tracking/Datasets/Anti-UAV-Tracking/Anti-UAV-Tracking-V0'
GT_DIR = DATASET_ROOT
DEVICE = 'mps'
OUTPUT_DIR = 'avaliacao_avancada/hibrido_mog2'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/tmp', exist_ok=True)

VIDEO_IDS = [f'video{i:02d}' for i in range(1, 21)]
PROBLEM_VIDEOS = ['video07', 'video12', 'video16', 'video17', 'video19']

INIT_CONF = 0.6
SWITCH_CONF = 0.6
YOLO_PERIOD = 15
PREDICTION_HORIZONS = [5, 10, 15, 30]
DRIFT_THRESHOLDS = [100, 250, 500]
MIN_EPISODE_LEN = 6

# MOG2 parameters
BG_HISTORY = 100
BG_THRESH = 16
MOTION_RATIO_THRESH = 0.05
# ========================================================

detector = YOLO(MODEL_PATH)

def compute_iou(boxA, boxB):
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    xi1 = max(xA, xB); yi1 = max(yA, yB)
    xi2 = min(xA + wA, xB + wB); yi2 = min(yA + hA, yB + hB)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    areaA = wA * hA; areaB = wB * hB
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
            coco_dt.append({
                "image_id": f_idx, "category_id": 1,
                "bbox": [float(v) for v in pred[1:5]],
                "score": float(pred[5])
            })
    ap50 = ap = 0.0
    if len(coco_gt["annotations"]) > 0 and len(coco_dt) > 0:
        gt_json = f'{tmp_dir}/gt_{video_id}.json'
        dt_json = f'{tmp_dir}/dt_{video_id}.json'
        with open(gt_json, 'w') as f: json.dump(coco_gt, f)
        with open(dt_json, 'w') as f: json.dump(coco_dt, f)
        cocoGt = COCO(gt_json); cocoDt = cocoGt.loadRes(dt_json)
        evaluator = COCOeval(cocoGt, cocoDt, iouType='bbox')
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        ap50 = evaluator.stats[1]; ap = evaluator.stats[0]
        os.remove(gt_json); os.remove(dt_json)
    return ap50, ap

def calc_pred_errors_frame(gt_centers, measurements, horizon):
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    initialized = False
    errors = {}
    for frame in sorted(measurements.keys()):
        cx, cy = measurements[frame]
        if not initialized:
            kalman.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
            kalman.statePost = np.array([[cx],[cy],[0],[0]], np.float32)
            initialized = True
            continue
        kalman.predict()
        kalman.correct(np.array([[np.float32(cx)],[np.float32(cy)]]))
        state = kalman.statePost.copy()
        future_frame = frame + horizon
        if future_frame in gt_centers and gt_centers[future_frame] is not None:
            state_pred = state
            for _ in range(horizon):
                state_pred = kalman.transitionMatrix @ state_pred
            pred_x, pred_y = state_pred[0,0], state_pred[1,0]
            gt_cx, gt_cy = gt_centers[future_frame]
            err = np.sqrt((pred_x - gt_cx)**2 + (pred_y - gt_cy)**2)
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

def process_hybrid_mog2(video_id):
    video_path = os.path.join(DATASET_ROOT, video_id, f'{video_id}.mp4')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None

    gt_file = os.path.join(GT_DIR, video_id, f'{video_id}_gt.txt')
    gt_frames = {}; gt_centers = {}; gt_bbox_areas = []
    with open(gt_file, 'r') as f:
        for idx, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) != 4: continue
            x, y, w, h = map(float, parts)
            if x < 0 or y < 0:
                gt_frames[idx] = None; gt_centers[idx] = None
            else:
                gt_frames[idx] = [x, y, w, h]
                gt_centers[idx] = (x + w/2, y + h/2)
                gt_bbox_areas.append(w * h)

    back_sub = cv2.createBackgroundSubtractorMOG2(history=BG_HISTORY, varThreshold=BG_THRESH, detectShadows=False)
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    tracker = None; last_det_conf = 0.0
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kalman_initialized = False

    frame_count = 0; yolo_calls = 0
    preds = []; times_per_frame = []; center_errors = []
    measurements = {}  # frame -> (cx, cy) para predição

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        t0 = time.time()

        fg_mask = back_sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_morph)

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
                cx, cy = x1 + w/2, y1 + h/2
                kalman.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
                kalman.statePost = np.array([[cx],[cy],[0],[0]], np.float32)
                kalman_initialized = True
                t1 = time.time()
                preds.append((frame_count, x1, y1, w, h, conf, (t1 - t0)*1000))
                measurements[frame_count] = (cx, cy)
                # Erro de centro
                gt_bbox = gt_frames.get(frame_count)
                if gt_bbox is not None:
                    gt_cx, gt_cy = gt_bbox[0] + gt_bbox[2]/2, gt_bbox[1] + gt_bbox[3]/2
                    center_errors.append(np.sqrt((cx - gt_cx)**2 + (cy - gt_cy)**2))
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
                preds.append((frame_count, x, y, w, h, conf, (t1 - t0)*1000))
                cx, cy = x + w/2, y + h/2
                measurements[frame_count] = (cx, cy)
                if kalman_initialized:
                    kalman.predict()
                    kalman.correct(np.array([[np.float32(cx)],[np.float32(cy)]]))

                # Verificação de movimento
                x1_c, y1_c = max(0, x), max(0, y)
                x2_c, y2_c = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
                bbox_mask = fg_mask[y1_c:y2_c, x1_c:x2_c]
                motion_ratio = np.sum(bbox_mask > 0) / bbox_mask.size if bbox_mask.size > 0 else 0.0
                force_yolo = (motion_ratio < MOTION_RATIO_THRESH)

                if (frame_count % YOLO_PERIOD == 0) or force_yolo:
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
                            cx_n, cy_n = x1 + w_new/2, y1 + h_new/2
                            kalman.statePre = np.array([[cx_n],[cy_n],[0],[0]], np.float32)
                            kalman.statePost = np.array([[cx_n],[cy_n],[0],[0]], np.float32)
                            kalman_initialized = True
                # Erro de centro
                gt_bbox = gt_frames.get(frame_count)
                if gt_bbox is not None:
                    gt_cx, gt_cy = gt_bbox[0] + gt_bbox[2]/2, gt_bbox[1] + gt_bbox[3]/2
                    center_errors.append(np.sqrt((cx - gt_cx)**2 + (cy - gt_cy)**2))
                else:
                    center_errors.append(0.0)
            else:
                tracker = None; kalman_initialized = False
                t1 = time.time()

        times_per_frame.append((t1 - t0)*1000)

    cap.release()

    pred_dict = {}
    for p in preds:
        if p[0] not in pred_dict: pred_dict[p[0]] = p

    frames = sorted(set(list(gt_frames.keys()) + list(pred_dict.keys())))
    total_frames = len(frames)
    frames_com_drone = sum(1 for f in frames if gt_frames.get(f) is not None)
    sa_correct = 0; sr_count_05 = 0
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

    # Erros de predição
    error_stats = {}
    for h in PREDICTION_HORIZONS:
        errs = calc_pred_errors_frame(gt_centers, measurements, h)
        vals = list(errs.values())
        if vals:
            error_stats[f'erro_medio_h{h}'] = np.mean(vals)
            error_stats[f'erro_mediana_h{h}'] = np.median(vals)
            error_stats[f'erro_std_h{h}'] = np.std(vals)
        else:
            error_stats[f'erro_medio_h{h}'] = 0.0
            error_stats[f'erro_mediana_h{h}'] = 0.0
            error_stats[f'erro_std_h{h}'] = 0.0

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

def process_hybrid_sem_mog2(video_id):
    """Versão original do híbrido (sem MOG2) para coleta de dados frame a frame."""
    video_path = os.path.join(DATASET_ROOT, video_id, f'{video_id}.mp4')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None

    gt_file = os.path.join(GT_DIR, video_id, f'{video_id}_gt.txt')
    gt_frames = {}; gt_centers = {}
    with open(gt_file, 'r') as f:
        for idx, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) != 4: continue
            x, y, w, h = map(float, parts)
            if x < 0 or y < 0:
                gt_frames[idx] = None; gt_centers[idx] = None
            else:
                gt_frames[idx] = [x, y, w, h]
                gt_centers[idx] = (x + w/2, y + h/2)

    tracker = None; last_det_conf = 0.0
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kalman_initialized = False

    frame_count = 0; preds = []; measurements = {}
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        if tracker is None:
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
                cx, cy = x1 + w/2, y1 + h/2
                kalman.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
                kalman.statePost = np.array([[cx],[cy],[0],[0]], np.float32)
                kalman_initialized = True
                preds.append((frame_count, x1, y1, w, h, conf))
                measurements[frame_count] = (cx, cy)
        else:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                conf = last_det_conf
                preds.append((frame_count, x, y, w, h, conf))
                cx, cy = x + w/2, y + h/2
                measurements[frame_count] = (cx, cy)
                if kalman_initialized:
                    kalman.predict()
                    kalman.correct(np.array([[np.float32(cx)],[np.float32(cy)]]))
                if frame_count % YOLO_PERIOD == 0:
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
                            if preds: preds[-1] = (frame_count, x1, y1, w_new, h_new, best_conf)
                            cx_n, cy_n = x1 + w_new/2, y1 + h_new/2
                            kalman.statePre = np.array([[cx_n],[cy_n],[0],[0]], np.float32)
                            kalman.statePost = np.array([[cx_n],[cy_n],[0],[0]], np.float32)
                            kalman_initialized = True
            else:
                tracker = None; kalman_initialized = False

    cap.release()
    return preds, gt_frames, gt_centers, measurements

# ==================== EXECUÇÃO PRINCIPAL ====================
if __name__ == '__main__':
    # 1. Híbrido com MOG2 (todos os vídeos)
    print("Processando Híbrido com MOG2...")
    mog2_metrics = []
    mog2_data = {}
    for vid in VIDEO_IDS:
        res = process_hybrid_mog2(vid)
        if res is None: continue
        met, preds, gt_frames, gt_centers, meas, c_errs = res
        mog2_metrics.append(met)
        mog2_data[vid] = (preds, gt_frames, gt_centers, meas, c_errs)
        print(f"  {vid}: SA={met['SA']:.4f} AP@0.5={met['AP@0.5']:.4f} FPS={met['FPS']:.1f}")

    df_mog2 = pd.DataFrame(mog2_metrics)
    df_mog2.to_csv(f'{OUTPUT_DIR}/metricas_hibrido_mog2.csv', index=False)

    # 2. Híbrido sem MOG2 (apenas vídeos problemáticos) para gráficos
    print("\nProcessando Híbrido sem MOG2 (vídeos problemáticos)...")
    original_data = {}
    for vid in PROBLEM_VIDEOS:
        res = process_hybrid_sem_mog2(vid)
        if res:
            original_data[vid] = res
            print(f"  {vid} OK")

    # 3. Gráficos comparativos para vídeos problemáticos
    for vid in PROBLEM_VIDEOS:
        if vid not in mog2_data or vid not in original_data: continue
        preds_mog2, gt_f_mog2, gt_c_mog2, meas_mog2, c_errs_mog2 = mog2_data[vid]
        preds_orig, gt_f_orig, gt_c_orig, meas_orig = original_data[vid]

        # Função para extrair arrays alinhados
        def extract_frame_arrays(preds, gt_frames, gt_centers):
            pred_dict = {p[0]: p for p in preds}
            frames = sorted(gt_centers.keys())
            ious = []; errs_center = []
            for f in frames:
                gt_bbox = gt_frames.get(f)
                pred = pred_dict.get(f)
                if gt_bbox is None or pred is None:
                    ious.append(np.nan); errs_center.append(np.nan)
                else:
                    iou = compute_iou(gt_bbox, pred[1:5])
                    px, py = pred[1] + pred[3]/2, pred[2] + pred[4]/2
                    gt_cx, gt_cy = gt_centers[f]
                    err = np.sqrt((px - gt_cx)**2 + (py - gt_cy)**2)
                    ious.append(iou); errs_center.append(err)
            return frames, ious, errs_center

        frames_m, ious_m, errs_m = extract_frame_arrays(preds_mog2, gt_f_mog2, gt_c_mog2)
        frames_o, ious_o, errs_o = extract_frame_arrays(preds_orig, gt_f_orig, gt_c_orig)

        # IoU
        plt.figure(figsize=(12,5))
        plt.plot(frames_m, ious_m, label='Com MOG2', alpha=0.7)
        plt.plot(frames_o, ious_o, label='Sem MOG2', alpha=0.7)
        plt.title(f'{vid} – IoU (Híbrido com vs sem MOG2)')
        plt.xlabel('Frame'); plt.ylabel('IoU'); plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(f'{OUTPUT_DIR}/plots/{vid}_iou_comparacao.png'); plt.close()

        # Erro de centro
        plt.figure(figsize=(12,5))
        plt.plot(frames_m, errs_m, label='Com MOG2', alpha=0.7)
        plt.plot(frames_o, errs_o, label='Sem MOG2', alpha=0.7)
        plt.title(f'{vid} – Erro de Centro (Híbrido com vs sem MOG2)')
        plt.xlabel('Frame'); plt.ylabel('Erro (pixels)'); plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(f'{OUTPUT_DIR}/plots/{vid}_centro_comparacao.png'); plt.close()

        # Erro H15 (calculado para ambos)
        err_h15_m = calc_pred_errors_frame(gt_c_mog2, meas_mog2, 15)
        err_h15_o = calc_pred_errors_frame(gt_c_orig, meas_orig, 15)
        frames_h_m = sorted(err_h15_m.keys()); vals_h_m = [err_h15_m[f] for f in frames_h_m]
        frames_h_o = sorted(err_h15_o.keys()); vals_h_o = [err_h15_o[f] for f in frames_h_o]
        plt.figure(figsize=(12,5))
        plt.plot(frames_h_m, vals_h_m, label='Com MOG2', alpha=0.7)
        plt.plot(frames_h_o, vals_h_o, label='Sem MOG2', alpha=0.7)
        plt.title(f'{vid} – Erro Predição H15 (Híbrido com vs sem MOG2)')
        plt.xlabel('Frame'); plt.ylabel('Erro (pixels)'); plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(f'{OUTPUT_DIR}/plots/{vid}_h15_comparacao.png'); plt.close()

        # Salva CSVs com os dados frame a frame
        pd.DataFrame({'frame': frames_m, 'iou_mog2': ious_m, 'iou_orig': ious_o,
                      'err_centro_mog2': errs_m, 'err_centro_orig': errs_o}).to_csv(
            f'{OUTPUT_DIR}/plots/{vid}_frame_data.csv', index=False)

    print("\nGráficos comparativos gerados.")

    # 4. Tabela comparativa final (Híbrido, BoT-SORT, YOLO+Kalman, Híbrido+MOG2)
    print("\nGerando tabela comparativa final...")
    hib_path = 'avaliacao_avancada/metricas_hibrido.csv'
    bot_path = 'avaliacao_avancada/metricas_botsort.csv'
    yolo_path = 'avaliacao_avancada/yolo_kalman/metricas_yolo_kalman.csv'
    mog2_path = f'{OUTPUT_DIR}/metricas_hibrido_mog2.csv'

    dfs = []
    if os.path.exists(hib_path):
        df_hib = pd.read_csv(hib_path); df_hib['Metodo'] = 'Híbrido'; dfs.append(df_hib)
    if os.path.exists(bot_path):
        df_bot = pd.read_csv(bot_path); df_bot['Metodo'] = 'BoT-SORT'; dfs.append(df_bot)
    if os.path.exists(yolo_path):
        df_yolo = pd.read_csv(yolo_path); df_yolo['Metodo'] = 'YOLO+Kalman'; dfs.append(df_yolo)
    if os.path.exists(mog2_path):
        df_mog2 = pd.read_csv(mog2_path); df_mog2['Metodo'] = 'Híbrido+MOG2'; dfs.append(df_mog2)

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True, sort=False)
        metric_cols = ['SA', 'AP@0.5', 'AP@0.5:0.95', 'FPS',
                       'erro_medio_h5', 'erro_medio_h10', 'erro_medio_h15', 'erro_medio_h30',
                       'drift_ep_100_count', 'drift_ep_250_count', 'drift_ep_500_count']
        for col in metric_cols:
            if col not in df_all.columns: df_all[col] = np.nan
        summary = df_all.groupby('Metodo')[metric_cols].mean().round(4)
        summary.to_csv(f'{OUTPUT_DIR}/comparacao_final.csv')
        print("\nComparação final (médias globais):")
        print(summary.to_string())

    print(f"\n✅ Todos os resultados salvos em '{OUTPUT_DIR}'")