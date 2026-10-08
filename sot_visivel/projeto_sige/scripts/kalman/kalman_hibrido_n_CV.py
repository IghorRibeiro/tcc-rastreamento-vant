import os, cv2, time, csv, json
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==================== CONFIGURAÇÕES ====================
EXPERIMENT_NAME = "Hibrido_YOLO26n_CSRT_Trigger_Kalman_CV"
DATASET_NAME = "Anti-UAV-Tracking-V0"
MODEL_PATH = '/Users/ighorribeiro/PycharmProjects/TesteYolo26/pesos_treinados/yolo26n_dut_best.pt'
DATASET_ROOT = '/Volumes/1T Ighor/Drone_Tracking/Datasets/Anti-UAV-Tracking/Anti-UAV-Tracking-V0'
GT_DIR = DATASET_ROOT
IMAGES_DIR = '/Users/ighorribeiro/PycharmProjects/TesteYolo26/datasets/anti_uav_tracking_test_final/images/test'
OUTPUT_DIR = 'resultados_avaliacao_CV'
DEVICE = 'mps'

INIT_CONF = 0.6
SWITCH_CONF = 0.6
YOLO_PERIOD = 15

# NOVO: horizontes estendidos (1,2,3,4 segundos a 30 fps)
PREDICTION_HORIZONS = [30, 60, 90, 120]   # 1s, 2s, 3s, 4s
# ========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
detector = YOLO(MODEL_PATH)
VIDEO_IDS = [f'video{i:02d}' for i in range(1, 21)]

all_pred_errors = {h: [] for h in PREDICTION_HORIZONS}
all_innovations = {video_id: [] for video_id in VIDEO_IDS}  # NOVO: inovação por vídeo

def compute_iou(boxA, boxB):
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    xi1 = max(xA, xB); yi1 = max(yA, yB)
    xi2 = min(xA + wA, xB + wB); yi2 = min(yA + hA, yB + hB)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    areaA = wA * hA; areaB = wB * hB
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def run_hybrid_and_evaluate():
    all_metrics = []

    for video_id in VIDEO_IDS:
        video_path = os.path.join(DATASET_ROOT, video_id, f'{video_id}.mp4')
        if not os.path.exists(video_path):
            print(f"AVISO: {video_path} não encontrado. Pulando.")
            continue

        print(f"\nProcessando {video_id} ...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erro ao abrir {video_path}")
            continue

        # Ground truth
        gt_file = os.path.join(GT_DIR, video_id, f'{video_id}_gt.txt')
        gt_frames = {}
        gt_centers = {}
        with open(gt_file, 'r') as f:
            for idx, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                x, y, w, h = map(float, parts)
                if x < 0 or y < 0:
                    gt_frames[idx] = None
                    gt_centers[idx] = None
                else:
                    gt_frames[idx] = [x, y, w, h]
                    gt_centers[idx] = (x + w/2, y + h/2)

        tracker = None
        last_det_conf = 0.0
        # Kalman CV (4 estados)
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kalman_initialized = False

        frame_count = 0
        yolo_calls = 0
        preds = []
        times_per_frame = []
        pred_errors = {h: [] for h in PREDICTION_HORIZONS}
        innovations = []  # NOVO: lista de inovações (distância euclidiana)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
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
                    cx, cy = x1 + w/2, y1 + h/2
                    kalman.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
                    kalman.statePost = np.array([[cx],[cy],[0],[0]], np.float32)
                    kalman_initialized = True
                    t1 = time.time()
                    preds.append((frame_count, x1, y1, w, h, conf, (t1 - t0)*1000))
                    # NOVO: inovação na inicialização é zero
                    innovations.append(0.0)
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
                    if kalman_initialized:
                        # NOVO: calcula inovação (distância entre medição e predição antes da correção)
                        prediction = kalman.predict()  # retorna estado predito [x,y,vx,vy]
                        pred_meas = np.array([[prediction[0,0]], [prediction[1,0]]])  # medição predita
                        meas = np.array([[np.float32(cx)], [np.float32(cy)]])         # medição real
                        innov = np.linalg.norm(meas - pred_meas)
                        innovations.append(float(innov))
                        # Corrige
                        kalman.correct(meas)
                    else:
                        innovations.append(0.0)

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
                                if preds:
                                    preds[-1] = (frame_count, x1, y1, w_new, h_new, best_conf, preds[-1][6])
                                cx_n, cy_n = x1 + w_new/2, y1 + h_new/2
                                kalman.statePre = np.array([[cx_n],[cy_n],[0],[0]], np.float32)
                                kalman.statePost = np.array([[cx_n],[cy_n],[0],[0]], np.float32)
                                kalman_initialized = True
                else:
                    tracker = None
                    kalman_initialized = False
                    t1 = time.time()
                    innovations.append(0.0)  # NOVO: sem inovação quando tracker falha

            times_per_frame.append((t1 - t0)*1000)

            # Predição de trajetória (NOVO: usando horizontes estendidos)
            if kalman_initialized:
                state = kalman.statePost.copy()
                current_frame = frame_count
                for horizon in PREDICTION_HORIZONS:
                    future_frame = current_frame + horizon
                    # Propaga estado
                    state_pred = state
                    for _ in range(horizon):
                        state_pred = kalman.transitionMatrix @ state_pred
                    pred_x, pred_y = state_pred[0,0], state_pred[1,0]
                    if future_frame in gt_centers and gt_centers[future_frame] is not None:
                        gt_cx, gt_cy = gt_centers[future_frame]
                        error = np.sqrt((pred_x - gt_cx)**2 + (pred_y - gt_cy)**2)
                        pred_errors[horizon].append(error)
                        all_pred_errors[horizon].append((video_id, current_frame, error))

        cap.release()
        all_innovations[video_id] = innovations  # NOVO

        # ===== CÁLCULO DAS MÉTRICAS (igual ao anterior) =====
        pred_dict = {}
        for p in preds:
            if p[0] not in pred_dict:
                pred_dict[p[0]] = p

        frames = sorted(set(list(gt_frames.keys()) + list(pred_dict.keys())))
        total_frames = len(frames)
        frames_com_drone = sum(1 for f in frames if gt_frames.get(f) is not None)
        sa_correct = 0
        iou_thresholds = np.linspace(0, 1, 21)
        sr_counts = {th: 0 for th in iou_thresholds}

        for frame in frames:
            gt_bbox = gt_frames.get(frame)
            pred = pred_dict.get(frame)
            pred_bbox = None if pred is None else pred[1:5]
            conf = 0 if pred is None else pred[5]

            if gt_bbox is not None:
                if pred_bbox is not None:
                    iou = compute_iou(gt_bbox, pred_bbox)
                    for th in iou_thresholds:
                        if iou >= th:
                            sr_counts[th] += 1
                    if iou >= 0.5:
                        sa_correct += 1
            else:
                if pred is None or conf < 0.25:
                    sa_correct += 1

        sa = sa_correct / total_frames if total_frames > 0 else 0
        sr_curve = {th: sr_counts[th] / frames_com_drone if frames_com_drone else 0 for th in iou_thresholds}
        x_auc = np.array(list(sr_curve.keys()))
        y_auc = np.array(list(sr_curve.values()))
        if hasattr(np, 'trapz'):
            sr_auc = np.trapz(y_auc, x_auc)
        else:
            sr_auc = np.sum(np.diff(x_auc) * (y_auc[:-1] + y_auc[1:])) / 2.0

        tempo_medio = np.mean(times_per_frame) if times_per_frame else 0
        fps = 1000 / tempo_medio if tempo_medio > 0 else 0
        yolo_calls_percent = (yolo_calls / frame_count) * 100 if frame_count > 0 else 0

        # AP com pycocotools
        coco_gt = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "drone"}]}
        coco_dt = []
        ann_id = 0
        for f_idx, frame in enumerate(frames, start=1):
            gt_bbox = gt_frames.get(frame)
            pred = pred_dict.get(frame)
            coco_gt["images"].append({"id": f_idx})
            if gt_bbox is not None:
                coco_gt["annotations"].append({
                    "id": ann_id,
                    "image_id": f_idx,
                    "category_id": 1,
                    "bbox": gt_bbox,
                    "area": gt_bbox[2] * gt_bbox[3],
                    "iscrowd": 0,
                    "ignore": 0
                })
                ann_id += 1
            if pred is not None:
                pred_bbox = pred[1:5]
                conf = pred[5]
                coco_dt.append({
                    "image_id": f_idx,
                    "category_id": 1,
                    "bbox": list(pred_bbox),
                    "score": conf
                })

        ap50 = 0.0
        ap = 0.0
        if len(coco_gt["annotations"]) > 0 and len(coco_dt) > 0:
            gt_json = f'{OUTPUT_DIR}/temp_gt_{video_id}.json'
            dt_json = f'{OUTPUT_DIR}/temp_dt_{video_id}.json'
            with open(gt_json, 'w') as f:
                json.dump(coco_gt, f)
            with open(dt_json, 'w') as f:
                json.dump(coco_dt, f)
            cocoGt = COCO(gt_json)
            cocoDt = cocoGt.loadRes(dt_json)
            evaluator = COCOeval(cocoGt, cocoDt, iouType='bbox')
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            ap50 = evaluator.stats[1]
            ap = evaluator.stats[0]
            os.remove(gt_json)
            os.remove(dt_json)

        # Métricas de predição
        pred_metrics = {}
        for h in PREDICTION_HORIZONS:
            errs = pred_errors[h]
            if len(errs) > 0:
                mean_err = np.mean(errs)
                median_err = np.median(errs)
                std_err = np.std(errs)
                max_err = np.max(errs)
                p95_err = np.percentile(errs, 95)
            else:
                mean_err = median_err = std_err = max_err = p95_err = 0.0
            pred_metrics[f'erro_medio_h{h}'] = mean_err
            pred_metrics[f'erro_mediana_h{h}'] = median_err
            pred_metrics[f'erro_std_h{h}'] = std_err
            pred_metrics[f'erro_max_h{h}'] = max_err
            pred_metrics[f'erro_p95_h{h}'] = p95_err

        # NOVO: inovação média e std para este vídeo
        innov_array = np.array(innovations)
        innov_mean = np.mean(innov_array) if len(innov_array) > 0 else 0.0
        innov_std = np.std(innov_array) if len(innov_array) > 0 else 0.0

        print(f"  {video_id}: SA={sa:.4f}, SR_AUC={sr_auc:.4f}, AP@0.5={ap50:.4f}, AP={ap:.4f}, "
              f"Tempo={tempo_medio:.2f} ms, FPS={fps:.1f}, YOLO={yolo_calls} ({yolo_calls_percent:.1f}%)")
        print(f"    Inovação: média={innov_mean:.2f} px, std={innov_std:.2f} px")
        for h in PREDICTION_HORIZONS:
            print(f"    Predição h={h}: erro médio={pred_metrics[f'erro_medio_h{h}']:.2f} px, mediana={pred_metrics[f'erro_mediana_h{h}']:.2f} px, "
                  f"std={pred_metrics[f'erro_std_h{h}']:.2f} px, max={pred_metrics[f'erro_max_h{h}']:.2f} px, p95={pred_metrics[f'erro_p95_h{h}']:.2f} px")

        row = {
            'experimento': EXPERIMENT_NAME,
            'dataset': DATASET_NAME,
            'video': video_id,
            'SA': sa,
            'AP@0.5': ap50,
            'AP@0.5:0.95': ap,
            'SR_AUC': sr_auc,
            'tempo_medio_ms': tempo_medio,
            'FPS': fps,
            'yolo_calls': yolo_calls,
            'yolo_calls_percent': yolo_calls_percent,
            'inovacao_media': innov_mean,
            'inovacao_std': innov_std,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        row.update(pred_metrics)
        all_metrics.append(row)

    return all_metrics

# ==================== EXECUÇÃO ====================
if __name__ == '__main__':
    results = run_hybrid_and_evaluate()

    # CSV consolidado
    csv_path = os.path.join(OUTPUT_DIR, f'metricas_{EXPERIMENT_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Métricas agregadas salvas em '{csv_path}'")

    # CSVs de erros individuais por horizonte
    for h in PREDICTION_HORIZONS:
        err_csv = os.path.join(OUTPUT_DIR, f'erros_predicao_h{h}.csv')
        with open(err_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_id', 'frame', 'erro_px'])
            for video_id, frame, error in all_pred_errors[h]:
                writer.writerow([video_id, frame, error])
        print(f"📊 Erros individuais (h={h}) salvos em '{err_csv}' ({len(all_pred_errors[h])} amostras)")

    # NOVO: Salvar inovações por vídeo
    innov_csv = os.path.join(OUTPUT_DIR, 'inovacoes_por_video.csv')
    with open(innov_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_id', 'frame', 'inovacao_px'])
        for video_id, innovations in all_innovations.items():
            for frame_idx, innov in enumerate(innovations, start=1):
                writer.writerow([video_id, frame_idx, innov])
    print(f"📊 Inovações salvas em '{innov_csv}'")