import os, cv2, time
from ultralytics import YOLO
import numpy as np

# ==================== CONFIGURAÇÕES ====================
MODEL_PATH = '/Users/ighorribeiro/PycharmProjects/TesteYolo26/pesos_treinados/yolo26n_dut_best.pt'
DATASET_ROOT = '/Volumes/1T Ighor/Drone_Tracking/Datasets/Anti-UAV-Tracking/Anti-UAV-Tracking-V0'
OUTPUT_DIR = '../avaliacao_tracking/predicoes/kcf_hibrido'  # 👈 saída padronizada

INIT_CONF_THRESHOLD = 0.5          # confiança para iniciar o tracking local
SWITCH_CONF_THRESHOLD = 0.7        # confiança para substituir o tracker pelo YOLO
YOLO_PERIOD = 30                   # reavaliar com YOLO a cada N frames
DEVICE = 'mps'                     # ou 'cpu'
# ========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
detector = YOLO(MODEL_PATH)
VIDEO_IDS = [f'video{i:02d}' for i in range(1, 21)]

for video_id in VIDEO_IDS:
    video_path = os.path.join(DATASET_ROOT, video_id, f'{video_id}.mp4')
    if not os.path.exists(video_path):
        print(f"AVISO: {video_path} não encontrado. Pulando.")
        continue

    print(f"Processando {video_id} ...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir {video_path}")
        continue

    frame_count = 0
    tracker = None
    last_det_conf = 0.0
    preds = []  # (frame, x, y, w, h, conf, tempo_ms)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        t0 = time.time()

        # ----- Modo Global (YOLO) -----
        if tracker is None:
            results = detector(frame, device=DEVICE, conf=INIT_CONF_THRESHOLD, verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().numpy()
                best_idx = confs.argmax()
                x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
                conf = float(confs[best_idx])
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                # Inicializa o KCF
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (x1, y1, w, h))
                last_det_conf = conf
                t1 = time.time()
                preds.append((frame_count, x1, y1, w, h, conf, (t1 - t0) * 1000))
            # else: sem detecção → não adiciona predição
        else:
            # ----- Modo Local (KCF) -----
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                # KCF não fornece confiança; usamos a última do YOLO
                conf = last_det_conf
                t1 = time.time()
                preds.append((frame_count, x, y, w, h, conf, (t1 - t0) * 1000))

                # Reavaliação periódica com YOLO
                if frame_count % YOLO_PERIOD == 0:
                    results = detector(frame, device=DEVICE, conf=SWITCH_CONF_THRESHOLD, verbose=False)
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        confs = boxes.conf.cpu().numpy()
                        best_conf = float(confs.max())
                        if best_conf > SWITCH_CONF_THRESHOLD:
                            best_idx = confs.argmax()
                            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            w_new, h_new = x2 - x1, y2 - y1
                            # Reinicializa o KCF
                            tracker = cv2.TrackerKCF_create()
                            tracker.init(frame, (x1, y1, w_new, h_new))
                            last_det_conf = best_conf
                            # Substitui a última predição (que era do KCF) pela do YOLO
                            if preds:
                                preds[-1] = (frame_count, x1, y1, w_new, h_new, best_conf, preds[-1][6])
            else:
                # KCF perdeu o objeto → volta ao modo global
                tracker = None
                last_det_conf = 0.0
                # Não adiciona predição

    cap.release()

    # Salva no formato padronizado: frame, x, y, w, h, conf, tempo
    output_file = os.path.join(OUTPUT_DIR, f'{video_id}.txt')
    with open(output_file, 'w') as f:
        for det in preds:
            frame, x, y, w, h, conf, tempo = det
            f.write(f"{frame},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},{tempo:.2f}\n")

    print(f"   {video_id}: {frame_count} frames, {len(preds)} predições")

print(f"\n✅ Trajetórias KCF salvas em '{OUTPUT_DIR}'")