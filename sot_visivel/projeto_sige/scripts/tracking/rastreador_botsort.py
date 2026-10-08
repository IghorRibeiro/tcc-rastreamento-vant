from ultralytics import YOLO
import cv2

# ==================== CONFIGURAÇÕES ====================
# Modelo treinado (YOLO26n no DUT)
MODEL_PATH = 'pesos_treinados/yolo26n_dut_best.pt'

# Vídeo de entrada (substitua pelo caminho do seu vídeo)
VIDEO_PATH = '/Volumes/1T Ighor/Drone_Tracking/Datasets/Anti-UAV-Tracking/Anti-UAV-Tracking-V0/video07/video07.mp4'

# Nome do vídeo de saída
OUTPUT_VIDEO = 'output_tracking.mp4'

# Limiar de confiança para exibir as detecções
CONF_THRESHOLD = 0.25
# ========================================================

# Carrega o modelo
model = YOLO(MODEL_PATH)

# Abre o vídeo
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Erro ao abrir vídeo: {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Vídeo: {width}x{height}, FPS: {fps:.2f}")

# Configurar o gravador de vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# Loop de rastreamento
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rastreamento com BoT‑SORT (persist=True mantém os IDs entre frames)
    results = model.track(
        frame,
        device='mps',         # ou 'cpu' se MPS falhar
        persist=True,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    # Desenha as caixas e IDs no frame (o método plot já inclui os IDs)
    annotated_frame = results[0].plot()

    # Salva no vídeo de saída
    out.write(annotated_frame)

    # Exibe em tempo real
    cv2.imshow('YOLO26n + BoT-SORT', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Vídeo rastreado salvo como '{OUTPUT_VIDEO}'")