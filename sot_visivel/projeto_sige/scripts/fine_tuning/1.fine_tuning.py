from ultralytics import YOLO
import torch

# Caminho do dataset (visible, frameskip=30)
DATA_YAML = '/Users/ighorribeiro/Downloads/DUT_YOLO/dataset.yaml'

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Dispositivo: {device}")

print("\n=== Treinando YOLO26n ===")
model = YOLO('../yolo26n.pt')

model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=8,
    device=device,
    workers=0,
    amp=False,           # obrigatório para não travar no MPS
    cache=False,
    project='fine_tune_drones_DUT',
    name='yolo26_n',
    exist_ok=True,
    patience=10,
    save_period=10,
    pretrained=True,
    optimizer='auto',
    val=True
)