from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data="configs/data.yaml",
    epochs=2,
    imgsz=640,
    batch=8,
    device="mps",
    patience=20,
    mosaic=0.3,
    name="yolo26n_ir_drone",
)