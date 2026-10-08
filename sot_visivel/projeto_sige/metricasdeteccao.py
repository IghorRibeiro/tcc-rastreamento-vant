from ultralytics import YOLO
import time
import csv
from pathlib import Path

DATASET = "/Users/ighorribeiro/PycharmProjects/TesteYolo26/datasets/anti_uav_tracking_test_final/dataset.yaml"

MODELOS = {
    "YOLO26n":
        "/Users/ighorribeiro/PycharmProjects/TesteYolo26/pesos_treinados/yolo26n_dut_best.pt",

    "YOLO26s":
        "/Users/ighorribeiro/PycharmProjects/TesteYolo26/pesos_treinados/yolo26s_dut_best.pt",

    "YOLO26m":
        "/Users/ighorribeiro/PycharmProjects/TesteYolo26/pesos_treinados/yolo26m_dut_best.pt"
}

print("\n" + "="*70)
print("VALIDAÇÃO DOS DETECTORES YOLO26")
print("="*70)

for nome, peso in MODELOS.items():

    print(f"\nValidando {nome}...")

    model = YOLO(peso)

    t0 = time.perf_counter()
    metrics = model.val(
        data=DATASET,
        split="test",
        verbose=False
    )
    elapsed = time.perf_counter() - t0

    print("\n" + "-"*50)
    print(nome)
    print("-"*50)

    print(f"Precision     : {metrics.box.mp:.4f}")
    print(f"Recall        : {metrics.box.mr:.4f}")
    print(f"mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95  : {metrics.box.map:.4f}")
    print(f"Tempo total (s): {elapsed:.2f}")
    print(f"Tempo por imagem (ms, aprox.): {(elapsed / 24804.0) * 1000:.3f}")

print("\nConcluído.")