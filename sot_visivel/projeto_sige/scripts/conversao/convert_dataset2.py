import os, cv2, yaml

# ==================== CONFIGURAÇÕES ====================
DATASET_ROOT = '/Volumes/1T Ighor/Drone_Tracking/Datasets/Anti-UAV-Tracking/Anti-UAV-Tracking-V0'
OUTPUT_DIR   = '/Users/ighorribeiro/Downloads/anti_uav_tracking_test'   # Pode mudar
# ========================================================

os.makedirs(f'{OUTPUT_DIR}/images/test', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/labels/test', exist_ok=True)

CLASS_ID = 0
seq_candidates = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
print(f"Pastas encontradas: {len(seq_candidates)}")

total_frames = 0

for seq in seq_candidates:
    seq_path = os.path.join(DATASET_ROOT, seq)
    gt_files = [f for f in os.listdir(seq_path) if f.endswith('_gt.txt')]
    if not gt_files:
        print(f"AVISO: nenhum *_gt.txt em {seq}, pulando.")
        continue
    gt_file = gt_files[0]
    txt_path = os.path.join(seq_path, gt_file)

    with open(txt_path, 'r') as f:
        annotations = [line.strip() for line in f if line.strip()]

    # Lista todas as imagens .jpg ordenadas (ex: 000001.jpg, 000002.jpg ...)
    images = sorted([f for f in os.listdir(seq_path) if f.lower().endswith('.jpg')])

    print(f"  {seq}: {len(images)} imagens, {len(annotations)} anotações", end="")

    # 🔧 Ajuste automático: se anotações são metade das imagens, usamos apenas imagens pares
    if len(images) == 2 * len(annotations):
        images = images[::2]          # pega índice 0,2,4,...
        print(" → usando apenas imagens pares (metade)")
    elif len(annotations) == 2 * len(images):
        annotations = annotations[::2]
        print(" → usando apenas anotações pares")
    else:
        print(" → ajuste manual necessário, pulando")
        continue

    # Agora images e annotations têm o mesmo tamanho
    for img_name, ann_line in zip(images, annotations):
        img_path = os.path.join(seq_path, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        parts = ann_line.split()
        if len(parts) != 4:
            continue
        x, y, bw, bh = map(int, parts)

        # YOLO normalizado
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        norm_w = bw / w
        norm_h = bh / h

        out_name = f"{seq}_{img_name}"      # ex: video20_000001.jpg
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'images', 'test', out_name), frame)

        label_line = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"
        with open(os.path.join(OUTPUT_DIR, 'labels', 'test', out_name.replace('.jpg', '.txt')), 'w') as f:
            f.write(label_line)

        total_frames += 1

print(f"\n✅ Total de frames convertidos: {total_frames}")

# Cria dataset.yaml
yaml_path = os.path.join(OUTPUT_DIR, 'dataset.yaml')
config = {
    'path': OUTPUT_DIR,
    'train': '',
    'val': '',
    'test': 'images/test',
    'nc': 1,
    'names': ['drone']
}
with open(yaml_path, 'w') as f:
    yaml.dump(config, f)
print(f"Dataset de teste salvo em {OUTPUT_DIR}")