import os, cv2, yaml

# ==================== CONFIGURAÇÕES ====================
DATASET_ROOT = '/Volumes/1T Ighor/Drone_Tracking/Datasets/Anti-UAV-Tracking/Anti-UAV-Tracking-V0'
OUTPUT_DIR   = '/Users/ighorribeiro/Downloads/anti_uav_tracking_test_final'
# ========================================================

os.makedirs(f'{OUTPUT_DIR}/images/test', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/labels/test', exist_ok=True)

CLASS_ID = 0
total_frames = 0

for video_folder in sorted(os.listdir(DATASET_ROOT)):
    video_path = os.path.join(DATASET_ROOT, video_folder)
    if not os.path.isdir(video_path):
        continue

    # Pega o arquivo _gt.txt (ignora _gt_first.txt)
    gt_files = [f for f in os.listdir(video_path) if f.endswith('_gt.txt') and '_first' not in f]
    if not gt_files:
        continue
    gt_file = gt_files[0]
    gt_path = os.path.join(video_path, gt_file)

    with open(gt_path, 'r') as f:
        annotations = [line.strip() for line in f if line.strip()]

    # Lista todas as imagens .jpg reais (ignora arquivos iniciados por ._)
    images = sorted([f for f in os.listdir(video_path) if f.lower().endswith('.jpg') and not f.startswith('._')])

    # Verifica se a quantidade confere
    if len(images) != len(annotations):
        print(f"⚠️ {video_folder}: {len(images)} imagens ≠ {len(annotations)} anotações. Pulando.")
        continue

    print(f"{video_folder}: {len(images)} frames (1:1)")

    for img_file, ann_line in zip(images, annotations):
        img_path = os.path.join(video_path, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        out_name = f"{video_folder}_{img_file}"
        out_img_path = os.path.join(OUTPUT_DIR, 'images', 'test', out_name)
        out_lbl_path = os.path.join(OUTPUT_DIR, 'labels', 'test', out_name.replace('.jpg', '.txt'))

        cv2.imwrite(out_img_path, frame)

        # Trata anotação de fundo (-100 -100 -100 -100) ou coordenadas negativas
        parts = ann_line.split()
        if len(parts) != 4:
            open(out_lbl_path, 'w').close()
            total_frames += 1
            continue

        x, y, bw, bh = map(float, parts)
        if x < 0 or y < 0 or bw < 0 or bh < 0:
            open(out_lbl_path, 'w').close()   # label vazio (fundo)
        else:
            x_center = (x + bw/2) / w
            y_center = (y + bh/2) / h
            norm_w = bw / w
            norm_h = bh / h
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            norm_w = max(0, min(1, norm_w))
            norm_h = max(0, min(1, norm_h))
            with open(out_lbl_path, 'w') as f:
                f.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        total_frames += 1

print(f"\n✅ Conversão finalizada. Total de frames: {total_frames}")

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

print(f"Arquivo dataset.yaml criado em {yaml_path}")