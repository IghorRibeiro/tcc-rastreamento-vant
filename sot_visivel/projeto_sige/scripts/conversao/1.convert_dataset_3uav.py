import os, json, cv2
import numpy as np

# ==================== CONFIGURAÇÕES ====================
DATASET_ROOT = '/Users/ighorribeiro/Downloads/Anti-UAV-RGBT'  # Ajuste o caminho!
OUTPUT_DIR = '/Users/ighorribeiro/Downloads/Anti_uav_3uav_visible'  # Pasta de saída
FRAME_SKIP = 1  # 1 = todos os frames
MODALITIES = ['visible']
# ========================================================

DATA_SPLITS = ['train', 'val', 'test']

for split in DATA_SPLITS:
    os.makedirs(f'{OUTPUT_DIR}/images/{split}', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/labels/{split}', exist_ok=True)

CLASS_ID = 0  # drone

for split_name in DATA_SPLITS:
    split_path = os.path.join(DATASET_ROOT, split_name)
    if not os.path.isdir(split_path):
        print(f"Split '{split_name}' não encontrado, pulando.")
        continue

    seqs = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    print(f"{split_name}: {len(seqs)} sequências encontradas.")

    for seq in seqs:
        seq_path = os.path.join(split_path, seq)
        for mod in MODALITIES:
            video_file = os.path.join(seq_path, f'{mod}.mp4')
            json_file = os.path.join(seq_path, f'{mod}.json')
            if not os.path.exists(video_file) or not os.path.exists(json_file):
                continue

            print(f"  {split_name}/{seq}/{mod} ...")
            with open(json_file) as f:
                data = json.load(f)

            exist = data['exist']
            gt_rect = data['gt_rect']

            # --- Detectar o formato de gt_rect ---
            # Se algum elemento for uma lista com 4 números, é o formato indexado (mesmo tamanho de exist, com [] nos frames sem objeto)
            # Se todos os elementos forem listas de 4 números e o tamanho for menor que exist, é o formato sequencial (apenas objetos)

            # Heurística: se o tamanho for igual ao de exist, é indexado; se for menor, é sequencial
            if len(gt_rect) == len(exist):
                formato = "indexado"  # gt_rect[i] pode ser [] se não houver objeto
            else:
                formato = "sequencial"  # gt_rect contém apenas as bboxes dos objetos

            if formato == "sequencial":
                gt_idx = 0  # índice para percorrer gt_rect sequencialmente

            cap = cv2.VideoCapture(video_file)
            frame_count = 0
            extracted = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                # Pular frames conforme FRAME_SKIP
                if (frame_count - 1) % FRAME_SKIP != 0:
                    continue

                # Verificar se o frame está dentro do range de 'exist'
                if frame_count > len(exist):
                    break

                # Obter bbox conforme o formato
                if formato == "indexado":
                    if exist[frame_count - 1] == 0:
                        continue  # frame sem objeto
                    bbox = gt_rect[frame_count - 1]
                    if not bbox or len(bbox) != 4:  # lista vazia ou inválida
                        continue
                else:  # sequencial
                    if exist[frame_count - 1] == 0:
                        continue
                    if gt_idx >= len(gt_rect):
                        print(f"    Aviso: gt_rect insuficiente no frame {frame_count} (sequencial)")
                        break
                    bbox = gt_rect[gt_idx]
                    gt_idx += 1
                    if not bbox or len(bbox) != 4:
                        continue

                # Dimensões da imagem
                h, w = frame.shape[:2]
                x, y, bw, bh = bbox

                # Converter para YOLO normalizado
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h

                # Nome único do arquivo
                base = f"{split_name}_{seq}_{mod}_{frame_count:06d}"
                img_name = f"{base}.jpg"
                txt_name = f"{base}.txt"

                # Salvar imagem e label
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'images', split_name, img_name), frame)
                label = f"{CLASS_ID} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"
                with open(os.path.join(OUTPUT_DIR, 'labels', split_name, txt_name), 'w') as f_label:
                    f_label.write(label)

                extracted += 1

            cap.release()
            print(f"    {frame_count} frames lidos, {extracted} extraídos.")

print("\n✅ Conversão concluída!")