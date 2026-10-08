import os
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path
import shutil

# ==================== CONFIGURAÇÕES ====================
# Pasta raiz do dataset original (contém train, val, test)
DATASET_ORIGINAL = '/Users/ighorribeiro/Downloads/DUT Anti-UAV Detection and Tracking'

# Pasta de saída para o formato YOLO
OUTPUT_DIR = '/Users/ighorribeiro/Downloads/DUT_YOLO'

# Splits a converter
SPLITS = ['train', 'val', 'test']

# Nome da classe única
CLASS_NAME = 'UAV'
CLASS_ID = 0
# ========================================================

# Cria estrutura de pastas
for split in SPLITS:
    os.makedirs(f'{OUTPUT_DIR}/images/{split}', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/labels/{split}', exist_ok=True)

total_imagens = 0

for split in SPLITS:
    img_dir = os.path.join(DATASET_ORIGINAL, split, 'img')
    xml_dir = os.path.join(DATASET_ORIGINAL, split, 'xml')

    if not os.path.isdir(img_dir) or not os.path.isdir(xml_dir):
        print(f"AVISO: {split} não contém img/ ou xml/. Pulando.")
        continue

    # Lista todos os arquivos XML
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"\nProcessando {split}: {len(xml_files)} imagens")

    for xml_file in xml_files:
        # Caminho completo do XML
        xml_path = os.path.join(xml_dir, xml_file)

        # Nome base da imagem (ex: 00001.jpg)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            filename = root.find('filename').text
        except Exception as e:
            print(f"Erro ao ler {xml_path}: {e}")
            continue

        # Caminho da imagem original
        img_path_original = os.path.join(img_dir, filename)
        if not os.path.exists(img_path_original):
            print(f"Imagem {filename} não encontrada em {img_dir}. Pulando.")
            continue

        # Copia a imagem para o destino
        img_dest = os.path.join(OUTPUT_DIR, 'images', split, filename)
        shutil.copy(img_path_original, img_dest)

        # Lê dimensões da imagem
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # Coleta objetos (bounding boxes)
        labels = []
        for obj in root.findall('object'):
            name = obj.find('name').text.strip()
            if name != CLASS_NAME:
                continue  # só a classe UAV
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Converte para YOLO: class_id x_center y_center width height (normalizados)
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            # Clampe para garantir que fique entre 0 e 1
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w = max(0, min(1, w))
            h = max(0, min(1, h))

            labels.append(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # Escreve arquivo de label
        label_filename = filename.replace('.jpg', '.txt')
        label_path = os.path.join(OUTPUT_DIR, 'labels', split, label_filename)
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))

        total_imagens += 1

print(f"\n✅ Conversão concluída! Total de imagens processadas: {total_imagens}")

# Cria o dataset.yaml
yaml_path = os.path.join(OUTPUT_DIR, 'dataset.yaml')
config = {
    'path': OUTPUT_DIR,
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': 1,
    'names': [CLASS_NAME]
}

with open(yaml_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"Arquivo dataset.yaml criado em {yaml_path}")
print("Conteúdo do dataset.yaml:")
print(open(yaml_path).read())