import json, os

# ================= CONFIGURAÇÕES =================
# Pasta onde estão os JSONs do híbrido (saída do avaliar_hibrido_v2.py)
HIBRIDO_JSON_DIR = 'outputs/trajetorias_hibrido_v2'
# Pasta onde estão os TXT do BoT-SORT (experimento 3, por exemplo)
BOTSORT_TXT_DIR = 'Experimentos/exp03_yolo26n_botsort_ajustado/outputs/trajetorias'

# Destino padronizado
DEST_DIR = 'avaliacao_tracking/avaliacao_tracking/predicoes/hibrido_csrt'
DEST_DIR_BOTSORT = 'avaliacao_tracking/predicoes/exp03_botsort_ajustado'
# =================================================

def convert_json_to_txt(json_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith('_preds.json'):
            continue
        video_id = fname.replace('_preds.json', '')
        with open(os.path.join(json_dir, fname), 'r') as f:
            preds = json.load(f)

        out_path = os.path.join(dest_dir, f'{video_id}.txt')
        with open(out_path, 'w') as out:
            for p in preds:
                # p pode ter 7 ou 8 elementos
                if len(p) >= 8:
                    frame, track_id, x, y, w, h, conf, tempo = p[:8]
                else:  # apenas 7 (sem tempo)
                    frame, track_id, x, y, w, h, conf = p[:7]
                    tempo = 0.0
                # Escreve no formato padronizado: frame, x, y, w, h, conf, tempo
                out.write(f"{int(frame)},{float(x):.2f},{float(y):.2f},{float(w):.2f},{float(h):.2f},{float(conf):.4f},{float(tempo):.2f}\n")
        print(f"Convertido {video_id}")

def convert_botsort_txt_to_padrao(txt_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for fname in sorted(os.listdir(txt_dir)):
        if not fname.endswith('.txt'):
            continue
        video_id = fname.replace('.txt', '')
        in_path = os.path.join(txt_dir, fname)
        out_path = os.path.join(dest_dir, f'{video_id}.txt')
        with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
            for line in fin:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                # Formato original: frame, track_id, x, y, w, h, conf, -1, -1, -1
                frame = int(parts[0])
                # track_id = parts[1]  # ignorado
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                # Tempo não disponível → 0.0
                fout.write(f"{frame},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},0.0\n")
        print(f"Convertido {video_id}")

# 1. Converte híbrido
convert_json_to_txt(HIBRIDO_JSON_DIR, DEST_DIR)

# 2. Converte BoT-SORT (experimento 3)
convert_botsort_txt_to_padrao(BOTSORT_TXT_DIR, DEST_DIR_BOTSORT)