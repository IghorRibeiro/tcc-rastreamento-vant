import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================

VIDEO = "video14"

FRAMES = [400, 401, 414, 415]

HORIZONTE = 15

CROP_SIZE = 150

IMG_DIR = Path(
    "/Users/ighorribeiro/PycharmProjects/TesteYolo26/datasets/anti_uav_tracking_test_final/images/test"
)

LBL_DIR = Path(
    "/Users/ighorribeiro/PycharmProjects/TesteYolo26/datasets/anti_uav_tracking_test_final/labels/test"
)

STATES_CSV = (
    "/Users/ighorribeiro/PycharmProjects/TesteYolo26/hitratepos/hibrido_damped/estados_kalman/video14_states.csv"
)

OUT_DIR = Path("figuras_drawio")
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# KALMAN
# ============================================================

states = pd.read_csv(STATES_CSV)

# ============================================================
# FUNÇÕES
# ============================================================

def yolo_to_bbox(label_path, img_w, img_h):

    with open(label_path) as f:
        line = f.readline().strip()

    cls, cx, cy, bw, bh = map(float, line.split())

    cx *= img_w
    cy *= img_h

    bw *= img_w
    bh *= img_h

    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)

    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)

    return x1, y1, x2, y2, cx, cy


def crop_center(img, cx, cy, size):

    h, w = img.shape[:2]

    x1 = int(cx - size // 2)
    y1 = int(cy - size // 2)

    x2 = x1 + size
    y2 = y1 + size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)

    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(w, x2)
    y2 = min(h, y2)

    crop = img[y1:y2, x1:x2]

    crop = cv2.copyMakeBorder(
        crop,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    return crop, x1, y1

# ============================================================
# PROCESSAMENTO
# ============================================================

nomes = [
    "t",
    "t1",
    "t14",
    "t15"
]

for idx, frame in enumerate(FRAMES):

    print(f"Processando frame {frame}")

    img_path = IMG_DIR / f"{VIDEO}_{frame:05d}.jpg"
    lbl_path = LBL_DIR / f"{VIDEO}_{frame:05d}.txt"

    img = cv2.imread(str(img_path))

    if img is None:
        print(f"Erro ao abrir {img_path}")
        continue

    H, W = img.shape[:2]

    x1, y1, x2, y2, gt_x, gt_y = yolo_to_bbox(
        lbl_path,
        W,
        H
    )

    crop, offset_x, offset_y = crop_center(
        img,
        gt_x,
        gt_y,
        CROP_SIZE
    )

    # ========================================================
    # TRACKING
    # ========================================================

    tracking = crop.copy()

    cor_bbox = (
        (0, 0, 255)
        if idx in [0, 3]
        else (0, 255, 0)
    )

    bx1 = x1 - offset_x
    by1 = y1 - offset_y

    bx2 = x2 - offset_x
    by2 = y2 - offset_y

    cv2.rectangle(
        tracking,
        (bx1, by1),
        (bx2, by2),
        cor_bbox,
        2
    )

    cv2.imwrite(
        str(OUT_DIR / f"tracking_{nomes[idx]}.png"),
        tracking
    )

    # ========================================================
    # PREDIÇÃO H15
    # ========================================================

    prediction = crop.copy()

    frame_pred = frame - HORIZONTE

    row = states[
        states["frame"] == frame_pred
    ]

    if (
        len(row)
        and pd.notna(row.iloc[0]["x"])
        and pd.notna(row.iloc[0]["y"])
        and pd.notna(row.iloc[0]["vx"])
        and pd.notna(row.iloc[0]["vy"])
    ):

        x0 = float(row.iloc[0]["x"])
        y0 = float(row.iloc[0]["y"])

        vx = float(row.iloc[0]["vx"])
        vy = float(row.iloc[0]["vy"])

        pred_x = x0 + vx * HORIZONTE
        pred_y = y0 + vy * HORIZONTE

    else:

        pred_x = gt_x
        pred_y = gt_y

    gx = int(gt_x - offset_x)
    gy = int(gt_y - offset_y)

    px = int(pred_x - offset_x)
    py = int(pred_y - offset_y)

    erro = np.sqrt(
        (gt_x - pred_x) ** 2 +
        (gt_y - pred_y) ** 2
    )

    # linha do erro (vermelha)

    cv2.line(
        prediction,
        (gx, gy),
        (px, py),
        (0, 0, 255),
        2
    )

    # centro previsto (azul)

    cv2.circle(
        prediction,
        (px, py),
        6,
        (255, 0, 0),
        -1
    )

    # centro real (verde)

    cv2.circle(
        prediction,
        (gx, gy),
        6,
        (0, 255, 0),
        -1
    )

    cv2.putText(
        prediction,
        f"{erro:.1f}px",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    cv2.imwrite(
        str(OUT_DIR / f"prediction_{nomes[idx]}.png"),
        prediction
    )

print("\nImagens geradas em:")
print(OUT_DIR.resolve())