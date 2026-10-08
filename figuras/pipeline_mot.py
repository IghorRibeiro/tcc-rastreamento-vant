#!/usr/bin/env python3
"""Figura pipeline_mot.pdf -- fluxo do HIT-Tracker no mesmo estilo da pipeline_sot.pdf.

A pipeline_sot.pdf foi montada no draw.io (o XML do diagrama esta embutido no proprio
PDF, campo Subject) com recortes gerados por TesteYolo26/figuras_drawio.py. Este script
segue a mesma linguagem visual:
  * caixas arredondadas (raio = 15% do menor lado, como o draw.io), contorno preto fino;
  * cores do diagrama original: #FFF2CC (amarelo), #CFE8FF (azul), #D9F2D9 (verde),
    #F4CCCC (vermelho); a caixa da associacao usa o lilas padrao do draw.io (#E1D5E7);
  * Times New Roman, 16 (titulo das caixas externas) e 13 (demais textos), em unidades
    do draw.io; a pagina tem ~760 unidades de largura, como a da pipeline_sot, entao as
    duas figuras ficam com o mesmo corpo de letra quando inseridas com 16 cm;
  * sobreposicoes desenhadas com OpenCV como em figuras_drawio.py: caixa vermelha =
    deteccao, verde = rastreador, ponto azul = posicao predita, ponto verde = posicao
    real, linha e texto vermelhos = erro.

Duas etapas:
  1) extrair (roda uma vez, ~1 min): le os quadros do Track 3, roda o YOLO26n ajustado por
     hard-example mining com SAHI 512 SO NO QUADRO ESCOLHIDO (para desenhar as caixas;
     nenhum resultado do TCC e recalculado), le as trilhas salvas do HIT-Tracker (28/09),
     faz a correspondencia CLEAR MOT com o GT e calcula a predicao com o Kalman do SIGE
     (mesmo codigo do TCC). Grava cache/pipeline_mot.json e os quadros em PNG.
  2) desenhar (segundos): monta a figura a partir do cache. Para ajustar so o desenho,
     rode de novo; o cache e reaproveitado (use --refazer-cache para extrair de novo).

Uso (com o .venv do DroneTrackingMOT):
  python figuras/pipeline_mot.py --raiz-mot ~/PycharmProjects/DroneTrackingMOT \
      --saida /caminho/do/TCC/Figuras/pipeline_mot.pdf
Opcoes: --sem-detector  usa as caixas das trilhas salvas no lugar das deteccoes (nao
        carrega o modelo); --quadro N, --trilha ID, --seq UAVxxx sobrescrevem o CONFIG.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

# =============================================================================
# CONFIG -- tudo o que se ajusta na figura esta aqui
# =============================================================================
CONFIG = {
    # dados
    "seq": "UAV007",                 # validacao do Track 3: ~40 alvos/quadro, alvos de ~21x15 px
    "quadro": 300,                   # quadro t
    "trilha": None,                  # None = escolhe a trilha com o erro H30 mediano entre as elegiveis
    "horizontes": (5, 10, 15, 30),
    "gamma_kalman": 1.0,             # 1,0 = persistente; 0,95 = amortecido (Kalman externo do SIGE)
    "passo_trilhas": 15,             # segundo recorte das trilhas: quadro t + passo
    "sahi": {"slice": 512, "overlap": 0.20, "conf": 0.25},
    "gm": {"gamma": 0.55, "sigma": 0.15},   # custo de Geman-McClure do HIT-Tracker
    "k_matriz": 4,                   # trilhas x deteccoes na matriz de custo ilustrativa
    # recortes
    "recorte_trilhas_px": 150,       # regiao (px do quadro) dos recortes das trilhas
    "recorte_predicao_px": 96,       # regiao minima (px do quadro) dos recortes da predicao
    "canvas_px": 450,                # lado do recorte depois de ampliado (3 x 150 do SOT)
    "escala_sobreposicao": 3.0,      # espessuras/raios de figuras_drawio.py (feitos para 150 px) x 3
    "espessura_det_px": 2,           # caixas de deteccao no quadro inteiro (px do quadro original)
    # pagina (unidades do draw.io) e texto
    "largura_cm": 16.0,
    "fonte": ["Times New Roman", "Liberation Serif", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
    "tam_titulo": 16,
    "tam_texto": 13,
    "tam_pequeno": 11,
    "linha": 1.0,                    # contorno das caixas (unidades)
    "cores": {"amarelo": "#FFF2CC", "azul": "#CFE8FF", "verde": "#D9F2D9", "vermelho": "#F4CCCC",
              "lilas": "#E1D5E7", "contorno": "#000000",
              "recorte1": "#E69F00", "recorte2": "#CC79A7", "regiao": "#000000",
              "atribuicao": "#00A000"},
    "mostrar_regiao_zoom": True,     # retangulo tracejado no quadro inteiro marcando a regiao das trilhas
    "mostrar_legenda": True,         # legenda dos pontos na caixa de predicao
    "legenda_x": (230, 400),         # posicao (unidades) dos dois marcadores da legenda
    # textos (pt-BR)
    "textos": {
        "titulo_topo": "HIT-Tracker: Detecção e Associação de Múltiplos Alvos",
        "det": ["YOLO26n", "Hard-Example Mining", "Recortes de 512 px"],
        "assoc": ["Associação Húngara", "Custo de Geman-McClure", "e NSA-Kalman"],
        "trilhas": ["BoT-SORT", "Trilhas com Identidade"],
        "quadro": "Quadro {n}",
        "titulo_base": "Predição de Trajetória com Filtro de Kalman Externo (H5 a H30)",
        "horizonte": "H{h} (quadro {n})",
        "legenda_predito": "posição predita pelo Kalman",
        "legenda_real": "posição real (ground truth)",
        "linhas_matriz": "ID {id}",
        "colunas_matriz": "D{j}",
    },
}

# Sobreposicoes (BGR, como em figuras_drawio.py)
COR_DET = (0, 0, 255)
COR_TRILHA = (0, 255, 0)
COR_PREDITO = (255, 0, 0)
COR_REAL = (0, 255, 0)
COR_ERRO = (0, 0, 255)


# =============================================================================
# 1) Extracao (dados reais, uma vez)
# =============================================================================

def _iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def extrair(raiz: Path, cache_dir: Path, cfg_fig: dict, usar_detector: bool) -> dict:
    import cv2
    from scipy.optimize import linear_sum_assignment

    sys.path.insert(0, str(raiz / "mot_prediction"))
    from motpred import common as mc  # noqa: E402
    from motpred import data, mot_eval, prediction  # noqa: E402

    cfg = mc.load_config(raiz / "mot_prediction" / "config.yaml")
    uc = mc.uav_common(cfg)
    seq, t = cfg_fig["seq"], int(cfg_fig["quadro"])
    hs = [int(h) for h in cfg_fig["horizontes"]]
    gamma = float(cfg_fig["gamma_kalman"])
    passo = int(cfg_fig["passo_trilhas"])

    gt_df, gt_index = data.load_gt(cfg, seq)
    traj = raiz / "mot_prediction" / "results" / "exp_b" / "traj" / "hit_ref" / f"{seq}.txt"
    if not traj.exists():
        raise SystemExit(f"Trajetoria do HIT-Tracker nao encontrada: {traj}")
    pred_df = data.load_mot(cfg, traj)
    print(f"[extrair] {seq}: correspondencia CLEAR MOT (py-motmetrics)...", flush=True)
    _, corr = mot_eval.evaluate_mot_sequence(gt_df, pred_df, seq)
    clear = corr["clear"]

    # -- quadros necessarios --------------------------------------------------------
    need = {t - 1, t, t + passo} | {t + h for h in hs}
    frames = {}
    for idx, img in data.iter_video_frames(data.video_path(cfg, seq), max_frames=max(need)):
        if idx in need:
            frames[idx] = img
    missing = sorted(need - set(frames))
    if missing:
        raise SystemExit(f"Quadros fora do video: {missing}")
    H, W = frames[t].shape[:2]
    cache_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in frames.items():
        cv2.imwrite(str(cache_dir / f"{seq}_{idx:05d}.png"), img)

    # -- trilhas salvas ---------------------------------------------------------------
    def boxes_at(frame):
        sel = pred_df[pred_df["frame"] == frame]
        return [[int(r.id), float(r.x), float(r.y), float(r.w), float(r.h)] for r in sel.itertuples()]

    # -- deteccoes no quadro t (YOLO26n HEM + SAHI 512, como no HIT-Tracker) ---------
    s = cfg_fig["sahi"]
    slices = []
    try:
        from sahi.slicing import get_slice_bboxes

        slices = [list(map(int, b)) for b in get_slice_bboxes(
            image_height=H, image_width=W, slice_height=int(s["slice"]), slice_width=int(s["slice"]),
            auto_slice_resolution=False, overlap_height_ratio=float(s["overlap"]),
            overlap_width_ratio=float(s["overlap"]))]
    except Exception as exc:  # noqa: BLE001
        print(f"[extrair] aviso: nao calculei os recortes do SAHI ({exc!r})")
    if cfg_fig.get("deteccoes_json"):
        pre = json.loads(Path(cfg_fig["deteccoes_json"]).read_text(encoding="utf-8"))
        dets = [[float(v) for v in d[:5]] for d in pre["deteccoes"]]
        slices = pre.get("recortes_sahi") or slices
        fonte_det = f"YOLO26n HEM + SAHI 512 (deteccoes pre-calculadas: {Path(cfg_fig['deteccoes_json']).name})"
    elif usar_detector:
        from sahi import AutoDetectionModel

        dev = mc.resolve_device("auto")
        dev = f"cuda:{dev}" if dev.isdigit() else dev
        weights = mc.weights_path(cfg, "hardmining")
        print(f"[extrair] YOLO26n HEM + SAHI {s['slice']} no quadro {t} ({weights.name}, {dev})", flush=True)
        model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
                                                   confidence_threshold=float(s["conf"]), device=dev)
        runner = mc.lafusion_sahi_runner(cfg)
        dets = runner.build_sahi_detections(frame=frames[t], det_model=model, slice_h=int(s["slice"]),
                                            slice_w=int(s["slice"]), overlap=float(s["overlap"]),
                                            conf_thresh=float(s["conf"]))
        dets = [[float(v) for v in d[:5]] for d in dets]
        fonte_det = "YOLO26n HEM + SAHI 512 (rodado so neste quadro)"
    else:
        dets = [[x, y, x + w, y + h, 1.0] for _, x, y, w, h in boxes_at(t)]
        fonte_det = "caixas das trilhas salvas do HIT-Tracker (--sem-detector)"

    # -- trilha escolhida e predicao (Kalman do SIGE) -----------------------------------
    centers_all = prediction.track_centers(pred_df)
    cand = []
    for hid, centers in centers_all.items():
        if t not in centers:
            continue
        g = clear.get((t, hid))
        if g is None:
            continue
        fut = gt_index.by_id.get(g, {})
        if any((t + h) not in fut for h in hs):
            continue
        hist = {f: c for f, c in centers.items() if f <= t}
        if len(hist) < 20:
            continue
        cx, cy = centers[t]
        if not (60 <= cx <= W - 60 and 60 <= cy <= H - 60):
            continue
        state = None
        for frame, st in prediction.kalman_states(uc, hist, gamma):
            if frame == t:
                state = st
        if state is None:
            continue
        preds, erros = {}, {}
        for h in hs:
            p = uc.forward(state, h, prediction.TRANSITION, gamma)
            gx, gy, gd = fut[t + h]
            preds[h] = (float(p[0, 0]), float(p[1, 0]))
            erros[h] = {"px": float(math.hypot(p[0, 0] - gx, p[1, 0] - gy)),
                        "norm": float(math.hypot(p[0, 0] - gx, p[1, 0] - gy) / gd) if gd > 0 else None,
                        "real": (float(gx), float(gy)), "caixa_real": list(gt_index.boxes_by_id[g][t + h])}
        cand.append({"hid": int(hid), "g": int(g), "preds": preds, "erros": erros,
                     "historico": [list(hist[f]) for f in sorted(hist)][-30:], "centro_t": [cx, cy]})
    if not cand:
        raise SystemExit("Nenhuma trilha elegivel nesse quadro; mude CONFIG['quadro'] ou a sequencia.")
    if cfg_fig.get("trilha") is not None:
        escolha = next((c for c in cand if c["hid"] == int(cfg_fig["trilha"])), None)
        if escolha is None:
            raise SystemExit(f"Trilha {cfg_fig['trilha']} nao e elegivel no quadro {t}. Elegiveis: "
                             f"{sorted(c['hid'] for c in cand)}")
    else:
        cand.sort(key=lambda c: c["erros"][hs[-1]]["px"])
        escolha = cand[len(cand) // 2]
    print(f"[extrair] trilha {escolha['hid']} (GT {escolha['g']}); {len(cand)} elegiveis; erros: "
          + ", ".join(f"H{h}={escolha['erros'][h]['px']:.1f}px" for h in hs), flush=True)

    # -- matriz de custo ilustrativa (trilhas em t-1 x deteccoes em t) -----------------
    # Mesma formula de robust_flit_association.make_flit_get_dists + fuse_score do
    # Ultralytics; a posicao da trilha e a caixa de t-1 (no rastreador e a predicao do
    # Kalman interno), entao a matriz e ilustrativa.
    k = int(cfg_fig["k_matriz"])
    prev = boxes_at(t - 1)
    ref = escolha["centro_t"]
    prev.sort(key=lambda b: math.hypot(b[1] + b[3] / 2 - ref[0], b[2] + b[4] / 2 - ref[1]))
    rows = sorted(prev[:k], key=lambda b: b[0])
    dsel = sorted(dets, key=lambda d: math.hypot((d[0] + d[2]) / 2 - ref[0], (d[1] + d[3]) / 2 - ref[1]))[:k]
    dsel.sort(key=lambda d: (d[0] + d[2]) / 2)
    diag = math.hypot(W, H)
    gm = cfg_fig["gm"]
    cost = np.zeros((len(rows), len(dsel)))
    for i, (_, x, y, w, h) in enumerate(rows):
        tb = [x, y, x + w, y + h]
        tc = (x + w / 2, y + h / 2)
        for j, d in enumerate(dsel):
            d_iou = 1.0 - _iou_xyxy(tb, d[:4])
            ell = min(1.0, math.hypot(tc[0] - (d[0] + d[2]) / 2, tc[1] - (d[1] + d[3]) / 2) / diag)
            rho = ell ** 2 / (ell ** 2 + gm["sigma"] ** 2)
            c = (1 - gm["gamma"]) * d_iou + gm["gamma"] * rho
            cost[i, j] = 1.0 - (1.0 - c) * d[4]          # fuse_score
    ri, ci = linear_sum_assignment(cost) if cost.size else ([], [])

    out = {
        "seq": seq, "t": t, "W": W, "H": H, "passo": passo, "horizontes": hs, "gamma_kalman": gamma,
        "deteccoes": dets, "fonte_deteccoes": fonte_det, "recortes_sahi": slices,
        "trilhas_t": boxes_at(t), "trilhas_t_passo": boxes_at(t + passo),
        "escolha": escolha, "elegiveis": len(cand),
        "matriz": {"ids": [b[0] for b in rows], "custo": cost.tolist(),
                   "atribuicao": [[int(a), int(b)] for a, b in zip(ri, ci)]},
    }
    out["escolha"]["preds"] = {str(h): v for h, v in escolha["preds"].items()}
    out["escolha"]["erros"] = {str(h): v for h, v in escolha["erros"].items()}
    (cache_dir / "pipeline_mot.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


# =============================================================================
# 2) Recortes com sobreposicoes (OpenCV, como figuras_drawio.py)
# =============================================================================

def _ler(cache_dir, seq, n):
    import cv2

    img = cv2.imread(str(cache_dir / f"{seq}_{n:05d}.png"))
    if img is None:
        raise SystemExit(f"Quadro {n} nao esta no cache; rode com --refazer-cache")
    return img


def recorte(img, cx, cy, lado):
    """Recorte quadrado centrado (com borda preta fora do quadro), como crop_center()."""
    import cv2

    h, w = img.shape[:2]
    x1, y1 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
    x2, y2 = x1 + lado, y1 + lado
    pl, pt, pr, pb = max(0, -x1), max(0, -y1), max(0, x2 - w), max(0, y2 - h)
    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
    return crop, x1, y1


def painel_trilhas(img, caixas, cx, cy, cfg_fig):
    import cv2

    lado = int(cfg_fig["recorte_trilhas_px"])
    S = int(cfg_fig["canvas_px"])
    e = float(cfg_fig["escala_sobreposicao"])
    crop, ox, oy = recorte(img, cx, cy, lado)
    k = S / lado
    can = cv2.resize(crop, (S, S), interpolation=cv2.INTER_CUBIC)
    for tid, x, y, w, h in caixas:
        p1 = (int(round((x - ox) * k)), int(round((y - oy) * k)))
        p2 = (int(round((x + w - ox) * k)), int(round((y + h - oy) * k)))
        if p2[0] < 0 or p2[1] < 0 or p1[0] > S or p1[1] > S:
            continue
        cv2.rectangle(can, p1, p2, COR_TRILHA, max(1, int(round(2 * e))))
        cv2.putText(can, str(tid), (p1[0], max(int(14 * e), p1[1] - int(4 * e))), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * e, COR_TRILHA, max(1, int(round(1.3 * e))), cv2.LINE_AA)
    return can


def painel_predicao(img, pred, real, cfg_fig):
    import cv2

    S = int(cfg_fig["canvas_px"])
    e = float(cfg_fig["escala_sobreposicao"])
    mx, my = (pred[0] + real[0]) / 2, (pred[1] + real[1]) / 2
    lado = int(max(cfg_fig["recorte_predicao_px"], 1.3 * max(abs(pred[0] - real[0]), abs(pred[1] - real[1])) + 24))
    crop, ox, oy = recorte(img, mx, my, lado)
    k = S / lado
    can = cv2.resize(crop, (S, S), interpolation=cv2.INTER_CUBIC)
    P = (int(round((pred[0] - ox) * k)), int(round((pred[1] - oy) * k)))
    R = (int(round((real[0] - ox) * k)), int(round((real[1] - oy) * k)))
    cv2.line(can, R, P, COR_ERRO, max(1, int(round(2 * e))), cv2.LINE_AA)
    cv2.circle(can, P, int(round(6 * e)), COR_PREDITO, -1, cv2.LINE_AA)
    cv2.circle(can, R, int(round(6 * e)), COR_REAL, -1, cv2.LINE_AA)
    return can


def painel_deteccao(img, dets, cfg_fig):
    import cv2

    fator = 2
    h, w = img.shape[:2]
    can = cv2.resize(img, (w * fator, h * fator), interpolation=cv2.INTER_CUBIC)
    esp = max(1, int(round(cfg_fig["espessura_det_px"] * fator)))
    for x1, y1, x2, y2, _s in dets:
        cv2.rectangle(can, (int(round(x1 * fator)), int(round(y1 * fator))),
                      (int(round(x2 * fator)), int(round(y2 * fator))), COR_DET, esp)
    return can


# =============================================================================
# 3) Desenho (matplotlib, unidades do draw.io, eixo y para baixo)
# =============================================================================

def desenhar(dados: dict, cache_dir: Path, saida: Path, cfg_fig: dict) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    c = cfg_fig["cores"]
    tx = cfg_fig["textos"]
    PW, PH = 760.0, 530.0
    pt_por_unid = (cfg_fig["largura_cm"] / 2.54 * 72.0) / PW
    plt.rcParams.update({"font.family": "serif", "font.serif": cfg_fig["fonte"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    def fs(u):
        return u * pt_por_unid

    lw = cfg_fig["linha"] * pt_por_unid
    fig = plt.figure(figsize=(cfg_fig["largura_cm"] / 2.54, cfg_fig["largura_cm"] / 2.54 * PH / PW))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PW)
    ax.set_ylim(PH, 0)
    ax.set_autoscale_on(False)        # imshow nao pode mexer nos limites da pagina
    ax.set_aspect("equal")
    ax.axis("off")

    def caixa(x, y, w, h, cor, z=1):
        r = 0.15 * min(w, h)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=cor,
                                    ec=c["contorno"], lw=lw, zorder=z))

    def texto(x, y, s, tam, **kw):
        ax.text(x, y, s, fontsize=fs(tam), ha=kw.pop("ha", "center"), va=kw.pop("va", "center"), zorder=10, **kw)

    def linhas(x, y, ls, tam=None):
        tam = tam or cfg_fig["tam_texto"]
        for i, s in enumerate(ls):
            texto(x, y + i * tam * 1.2, s, tam)

    def imagem(img_bgr, x, y, w, h):
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), extent=(x, x + w, y + h, y), interpolation="none",
                  zorder=5)

    def seta(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color="black", mutation_scale=fs(10),
                                    shrinkA=0, shrinkB=0))

    seq, t = dados["seq"], dados["t"]
    W, H = dados["W"], dados["H"]
    esc = dados["escolha"]
    hs = dados["horizontes"]
    T13 = cfg_fig["tam_texto"]

    # ---------------- caixa de cima ----------------
    caixa(30, 10, 700, 270, c["amarelo"])
    texto(380, 24, tx["titulo_topo"], cfg_fig["tam_titulo"])
    by, bh = 45, 219
    # B1 deteccao
    caixa(45, by, 200, bh, c["azul"], z=2)
    linhas(145, by + 13, tx["det"])
    iw = 170.0
    ih = iw * H / W
    ix = 145 - iw / 2
    iy = by + 13 + cfg_fig["tam_texto"] * len(tx["det"]) * 1.2 + 2    # imagem logo abaixo das linhas do titulo
    imagem(painel_deteccao(_ler(cache_dir, seq, t), dados["deteccoes"], cfg_fig), ix, iy, iw, ih)
    k = iw / W
    for j, b in enumerate(dados.get("recortes_sahi", [])[:2]):
        ins = 1.5 + 1.8 * j
        ax.add_patch(Rectangle((ix + b[0] * k + ins, iy + b[1] * k + ins), (b[2] - b[0]) * k - 2 * ins,
                               (b[3] - b[1]) * k - 2 * ins, fill=False, ec=c["recorte1" if j == 0 else "recorte2"],
                               lw=lw * 1.4, ls=(0, (4, 2)), zorder=7))
    lado_tr = cfg_fig["recorte_trilhas_px"]
    cx, cy = esc["centro_t"]
    if cfg_fig["mostrar_regiao_zoom"]:
        ax.add_patch(Rectangle((ix + (cx - lado_tr / 2) * k, iy + (cy - lado_tr / 2) * k), lado_tr * k, lado_tr * k,
                               fill=False, ec=c["regiao"], lw=lw, ls=(0, (2, 1.5)), zorder=7))
    texto(145, iy + ih + 14, tx["quadro"].format(n=t), T13)
    seta(245, by + bh / 2, 270, by + bh / 2)

    # B2 associacao
    caixa(270, by, 160, bh, c["lilas"], z=2)
    linhas(350, by + 13, tx["assoc"])
    mat = dados["matriz"]
    custo = np.asarray(mat["custo"], dtype=float)
    if custo.size:
        n_r, n_c = custo.shape
        cel = 21.0
        gx0 = 350 - n_c * cel / 2 + 14
        gy0 = by + 78
        cmin, cmax = float(custo.min()), float(custo.max())
        atrib = {tuple(a) for a in mat["atribuicao"]}
        for i in range(n_r):
            texto(gx0 - 4, gy0 + (i + 0.5) * cel, tx["linhas_matriz"].format(id=mat["ids"][i]),
                  cfg_fig["tam_pequeno"] - 1, ha="right")
            for j in range(n_c):
                v = (custo[i, j] - cmin) / (cmax - cmin) if cmax > cmin else 0.5
                cinza = 0.95 - 0.55 * v
                ok = (i, j) in atrib
                ax.add_patch(Rectangle((gx0 + j * cel, gy0 + i * cel), cel, cel, fc=(cinza, cinza, cinza),
                                       ec=c["atribuicao"] if ok else "white", lw=lw * (2.2 if ok else 0.6),
                                       zorder=8 if ok else 7))
        for j in range(n_c):
            texto(gx0 + (j + 0.5) * cel, gy0 - 8, tx["colunas_matriz"].format(j=j + 1), cfg_fig["tam_pequeno"] - 1)
        fy = gy0 + n_r * cel + 18
        texto(350, fy, r"$c=(1-\gamma)\,d_{\mathrm{IoU}}+\gamma\,\rho(\ell)$", cfg_fig["tam_pequeno"])
        texto(350, fy + 16, r"$\rho(\ell)=\ell^{2}/(\ell^{2}+\sigma^{2})$", cfg_fig["tam_pequeno"])
    seta(430, by + bh / 2, 455, by + bh / 2)

    # B3 trilhas com identidade
    caixa(455, by, 260, bh, c["verde"], z=2)
    linhas(585, by + 13, tx["trilhas"])
    tw = 115.0
    for j, (n, caixas_n) in enumerate(((t, dados["trilhas_t"]), (t + dados["passo"], dados["trilhas_t_passo"]))):
        x0 = 465 + j * (tw + 10)
        iy3 = by + 13 + cfg_fig["tam_texto"] * len(tx["trilhas"]) * 1.2 + 2
        imagem(painel_trilhas(_ler(cache_dir, seq, n), caixas_n, cx, cy, cfg_fig), x0, iy3, tw, tw)
        texto(x0 + tw / 2, iy3 + tw + 14, tx["quadro"].format(n=n), T13)

    # ---------------- caixa de baixo ----------------
    seta(380, 280, 380, 318)
    yb = 318
    caixa(30, yb, 700, 200, c["vermelho"])
    texto(380, yb + 14, tx["titulo_base"], cfg_fig["tam_titulo"])
    if cfg_fig["mostrar_legenda"]:
        ly = yb + 32
        for xm, cor, s in ((cfg_fig["legenda_x"][0], "#0000FF", tx["legenda_predito"]),
                           (cfg_fig["legenda_x"][1], "#00FF00", tx["legenda_real"])):
            ax.scatter([xm], [ly], s=fs(6) ** 2, c=[cor], edgecolors="none", zorder=11)
            texto(xm + 7, ly, s, cfg_fig["tam_pequeno"], ha="left")
    yi = yb + 46
    for i, h in enumerate(hs):
        xc = 30 + 700 * (2 * i + 1) / (2 * len(hs))
        x0 = xc - tw / 2
        e = esc["erros"][str(h)]
        img = _ler(cache_dir, seq, t + h)
        can = painel_predicao(img, esc["preds"][str(h)], e["real"], cfg_fig)
        rotulo = f"{e['px']:.1f} px".replace(".", ",")
        cv2.putText(can, rotulo, (int(15 * cfg_fig["escala_sobreposicao"]), int(35 * cfg_fig["escala_sobreposicao"])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * cfg_fig["escala_sobreposicao"], COR_ERRO,
                    max(1, int(round(2 * cfg_fig["escala_sobreposicao"]))), cv2.LINE_AA)
        imagem(can, x0, yi, tw, tw)
        texto(xc, yi + tw + 14, tx["horizonte"].format(h=h, n=t + h), T13)

    ax.set_xlim(0, PW)
    ax.set_ylim(PH, 0)
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, format="pdf")
    fig.savefig(saida.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"[desenhar] {saida}  (+ .png de conferencia)")
    print(f"[desenhar] {seq}, quadro {t}, trilha {esc['hid']}; deteccoes: {dados['fonte_deteccoes']}; "
          f"erros (px): " + ", ".join(f"H{h}={esc['erros'][str(h)]['px']:.1f}" for h in hs))


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera Figuras/pipeline_mot.pdf (fluxo do HIT-Tracker).")
    ap.add_argument("--raiz-mot", default=os.path.expanduser("~/PycharmProjects/DroneTrackingMOT"),
                    help="pasta do projeto do Track 3 (com mot_prediction/, dataset/, weights/ e results de 28/09)")
    ap.add_argument("--saida", default="Figuras/pipeline_mot.pdf")
    ap.add_argument("--cache", default=None, help="pasta do cache (padrao: figuras/cache_pipeline_mot, ao lado deste script)")
    ap.add_argument("--refazer-cache", action="store_true")
    ap.add_argument("--sem-detector", action="store_true")
    ap.add_argument("--deteccoes", help="JSON com deteccoes ja calculadas no quadro t ({deteccoes: [[x1,y1,x2,y2,score]], "
                                        "recortes_sahi: [...]}); dispensa carregar o modelo")
    ap.add_argument("--seq")
    ap.add_argument("--quadro", type=int)
    ap.add_argument("--trilha", type=int)
    a = ap.parse_args()
    cfg_fig = json.loads(json.dumps(CONFIG))
    for key in ("seq", "quadro", "trilha"):
        if getattr(a, key) is not None:
            cfg_fig[key] = getattr(a, key)
    if a.deteccoes:
        cfg_fig["deteccoes_json"] = str(Path(os.path.expanduser(a.deteccoes)).resolve())
    saida = Path(os.path.expanduser(a.saida)).resolve()
    cache_dir = (Path(os.path.expanduser(a.cache)).resolve() if a.cache
                 else Path(__file__).resolve().parent / "cache_pipeline_mot")
    cj = cache_dir / "pipeline_mot.json"
    dados = None
    if cj.exists() and not a.refazer_cache:
        dados = json.loads(cj.read_text(encoding="utf-8"))
        if (dados["seq"], dados["t"]) != (cfg_fig["seq"], int(cfg_fig["quadro"])) or (
                cfg_fig.get("trilha") is not None and dados["escolha"]["hid"] != int(cfg_fig["trilha"])):
            dados = None
    if dados is None:
        dados = extrair(Path(os.path.expanduser(a.raiz_mot)).resolve(), cache_dir, cfg_fig, not a.sem_detector)
    desenhar(dados, cache_dir, saida, cfg_fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
