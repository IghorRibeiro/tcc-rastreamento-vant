"""Parte 1a -- DUT Anti-UAV (SIGE): 9 metodos a partir das saidas salvas, sem rodar rastreador.

Duas contas, ambas sobre as MESMAS trajetorias salvas:
  (1) avaliador das tabelas publicadas: scripts/reviewer_validation/evaluate_tracking.py
      do TesteYolo26, IMPORTADO sem alteracao (SA, SR@0,5, IoU, AUC, cobertura e o erro
      de predicao como foi publicado). Serve para escolher a trajetoria que reproduz o
      CSV por video salvo e para conferir as Tabelas 2 e 3.
  (2) Kalman do SIGE publicado (publicar_github/.../src/uav_common.py: P0 = I e
      amortecimento a cada passo da previsao), o MESMO usado no MOT (Parte 1b).
      Gera erro em px e normalizado pela diagonal do GT em t+H.
GT por quadro: rotulos YOLO (labels/test) convertidos com a resolucao real de cada video,
exatamente como o avaliador das tabelas. Diagonal do DUT: GT oficial (videoNN_gt.txt),
sem os quadros -100.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from . import base, estat

NAN = float("nan")


def video_ids(images_dir, smoke: bool) -> list[str]:
    ids = sorted({p.name.split("_")[0] for p in images_dir.glob("video*_*.jpg")})
    if smoke:
        wanted = base.load_cfg()["smoke"]["dut_videos"]
        ids = [v for v in wanted if v in ids]
    return ids


def image_size(images_dir, video: str):
    import cv2

    sample = images_dir / f"{video}_00001.jpg"
    if not sample.exists():
        sample = next(iter(images_dir.glob(f"{video}_*.jpg")))
    image = cv2.imread(str(sample))
    if image is None:
        raise RuntimeError(f"Imagem invalida: {sample}")
    h, w = image.shape[:2]
    return w, h


# ---------------------------------------------------------------------------
# Diagonal do alvo no DUT pelo GT oficial
# ---------------------------------------------------------------------------

def diagonal_oficial(gt_dir):
    rows = []
    for path in sorted(gt_dir.glob("video*_gt.txt")):
        video = path.name.split("_")[0]
        n_lines = n_abs = 0
        ws, hs = [], []
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = raw.replace(",", " ").split()
            if len(parts) < 4:
                continue
            n_lines += 1
            x, y, w, h = map(float, parts[:4])
            if x == -100 and y == -100 and w == -100 and h == -100:
                n_abs += 1
                continue
            ws.append(w)
            hs.append(h)
        ws = np.asarray(ws, dtype=float)
        hs = np.asarray(hs, dtype=float)
        dg = np.hypot(ws, hs)
        rows.append({"video": video, "quadros": n_lines, "ausentes_-100": n_abs, "quadros_com_alvo": int(ws.size),
                     "largura_media_px": float(ws.mean()) if ws.size else NAN,
                     "altura_media_px": float(hs.mean()) if hs.size else NAN,
                     "diagonal_media_px": float(dg.mean()) if dg.size else NAN,
                     "diagonal_mediana_px": float(np.median(dg)) if dg.size else NAN,
                     "_w": ws, "_h": hs})
    if not rows:
        return pd.DataFrame(), {}
    all_w = np.concatenate([r["_w"] for r in rows])
    all_h = np.concatenate([r["_h"] for r in rows])
    all_d = np.hypot(all_w, all_h)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    resumo = {
        "fonte": str(gt_dir),
        "videos": int(len(df)),
        "linhas_total": int(df["quadros"].sum()),
        "linhas_-100": int(df["ausentes_-100"].sum()),
        "quadros_com_alvo": int(all_w.size),
        "por_quadro": {"largura_media_px": float(all_w.mean()), "altura_media_px": float(all_h.mean()),
                       "diagonal_media_px": float(all_d.mean()), "diagonal_mediana_px": float(np.median(all_d)),
                       "diagonal_dp_px": float(all_d.std(ddof=1))},
        "media_das_medias_por_video": {"largura_media_px": float(df["largura_media_px"].mean()),
                                       "altura_media_px": float(df["altura_media_px"].mean()),
                                       "diagonal_media_px": float(df["diagonal_media_px"].mean())},
    }
    return df, resumo


# ---------------------------------------------------------------------------
# Kalman do SIGE publicado (uav_common), pares por (quadro, H, gamma)
# ---------------------------------------------------------------------------

def pares_sige(uc, prediction, centers: dict, gt: dict, dampings, horizons):
    rec = []
    for damping in dampings:
        for frame, state in prediction.kalman_states(uc, centers, damping):
            for horizon in horizons:
                box = gt.get(frame + horizon)
                if box is None:
                    continue
                pred = uc.forward(state, horizon, prediction.TRANSITION, damping)
                cx = box[0] + box[2] / 2.0
                cy = box[1] + box[3] / 2.0
                err = float(np.hypot(pred[0, 0] - cx, pred[1, 0] - cy))
                diag = math.hypot(box[2], box[3])
                rec.append((int(frame), int(horizon), float(damping), err, err / diag if diag > 0 else NAN))
    return pd.DataFrame(rec, columns=["frame", "H", "damping", "e_px", "e_norm"])


def possiveis(gt: dict, horizons) -> dict:
    """Quantos quadros t tem alvo em t+H (denominador da cobertura dos pares)."""
    present = {f for f, b in gt.items() if b is not None}
    frames = sorted(gt)
    first = frames[0] if frames else 1
    return {h: sum(1 for f in present if f - h >= first) for h in horizons}


def compara_ref(rows: pd.DataFrame, ref: pd.DataFrame, cfg_sot: dict) -> dict:
    out = {"n_videos": int(len(rows)), "npred_iguais": False, "max_dif_H30_px": NAN,
           "max_dif_SR": NAN, "max_dif_SA": NAN, "reproduz": False}
    if ref is None or rows.empty:
        return out
    m = rows.merge(ref, on="video", suffixes=("", "_ref"))
    if len(m) != len(rows):
        return out
    out["npred_iguais"] = bool((m["n_predictions"].astype(int) == m["n_predictions_ref"].astype(int)).all())
    out["max_dif_H30_px"] = float(np.nanmax(np.abs(m["pred_error_H30_mean"] - m["pred_error_H30_mean_ref"])))
    out["max_dif_SR"] = float(np.nanmax(np.abs(m["SR@0.5"] - m["SR@0.5_ref"])))
    out["max_dif_SA"] = float(np.nanmax(np.abs(m["SA_legacy_IoU50"] - m["SA_legacy_IoU50_ref"])))
    out["reproduz"] = bool(out["npred_iguais"] and out["max_dif_H30_px"] < float(cfg_sot["tolerancia_h30_px"])
                           and out["max_dif_SR"] < float(cfg_sot["tolerancia_sr"]))
    return out


def main() -> int:
    smoke = base.smoke_flag()
    cfg = base.load_cfg()
    cs = cfg["sot"]
    out = base.sub(smoke, "parte1")
    t_start = time.perf_counter()

    from motpred import common as mc
    from motpred import prediction

    uc = mc.uav_common(base.mp_cfg())
    ev = base.load_module("sige_evaluate_tracking", base.P(cs["avaliador"]))
    images_dir = base.P(cs["images_dir"])
    labels_dir = base.P(cs["labels_dir"])
    horizons = [int(h) for h in cs["horizons"]]
    dampings = [float(d) for d in cs["dampings"]]
    if tuple(horizons) != tuple(int(h) for h in uc.HORIZONS) or tuple(horizons) != tuple(ev.HORIZONS):
        raise SystemExit(f"Horizontes divergentes: config {horizons}, uav_common {uc.HORIZONS}, avaliador {ev.HORIZONS}")

    # -- diagonal do DUT (GT oficial) -------------------------------------------------
    diag_df, diag_res = diagonal_oficial(base.P(cs["official_gt_dir"]))
    base.write_csv(diag_df, out / "dut_diagonal_por_video.csv")

    videos = video_ids(images_dir, smoke)
    base.log(f"Parte 1a: {len(videos)} videos do DUT")
    sizes = {v: image_size(images_dir, v) for v in videos}
    gts = {v: ev.load_sequence_gt(images_dir, labels_dir, v, *sizes[v]) for v in videos}

    # diagonal pelos rotulos YOLO (conferencia; mesma fonte do erro normalizado)
    yd = [math.hypot(b[2], b[3]) for v in videos for b in gts[v].values() if b is not None]
    diag_res["rotulos_yolo_conferencia"] = {"quadros_com_alvo": len(yd),
                                            "diagonal_media_px": float(np.mean(yd)) if yd else NAN}
    diag_res["videos_avaliados"] = videos
    diag_res["resolucoes"] = {v: f"{w}x{h}" for v, (w, h) in sizes.items()}
    base.write_json(out / "dut_diagonal_resumo.json", diag_res)

    origem_rows, aval_rows, pub_rows, pred_rows, fid_rows = [], [], [], [], []
    pairs_by_method: dict[str, pd.DataFrame] = {}
    for metodo, mcfg in cs["metodos"].items():
        ref = base.read_csv(base.P(mcfg["ref_por_video"]))
        if ref is not None:
            ref = ref[ref["video"].isin(videos)].copy()
            keep = ["video", "n_predictions", "coverage", "SA_legacy_IoU50", "SR@0.5", "mean_IoU_present",
                    "success_AUC"] + [c for c in ref.columns if c.startswith("pred_error_H")]
            pub = ref[[c for c in keep if c in ref.columns]].copy()
            pub.insert(0, "metodo", metodo)
            pub_rows.append(pub)
        chosen = None
        best = None
        for cand in mcfg["candidatos"]:
            cdir = base.P(cand)
            row = {"metodo": metodo, "candidato": str(cdir), "existe": cdir.is_dir()}
            if not cdir.is_dir():
                origem_rows.append(row)
                continue
            ev_rows = pd.DataFrame([ev.evaluate_video(images_dir, labels_dir, cdir / f"{v}.txt", v, *sizes[v],
                                                      damping=1.0) for v in videos])
            cmp = compara_ref(ev_rows, ref, cs)
            row.update(cmp)
            origem_rows.append(row)
            score = (0 if cmp["npred_iguais"] else 1,
                     cmp["max_dif_H30_px"] if np.isfinite(cmp["max_dif_H30_px"]) else math.inf)
            if best is None or score < best[0]:
                best = (score, cdir, ev_rows)
            if cmp["reproduz"]:
                chosen = (cdir, ev_rows, True)
                break
        if chosen is None and best is not None:
            chosen = (best[1], best[2], False)
        if chosen is None:
            base.log(f"AVISO {metodo}: nenhuma trajetoria encontrada")
            continue
        cdir, ev_rows, reproduz = chosen
        for r in origem_rows:
            if r["metodo"] == metodo:
                r["escolhido"] = (r["candidato"] == str(cdir))
        fonte = "original (reproduz o CSV publicado)" if reproduz else "reexecucao (original nao encontrada)"
        base.log(f"{metodo}: {cdir.name} -> {fonte}")

        # avaliador das tabelas: persistente (ja calculado) e amortecido (versao do avaliador)
        ev_rows = ev_rows.copy()
        ev_rows["damping"] = 1.0
        ev_damp = pd.DataFrame([ev.evaluate_video(images_dir, labels_dir, cdir / f"{v}.txt", v, *sizes[v],
                                                  damping=0.95) for v in videos])
        ev_damp["damping"] = 0.95
        both = pd.concat([ev_rows, ev_damp], ignore_index=True)
        both.insert(0, "trajetoria", str(cdir))
        both.insert(0, "trajetoria_original", reproduz)
        both.insert(0, "metodo", metodo)
        aval_rows.append(both)

        # Kalman do SIGE publicado: pares (px e normalizado)
        method_pairs = []
        for v in videos:
            preds = ev.load_predictions(cdir / f"{v}.txt")
            centers = {f: (p[0] + p[2] / 2.0, p[1] + p[3] / 2.0) for f, p in preds.items()}
            gt = gts[v]
            pr = pares_sige(uc, prediction, centers, gt, dampings, horizons)
            pr.insert(0, "video", v)
            method_pairs.append(pr)
            denom = possiveis(gt, horizons)
            groups = pr.groupby(["damping", "H"]) if len(pr) else []
            for (damping, horizon), grp in groups:
                e = grp["e_px"]
                en = grp["e_norm"].dropna()
                pred_rows.append({
                    "metodo": metodo, "video": v, "damping": float(damping), "H": int(horizon), "n": int(len(grp)),
                    "n_possiveis": int(denom[int(horizon)]),
                    "cobertura_pares": len(grp) / denom[int(horizon)] if denom[int(horizon)] else NAN,
                    "mean_px": float(e.mean()), "std_px": float(e.std(ddof=1)) if len(e) > 1 else NAN,
                    "median_px": float(e.median()), "mean_norm": float(en.mean()) if len(en) else NAN,
                    "median_norm": float(en.median()) if len(en) else NAN,
                    "trajetoria_original": reproduz})
            # fidelidade: reimplementacao == uav_common.prediction_errors (1o video de cada metodo)
            if v == videos[0]:
                gt_c = {f: (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0) for f, b in gt.items() if b is not None}
                fid = prediction.fidelity_check(uc, centers, gt_c, horizons, dampings)
                fid_rows.append({"metodo": metodo, "video": v, **fid})
        pairs_by_method[metodo] = pd.concat(method_pairs, ignore_index=True) if method_pairs else pd.DataFrame()

    origem = pd.DataFrame(origem_rows)
    base.write_csv(origem, out / "sot_origem_trajetorias.csv")
    aval = pd.concat(aval_rows, ignore_index=True) if aval_rows else pd.DataFrame()
    base.write_csv(aval, out / "sot_avaliador_por_video.csv")
    pub = pd.concat(pub_rows, ignore_index=True) if pub_rows else pd.DataFrame()
    base.write_csv(pub, out / "sot_publicado_por_video.csv")
    pred = pd.DataFrame(pred_rows)
    base.write_csv(pred, out / "sot_pred_por_video.csv")
    base.write_csv(pd.DataFrame(fid_rows), out / "sot_fidelidade_kalman.csv")

    # -- suporte comum: hibrido x cada referencia ---------------------------------------
    sup_rows = []
    hib = cs["hibrido"]
    if hib in pairs_by_method:
        ph = pairs_by_method[hib]
        for refm in cs["referencias"]:
            if refm not in pairs_by_method or ph.empty or pairs_by_method[refm].empty:
                continue
            m = ph.merge(pairs_by_method[refm], on=["video", "frame", "H", "damping"], suffixes=("_hib", "_ref"))
            for (v, damping, horizon), grp in m.groupby(["video", "damping", "H"]):
                sup_rows.append({"referencia": refm, "video": v, "damping": float(damping), "H": int(horizon),
                                 "n_comum": int(len(grp)),
                                 "mean_px_hibrido": float(grp["e_px_hib"].mean()),
                                 "mean_px_ref": float(grp["e_px_ref"].mean()),
                                 "mean_norm_hibrido": float(grp["e_norm_hib"].mean()),
                                 "mean_norm_ref": float(grp["e_norm_ref"].mean())})
    sup = pd.DataFrame(sup_rows)
    base.write_csv(sup, out / "sot_suporte_comum_por_video.csv")

    # -- resumo por metodo ---------------------------------------------------------------
    seed, nb = int(cfg["seed"]), int(cfg["bootstrap_n"])
    res_rows = []
    for metodo in cs["metodos"]:
        a = aval[(aval["metodo"] == metodo) & (aval["damping"] == 1.0)] if not aval.empty else pd.DataFrame()
        if a.empty:
            continue
        row = {"metodo": metodo, "rotulo": cs["metodos"][metodo]["rotulo"],
               "trajetoria_original": bool(a["trajetoria_original"].iloc[0]),
               "trajetoria": a["trajetoria"].iloc[0], "videos": int(len(a)),
               "SA": a["SA_legacy_IoU50"].mean(), "SR05": a["SR@0.5"].mean(), "IoU": a["mean_IoU_present"].mean(),
               "AUC": a["success_AUC"].mean(), "cobertura_media": a["coverage"].mean(),
               "cobertura_global": a["n_predictions"].sum() / a["n_frames"].sum()}
        for damping in dampings:
            tag = "pers" if damping == 1.0 else "amort"
            ad = aval[(aval["metodo"] == metodo) & (aval["damping"] == damping)]
            for h in horizons:
                row[f"aval_H{h}_px_{tag}"] = ad[f"pred_error_H{h}_mean"].mean()
                row[f"aval_H{h}_dp_{tag}"] = ad[f"pred_error_H{h}_mean"].std(ddof=1)
            for h in horizons:
                sel = pred[(pred["metodo"] == metodo) & (pred["damping"] == damping) & (pred["H"] == h)]
                row[f"H{h}_px_{tag}"] = sel["mean_px"].mean()
                row[f"H{h}_dp_{tag}"] = sel["mean_px"].std(ddof=1)
                row[f"H{h}_norm_{tag}"] = sel["mean_norm"].mean()
                row[f"H{h}_n_{tag}"] = int(sel["n"].sum())
                row[f"H{h}_cob_{tag}"] = sel["n"].sum() / sel["n_possiveis"].sum() if sel["n_possiveis"].sum() else NAN
                lo, hi, _ = estat.bca_mean(sel["mean_px"], seed=seed, n_resamples=nb)
                row[f"H{h}_px_{tag}_ic_inf"], row[f"H{h}_px_{tag}_ic_sup"] = lo, hi
                lo, hi, _ = estat.bca_mean(sel["mean_norm"], seed=seed, n_resamples=nb)
                row[f"H{h}_norm_{tag}_ic_inf"], row[f"H{h}_norm_{tag}_ic_sup"] = lo, hi
        pm = pub[pub["metodo"] == metodo] if not pub.empty else pd.DataFrame()
        if len(pm) and "pred_error_H30_mean" in pm.columns:
            row["salvo_H30_px"] = pm["pred_error_H30_mean"].mean()          # CSV por video da execucao publicada
            row["salvo_H30_dp"] = pm["pred_error_H30_mean"].std(ddof=1)
        else:
            row["salvo_H30_px"] = row["salvo_H30_dp"] = NAN
        p30 = row.get("H30_px_pers", NAN)
        d30 = row.get("H30_px_amort", NAN)
        row["reducao_H30_pct"] = 100.0 * (p30 - d30) / p30 if p30 and np.isfinite(p30) else NAN
        res_rows.append(row)
    resumo = pd.DataFrame(res_rows)
    base.write_csv(resumo, out / "sot_resumo_metodos.csv")

    # -- reproducao das tabelas publicadas -------------------------------------------------
    tab_dir = base.P(cs["tabelas"])
    t2 = base.read_csv(tab_dir / "table2_local_trackers.csv")
    t3 = base.read_csv(tab_dir / "table3_methods_comparison.csv")
    abl = base.read_csv(tab_dir / "kalman_damped_ablation.csv")
    rep = []

    def add(tabela, metodo, metrica, publicado, aval_v, sige_v):
        rep.append({"tabela": tabela, "metodo": metodo, "metrica": metrica, "publicado": publicado,
                    "recalc_avaliador": aval_v, "recalc_kalman_sige": sige_v,
                    "dif_avaliador": aval_v - publicado if np.isfinite(aval_v) and np.isfinite(publicado) else NAN,
                    "dif_kalman_sige": sige_v - publicado if np.isfinite(sige_v) and np.isfinite(publicado) else NAN})

    by = {r["metodo"]: r for r in res_rows}
    if t3 is not None:
        for metodo, mcfg in cs["metodos"].items():
            name = mcfg.get("tabela3")
            if not name or metodo not in by:
                continue
            sel = t3[t3["Metodo"] == name]
            if sel.empty:
                continue
            s, r = sel.iloc[0], by[metodo]
            add("Tabela 3", metodo, "SA", float(s["SA"]), r["SA"], NAN)
            add("Tabela 3", metodo, "SR@0,5", float(s["SR@0.5"]), r["SR05"], NAN)
            add("Tabela 3", metodo, "IoU médio", float(s["IoU_medio"]), r["IoU"], NAN)
            add("Tabela 3", metodo, "AUC", float(s["AUC"]), r["AUC"], NAN)
            add("Tabela 3", metodo, "H30 média (px)", float(s["H30_mean_px"]), r["aval_H30_px_pers"], r["H30_px_pers"])
            add("Tabela 3", metodo, "H30 dp (px)", float(s["H30_std_px"]), r["aval_H30_dp_pers"], r["H30_dp_pers"])
    if t2 is not None:
        for metodo, mcfg in cs["metodos"].items():
            name = mcfg.get("tabela2")
            if not name or metodo not in by:
                continue
            sel = t2[t2["Rastreador"] == name]
            if sel.empty:
                continue
            s, r = sel.iloc[0], by[metodo]
            add("Tabela 2", metodo, "SA", float(s["SA"]), r["SA"], NAN)
            add("Tabela 2", metodo, "SR@0,5", float(s["SR@0.5"]), r["SR05"], NAN)
            add("Tabela 2", metodo, "H30 média (px)", float(s["H30_mean_px"]), r["aval_H30_px_pers"], r["H30_px_pers"])
            add("Tabela 2", metodo, "H30 dp (px)", float(s["H30_std_px"]), r["aval_H30_dp_pers"], r["H30_dp_pers"])
    if abl is not None and hib in by:
        r = by[hib]
        for _, s in abl.iterrows():
            h = int(str(s["Horizonte"]).lstrip("H"))
            add("Ablação amortecimento", hib, f"H{h} persistente (px)", float(s["Persistente_mean_px"]),
                r.get(f"aval_H{h}_px_pers", NAN), r.get(f"H{h}_px_pers", NAN))
            add("Ablação amortecimento", hib, f"H{h} amortecido (px)", float(s["Damped_mean_px"]),
                r.get(f"aval_H{h}_px_amort", NAN), r.get(f"H{h}_px_amort", NAN))
    reprod = pd.DataFrame(rep)
    base.write_csv(reprod, out / "sot_reproducao_publicado.csv")

    # -- item 11: Tabela 2 (CSRT) e Tabela 3 (Hibrido) vem da mesma execucao? ------------
    mesma = {}
    if t2 is not None and t3 is not None:
        a2 = t2[t2["Rastreador"] == cs["metodos"][hib]["tabela2"]]
        a3 = t3[t3["Metodo"] == cs["metodos"][hib]["tabela3"]]
        if len(a2) and len(a3):
            a2, a3 = a2.iloc[0], a3.iloc[0]
            mesma = {
                "tabela2": {"SA": a2["SA"], "SR@0.5": a2["SR@0.5"], "H30_mean_px": a2["H30_mean_px"],
                            "H30_std_px": a2["H30_std_px"]},
                "tabela3": {"SA": a3["SA"], "SR@0.5": a3["SR@0.5"], "H30_mean_px": a3["H30_mean_px"],
                            "H30_std_px": a3["H30_std_px"]},
                "H30_igual_ao_arredondamento": bool(round(float(a2["H30_mean_px"]), 1) == round(float(a3["H30_mean_px"]), 1)
                                                    and round(float(a2["H30_std_px"]), 1) == round(float(a3["H30_std_px"]), 1)),
                "SR_igual_ao_arredondamento": bool(round(float(a2["SR@0.5"]), 3) == round(float(a3["SR@0.5"]), 3)),
                "SA_igual_ao_arredondamento": bool(round(float(a2["SA"]), 3) == round(float(a3["SA"]), 3)),
            }
            if hib in by:
                mesma["recalculado_da_trajetoria_escolhida"] = {"SA": by[hib]["SA"], "SR@0.5": by[hib]["SR05"],
                                                                "H30_mean_px": by[hib]["aval_H30_px_pers"],
                                                                "H30_std_px": by[hib]["aval_H30_dp_pers"],
                                                                "trajetoria": by[hib]["trajetoria"],
                                                                "original": by[hib]["trajetoria_original"]}
    base.write_json(out / "sot_tabela2_vs_tabela3.json", mesma)

    # -- detectores (valores publicados; nao reexecutados) -----------------------------
    det_rows = []
    t1 = base.read_csv(tab_dir / "table1_detectors.csv")
    if t1 is not None:
        for _, s in t1.iterrows():
            det_rows.append({"modelo": s["Modelo"], "precisao": s["Precisao"], "revocacao": s["Recall"],
                             "mAP50": s["mAP@0.5"], "mAP50_95": s["mAP@0.5:0.95"], "fonte": "Tabela 1 publicada",
                             "imagens_avaliadas": 24804})
    y26s = base.read_csv(base.P(cs["yolo26s"]))
    if y26s is not None and len(y26s):
        s = y26s.iloc[0]
        det_rows.append({"modelo": s["model"], "precisao": s["precision"], "revocacao": s["recall"],
                         "mAP50": s["mAP50"], "mAP50_95": s["mAP50_95"],
                         "fonte": "yolo26s_metrics.csv (revisao)", "imagens_avaliadas": int(s["validation_images"])})
    det = pd.DataFrame(det_rows)
    det["conjunto"] = "anti_uav_tracking_test_final, split test (as imagens de teste do rastreamento)"
    base.write_csv(det, out / "detectores.csv")
    base.log(f"Parte 1a concluida em {time.perf_counter() - t_start:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
