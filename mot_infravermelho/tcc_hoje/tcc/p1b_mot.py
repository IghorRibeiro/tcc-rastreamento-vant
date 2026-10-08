"""Parte 1b -- predicao com identidade no MOT sobre as trajetorias de 28/09.

Variantes: linha de base (baseline_ref, MOTA 75,96), HIT-Tracker (hit_ref, 79,91) e
hibrido CSRT (hybrid_csrt_n15, 60,92). Correspondencia primaria: CLEAR MOT (eventos
MATCH/SWITCH do py-motmetrics, IoU >= 0,5); a do IDF1 vai so para o CSV de apendice.
Kalman do SIGE publicado (motpred.prediction = laco de uav_common.prediction_errors),
persistente e amortecido (0,95), erro em px e normalizado pela diagonal do GT em t+H.
Uma tarefa por sequencia (as 3 variantes juntas, para o suporte comum), em paralelo.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd

from . import base, estat

NAN = float("nan")


def tarefa(seq: str, smoke: bool) -> str:
    from motpred import common as mc
    from motpred import data, mot_eval, prediction

    cfg = base.load_cfg()
    mpc = base.mp_cfg()
    parts = base.sub(smoke, "parte1") / "_mot_partes"
    final = parts / f"{seq}_metricas.csv"
    if final.exists():
        return seq
    uc = mc.uav_common(mpc)
    horizons = [int(h) for h in cfg["mot"]["horizons"]]
    dampings = [float(d) for d in cfg["mot"]["dampings"]]
    exp_b = base.P(cfg["paths"]["exp_b"])
    gt_df, gt_index = data.load_gt(mpc, seq)

    met_rows, pred_frames, sup_frames = [], [], []
    clear_pairs = {}
    for variant in cfg["mot"]["variantes"]:
        traj = exp_b / "traj" / variant / f"{seq}.txt"
        if not traj.exists():
            met_rows.append({"variante": variant, "seq": seq, "trajetoria_ausente": True})
            continue
        t0 = time.perf_counter()
        pred_df = data.load_mot(mpc, traj)
        metrics, corr = mot_eval.evaluate_mot_sequence(gt_df, pred_df, f"{variant}:{seq}")
        pairs, counts, gaps = prediction.pair_errors(uc, pred_df, gt_index, corr, horizons, dampings)
        agg = prediction.aggregate_sequence(pairs, counts)
        agg.insert(0, "seq", seq)
        agg.insert(0, "variante", variant)
        pred_frames.append(agg)
        clear_pairs[variant] = pairs[pairs["mode"] == "clear"]
        met_rows.append({"variante": variant, "seq": seq, "trajetoria_ausente": False,
                         **metrics, "densidade_media": gt_index.mean_density,
                         "densidade_max": gt_index.max_density, "quadros_anotados": gt_index.n_frames_annotated,
                         **{f"lacunas_{k}": v for k, v in gaps.items()},
                         "segundos": time.perf_counter() - t0})
    for a, b in cfg["mot"]["pares_suporte_comum"]:
        if a in clear_pairs and b in clear_pairs:
            cs = prediction.common_support(clear_pairs[a], clear_pairs[b], mode="clear")
            cs.insert(0, "seq", seq)
            cs.insert(0, "b", b)
            cs.insert(0, "a", a)
            sup_frames.append(cs)

    w = gt_df["w"].to_numpy(float)
    h = gt_df["h"].to_numpy(float)
    d = np.hypot(w, h)
    alvos = pd.DataFrame([{"seq": seq, "caixas_gt": int(len(gt_df)), "largura_media_px": float(w.mean()),
                           "altura_media_px": float(h.mean()), "diagonal_media_px": float(d.mean()),
                           "diagonal_mediana_px": float(np.median(d)),
                           "soma_w": float(w.sum()), "soma_h": float(h.sum()), "soma_d": float(d.sum())}])

    base.write_csv(pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame(),
                   parts / f"{seq}_pred.csv")
    base.write_csv(pd.concat(sup_frames, ignore_index=True) if sup_frames else pd.DataFrame(),
                   parts / f"{seq}_suporte.csv")
    base.write_csv(alvos, parts / f"{seq}_alvos.csv")
    base.write_csv(pd.DataFrame(met_rows), final)        # por ultimo: marca a sequencia como feita
    return seq


def main() -> int:
    smoke = base.smoke_flag()
    cfg = base.load_cfg()
    out = base.sub(smoke, "parte1")
    parts = out / "_mot_partes"
    parts.mkdir(parents=True, exist_ok=True)
    seqs = base.val_sequences(smoke)
    exp_b = base.P(cfg["paths"]["exp_b"])
    faltando = {v: [s for s in seqs if not (exp_b / "traj" / v / f"{s}.txt").exists()] for v in cfg["mot"]["variantes"]}
    base.write_json(out / "mot_trajetorias_checagem.json",
                    {"exp_b": str(exp_b), "sequencias": seqs, "faltando": faltando})
    if any(faltando.values()):
        base.log(f"AVISO: trajetorias ausentes: {faltando}")

    todo = [s for s in seqs if not (parts / f"{s}_metricas.csv").exists()]
    workers = max(1, min(int(cfg["mot"]["workers_cpu"]), len(todo) or 1))
    base.log(f"Parte 1b: {len(todo)} de {len(seqs)} sequencias a calcular ({workers} processos)")
    t0 = time.perf_counter()
    falhas = []
    if todo:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
            futs = {ex.submit(tarefa, s, smoke): s for s in todo}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    fut.result()
                    base.log(f"  {s} ok ({time.perf_counter() - t0:.0f} s)")
                except Exception as exc:  # noqa: BLE001
                    falhas.append(s)
                    base.log(f"  ERRO {s}: {exc!r}")

    def cat(suffix):
        frames = []
        for s in seqs:
            df = base.read_csv(parts / f"{s}_{suffix}.csv")
            if df is not None and len(df):
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    met = cat("metricas")
    pred = cat("pred")
    sup = cat("suporte")
    alvos = cat("alvos")
    base.write_csv(met, out / "mot_metricas_por_seq.csv")
    if not pred.empty:
        base.write_csv(pred[pred["mode"] == "clear"], out / "mot_pred_por_seq.csv")
        base.write_csv(pred[pred["mode"] == "idf1"], out / "mot_pred_apendice_idf1_por_seq.csv")
    base.write_csv(sup, out / "mot_suporte_comum_por_seq.csv")
    base.write_csv(alvos.drop(columns=["soma_w", "soma_h", "soma_d"], errors="ignore"), out / "mot_alvos_por_seq.csv")

    # -- agregados -------------------------------------------------------------------
    from motpred import mot_eval

    geral = []
    ov = base.read_csv(base.P(cfg["paths"]["exp_b_overview"]))
    for variant, label in cfg["mot"]["variantes"].items():
        rows = met[(met["variante"] == variant) & (~met["trajetoria_ausente"].astype(bool))].to_dict("records") \
            if not met.empty else []
        if not rows:
            continue
        o = mot_eval.overall(rows)
        row = {"variante": variant, "rotulo": label, "sequencias": o["sequencias"], "MOTA": 100 * o["mota"],
               "IDF1": 100 * o["idf1"], "IDs": o["num_switches"], "FP": o["num_false_positives"],
               "FN": o["num_misses"], "objetos_gt": o["num_objects"]}
        if ov is not None and "variante" in ov.columns:
            sel = ov[ov["variante"] == variant]
            if len(sel):
                for col in ("MOTA", "IDF1"):
                    for cand in (f"{col}_%", col, col.lower(), f"{col}(%)"):
                        if cand in sel.columns:
                            row[f"{col}_28_09"] = float(sel.iloc[0][cand])
                            break
        geral.append(row)
    base.write_csv(pd.DataFrame(geral), out / "mot_metricas_geral.csv")

    seed, nb = int(cfg["seed"]), int(cfg["bootstrap_n"])
    res = []
    if not pred.empty:
        for (variant, mode, damping, horizon), grp in pred.groupby(["variante", "mode", "damping", "H"]):
            n = grp["n"].to_numpy(float)
            row = {"variante": variant, "modo": mode, "damping": float(damping), "H": int(horizon),
                   "sequencias": int(len(grp)), "pares": int(n.sum()),
                   "media_seq_px": float(grp["mean_px"].mean()), "media_seq_norm": float(grp["mean_norm"].mean()),
                   "media_pares_px": float(np.nansum(grp["mean_px"] * n) / n.sum()) if n.sum() else NAN,
                   "media_pares_norm": float(np.nansum(grp["mean_norm"] * n) / n.sum()) if n.sum() else NAN,
                   "cobertura": float(grp["validos"].sum() / grp["passos"].sum()) if grp["passos"].sum() else NAN,
                   "sem_assoc": int(grp["sem_assoc"].sum()), "ausente": int(grp["ausente"].sum())}
            if mode == "clear":
                for col, key in (("mean_px", "px"), ("mean_norm", "norm")):
                    lo, hi, _ = estat.bca_mean(grp[col], seed=seed, n_resamples=nb)
                    row[f"ic95_{key}_inf"], row[f"ic95_{key}_sup"] = lo, hi
            res.append(row)
    resumo = pd.DataFrame(res)
    if not resumo.empty:
        base.write_csv(resumo[resumo["modo"] == "clear"], out / "mot_pred_resumo.csv")
        base.write_csv(resumo[resumo["modo"] == "idf1"], out / "mot_pred_apendice_idf1_resumo.csv")

    sres = []
    if not sup.empty:
        for (a, b, damping, horizon), grp in sup.groupby(["a", "b", "damping", "H"]):
            dn = grp["mean_norm_a"] - grp["mean_norm_b"]
            sres.append({"a": a, "b": b, "damping": float(damping), "H": int(horizon), "sequencias": int(len(grp)),
                         "pares_comuns": int(grp["n_comum"].sum()),
                         "media_px_a": float(grp["mean_px_a"].mean()), "media_px_b": float(grp["mean_px_b"].mean()),
                         "media_norm_a": float(grp["mean_norm_a"].mean()),
                         "media_norm_b": float(grp["mean_norm_b"].mean()),
                         "dif_norm_media": float(dn.mean())})
    base.write_csv(pd.DataFrame(sres), out / "mot_suporte_comum_resumo.csv")

    if not alvos.empty:
        tot = alvos["caixas_gt"].sum()
        base.write_json(out / "mot_alvos_resumo.json", {
            "sequencias": int(len(alvos)), "caixas_gt": int(tot),
            "largura_media_px": float(alvos["soma_w"].sum() / tot), "altura_media_px": float(alvos["soma_h"].sum() / tot),
            "diagonal_media_px": float(alvos["soma_d"].sum() / tot),
            "diagonal_media_das_seq_px": float(alvos["diagonal_media_px"].mean())})
    base.log(f"Parte 1b concluida em {time.perf_counter() - t0:.0f} s")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
