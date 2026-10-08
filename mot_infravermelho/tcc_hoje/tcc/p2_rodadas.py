"""Parte 2 -- rodadas novas, so acuracia (sem FPS), estado zerado por sequencia.

Correcao do persist: o Ultralytics reaproveita predictor.trackers quando persist=True
(ultralytics/trackers/track.py, on_predict_start). Aqui cada (configuracao, sequencia)
cria um modelo YOLO NOVO (ou um BOTSORT novo, no caminho SAHI), entao nenhuma
trilha, contador de id ou GMC passa de uma sequencia (ou de um gamma) para outra.

Fila (ordem do pedido): i) determinismo -> g) HEM -> h) HEM + SAHI 512 ->
f) GM + NSA-Kalman (gamma 0,55 e depois 0,15...0,85) -> e) leaderboard ->
[se o SAHI 512 novo diferir do antigo] SAHI 320/384/448/576 no fim.
Processos em paralelo (2 ou 3, pelo run_hoje.sh). Retomavel: cada tarefa grava
eval/<config>/<seq>.json por ultimo e e pulada se ele existir.
"""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context

import numpy as np
import pandas as pd

from . import base

NAN = float("nan")


# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------

def configs(cfg: dict) -> dict:
    p2 = cfg["parte2"]
    out = {}
    out["det_a"] = {"grupo": "i_determinismo", "tipo": "yolo", "pesos": "base", "tracker": "botsort.yaml",
                    "descricao": "Linha de base (best.pt + BoT-SORT), execução A"}
    out["det_b"] = dict(out["det_a"], descricao="Linha de base (best.pt + BoT-SORT), execução B")
    out["hem"] = {"grupo": "g_hem", "tipo": "yolo", "pesos": "hem", "tracker": "botsort.yaml",
                  "descricao": "HEM (best_hardmining.pt) + BoT-SORT padrão"}
    out["hem_sahi512"] = {"grupo": "h_sahi", "tipo": "sahi", "pesos": "hem", "tracker": "botsort.yaml", "slice": 512,
                          "overlap": float(p2["sahi_overlap"][512]),
                          "descricao": "HEM + SAHI 512 + BoT-SORT padrão"}
    for g in p2["gm_gammas"]:
        name = f"gm_nsa_g{int(round(float(g) * 100)):03d}"
        out[name] = {"grupo": "f_gm", "tipo": "yolo", "pesos": "base", "tracker": "botsort.yaml",
                     "gm": float(g), "nsa": True,
                     "descricao": f"best.pt + BoT-SORT + GM (gamma={g}) + NSA-Kalman"}
    for t in p2["leaderboard"]:
        out[f"lb_{t}"] = {"grupo": "e_leaderboard", "tipo": "yolo", "pesos": "base", "tracker": f"{t}.yaml",
                          "descricao": f"best.pt + {p2['rotulos_leaderboard'].get(t, t)} (config padrão)"}
    for s in p2["sahi_extra"]:
        out[f"hem_sahi{s}"] = {"grupo": "h_sahi_extra", "tipo": "sahi", "pesos": "hem", "tracker": "botsort.yaml",
                               "slice": int(s), "overlap": float(p2["sahi_overlap"][int(s)]),
                               "descricao": f"HEM + SAHI {s} + BoT-SORT padrão"}
    return out


def fila(cfg: dict, seqs: list[str], smoke: bool, com_extras: bool) -> list[tuple[str, str]]:
    p2 = cfg["parte2"]
    cf = configs(cfg)
    det = [s for s in p2["determinismo_seqs"] if s in seqs] or seqs[:2]
    q = [("det_a", s) for s in det] + [("det_b", s) for s in det]
    q += [("hem", s) for s in seqs]
    q += [("hem_sahi512", s) for s in seqs]
    for name in [n for n, c in cf.items() if c["grupo"] == "f_gm"]:
        q += [(name, s) for s in seqs]
    for name in [n for n, c in cf.items() if c["grupo"] == "e_leaderboard"]:
        q += [(name, s) for s in seqs]
    if com_extras:
        for name in [n for n, c in cf.items() if c["grupo"] == "h_sahi_extra"]:
            q += [(name, s) for s in seqs]
    return q


def paths(out, name: str, seq: str) -> dict:
    return {"traj": out / "traj" / name / f"{seq}.txt", "eval": out / "eval" / name / f"{seq}.json"}


# ---------------------------------------------------------------------------
# Execucao de uma tarefa (processo filho)
# ---------------------------------------------------------------------------

def _device():
    from motpred import common as mc

    dev = mc.resolve_device("auto")
    return f"cuda:{dev}" if dev.isdigit() else dev


def rodar_yolo(c: dict, weights, video, device: str, conf: float, max_frames, frame_wh):
    """Copia do laco de run_tracker_on_val / run_multitracker_leaderboard / run_robust_flit_sweep,
    quadro a quadro, com YOLO novo por sequencia."""
    from ultralytics import YOLO

    from motpred import common as mc
    from motpred.data import iter_video_frames

    mpc = base.mp_cfg()
    flit = nsa = None
    try:
        if c.get("gm") is not None:
            flit = mc.lafusion_flit(mpc)
            flit.patch_flit(gamma=float(c["gm"]), frame_w=int(frame_wh[0]), frame_h=int(frame_wh[1]))
        if c.get("nsa"):
            nsa = mc.lafusion_nsa(mpc)
            nsa.patch_nsa_kalman()
        model = YOLO(str(weights))
        lines = []
        n = 0
        for idx, frame in iter_video_frames(video, max_frames=max_frames):
            n = idx
            r = model.track(frame, persist=True, tracker=c["tracker"], conf=conf, device=device, verbose=False)[0]
            if r.boxes is None or r.boxes.id is None:
                continue
            boxes = r.boxes.xywh.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            for (cx, cy, w, h), tid in zip(boxes, ids):
                x = cx - w / 2
                y = cy - h / 2
                lines.append(f"{idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
        return lines, n
    finally:
        if nsa is not None:
            nsa.unpatch_nsa_kalman()
        if flit is not None:
            flit.unpatch_flit()


def rodar_sahi(c: dict, weights, video, device: str, conf: float, max_frames):
    """Copia de run_sahi_tracker.py::run_tracker_on_sequence (BOTSORT novo por sequencia)."""
    import torch
    from sahi import AutoDetectionModel
    from ultralytics.engine.results import Results
    from ultralytics.trackers import BOTSORT

    from motpred.data import iter_video_frames

    cfg = base.load_cfg()
    rst = base.load_module("lafusion_run_sahi_tracker", base.P(cfg["paths"]["run_sahi_tracker"]))
    det_model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
                                                   confidence_threshold=conf, device=device)
    tracker = BOTSORT(args=rst.load_tracker_config(rst.resolve_tracker_cfg(c["tracker"])))
    names = {0: "drone"}
    size = int(c["slice"])
    overlap = float(c["overlap"])
    lines = []
    n = 0
    for idx, frame in iter_video_frames(video, max_frames=max_frames):
        n = idx
        dets = rst.build_sahi_detections(frame, det_model, size, size, overlap, conf)
        box_tensor = torch.from_numpy(dets).to(device=device, dtype=torch.float32)
        result = Results(orig_img=frame, path="frame.jpg", names=names, boxes=box_tensor)
        det = result.boxes.cpu().numpy()
        tracks = tracker.update(det, frame)
        if len(tracks):
            for t in tracks:
                x1, y1, x2, y2 = t[:4]
                tid = int(t[4])
                lines.append(f"{idx},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1")
    return lines, n


def tarefa(name: str, seq: str, smoke: bool) -> dict:
    from motpred import data, mot_eval

    cfg = base.load_cfg()
    p2 = cfg["parte2"]
    c = configs(cfg)[name]
    out = base.sub(smoke, "parte2")
    pt = paths(out, name, seq)
    if pt["eval"].exists():
        return base.read_json(pt["eval"])
    mpc = base.mp_cfg()
    max_frames = int(cfg["smoke"]["max_frames"]) if smoke else None
    weights = base.P(p2["pesos"][c["pesos"]])
    video = data.video_path(mpc, seq)
    device = _device()
    conf = float(p2["conf"])
    t0 = time.perf_counter()
    if c["tipo"] == "yolo":
        lines, n = rodar_yolo(c, weights, video, device, conf, max_frames, p2["frame_wh"])
    else:
        lines, n = rodar_sahi(c, weights, video, device, conf, max_frames)
    base.write_text(pt["traj"], "\n".join(lines))
    gt_df, _ = data.load_gt(mpc, seq)
    if max_frames:
        gt_df = gt_df[gt_df["frame"] <= max_frames]
    pred_df = data.load_mot(mpc, pt["traj"])
    acc, _ = mot_eval.build_accumulator(gt_df, pred_df)
    metrics = mot_eval.compute_metrics(acc, f"{name}:{seq}")
    info = {"config": name, "seq": seq, "quadros": n, "max_frames": max_frames, "device": device,
            "pesos": str(weights), "tracker": c["tracker"], "segundos_parede": time.perf_counter() - t0,
            "pid": os.getpid(), **metrics}
    base.write_json(pt["eval"], info)
    return info


def _init_worker():
    os.environ.setdefault("YOLO_VERBOSE", "False")


# ---------------------------------------------------------------------------
# Comparacao com resultados antigos
# ---------------------------------------------------------------------------

def antigo_por_seq(mot_dir, arquivo: str):
    df = base.read_csv(mot_dir / arquivo, index_col=0)
    return df


def antigo_overall(df):
    if df is None:
        return {}
    if "OVERALL" in df.index:
        r = df.loc["OVERALL"]
        return {"MOTA": 100 * float(r["mota"]), "IDF1": 100 * float(r["idf1"]), "IDs": int(r["num_switches"])}
    return {}


def overall_of(rows: list[dict]) -> dict:
    from motpred import mot_eval

    if not rows:
        return {}
    o = mot_eval.overall(rows)
    return {"MOTA": 100 * o["mota"], "IDF1": 100 * o["idf1"], "IDs": int(o["num_switches"]),
            "FP": int(o["num_false_positives"]), "FN": int(o["num_misses"]), "sequencias": o["sequencias"]}


def coletar(out, name: str, seqs: list[str]) -> list[dict]:
    rows = []
    for s in seqs:
        p = paths(out, name, s)["eval"]
        if p.exists():
            rows.append(base.read_json(p))
    return rows


def decidir_sahi(cfg, out, seqs, smoke) -> dict | None:
    """SAHI 512 novo == antigo (por sequencia)? Grava parte2/sahi_decisao.json."""
    rows = coletar(out, "hem_sahi512", seqs)
    if len(rows) < len(seqs):
        return None
    old = antigo_por_seq(base.P(cfg["paths"]["mot_eval_dir"]), "mot_summary_hardmining_sahi512.csv")
    dif = []
    for r in rows:
        if old is None or r["seq"] not in old.index:
            dif.append({"seq": r["seq"], "sem_antigo": True})
            continue
        o = old.loc[r["seq"]]
        dif.append({"seq": r["seq"], "dif_mota": r["mota"] - float(o["mota"]), "dif_idf1": r["idf1"] - float(o["idf1"]),
                    "dif_ids": int(r["num_switches"]) - int(o["num_switches"])})
    identico = bool(old is not None and all(not d.get("sem_antigo") and abs(d["dif_mota"]) < 1e-9
                                            and abs(d["dif_idf1"]) < 1e-9 and d["dif_ids"] == 0 for d in dif))
    novo = overall_of(rows)
    dec = {"identico_por_sequencia": identico, "novo": novo, "antigo": antigo_overall(old),
           "antigo_publicado": {"MOTA": 79.6, "IDF1": 80.3, "IDs": 712},
           "acao": "manter as 4 linhas antigas (320/384/448/576) sem FPS" if identico
           else "rodar 320/384/448/576 no fim da fila", "por_sequencia": dif, "smoke": smoke}
    base.write_json(out / "sahi_decisao.json", dec)
    return dec


def determinismo(cfg, out, seqs, smoke) -> pd.DataFrame:
    from motpred import data

    mpc = base.mp_cfg()
    exp_b = base.P(cfg["paths"]["exp_b"])
    det = [s for s in cfg["parte2"]["determinismo_seqs"] if s in seqs] or seqs[:2]
    limit = int(cfg["smoke"]["max_frames"]) if smoke else None
    rows = []

    def load(path):
        if not path.exists():
            return None
        df = data.load_mot(mpc, path)
        return df[df["frame"] <= limit] if limit else df

    def compare(a, b, label, seq, ja=None, jb=None):
        row = {"seq": seq, "comparacao": label}
        if a is None or b is None:
            row["obs"] = "arquivo ausente"
            return row
        m = a.merge(b, on=["frame", "id"], how="outer", suffixes=("_a", "_b"), indicator=True)
        both = m[m["_merge"] == "both"]
        diff = np.abs(both[["x_a", "y_a", "w_a", "h_a"]].to_numpy() - both[["x_b", "y_b", "w_b", "h_b"]].to_numpy())
        row.update({"linhas_a": int(len(a)), "linhas_b": int(len(b)),
                    "so_em_a": int((m["_merge"] == "left_only").sum()), "so_em_b": int((m["_merge"] == "right_only").sum()),
                    "max_dif_px": float(diff.max()) if diff.size else 0.0,
                    "identicos": bool((m["_merge"] == "both").all() and (diff.size == 0 or diff.max() == 0.0))})
        if ja and jb:
            for k in ("mota", "idf1", "num_switches"):
                try:
                    row[f"dif_{k}"] = float(ja[k]) - float(jb[k])
                except (TypeError, ValueError, KeyError):
                    row[f"dif_{k}"] = NAN
        return row

    for s in det:
        pa, pb = paths(out, "det_a", s), paths(out, "det_b", s)
        a, b = load(pa["traj"]), load(pb["traj"])
        ja = base.read_json(pa["eval"]) if pa["eval"].exists() else None
        jb = base.read_json(pb["eval"]) if pb["eval"].exists() else None
        rows.append(compare(a, b, "A x B (mesma configuração, duas execuções)", s, ja, jb))
        ref = load(exp_b / "traj" / "baseline_ref" / f"{s}.txt")
        jr = None
        old_json = exp_b / "eval" / "baseline_ref" / f"{s}_mot.json"
        if not smoke and old_json.exists():
            o = base.read_json(old_json)
            jr = {"mota": o.get("mot_mota"), "idf1": o.get("mot_idf1"), "num_switches": o.get("mot_num_switches")}
        rows.append(compare(a, ref, "A x linha de base de 28/09", s, ja, jr))
    return pd.DataFrame(rows)


def antigos(cfg) -> dict:
    """Valores antigos (publicados/salvos) por configuracao, para 'Diferencas'."""
    mot_dir = base.P(cfg["paths"]["mot_eval_dir"])
    res = {"hem": antigo_overall(antigo_por_seq(mot_dir, "mot_summary_hardmining.csv")),
           "hem_sahi512": antigo_overall(antigo_por_seq(mot_dir, "mot_summary_hardmining_sahi512.csv"))}
    for s in cfg["parte2"]["sahi_extra"]:
        res[f"hem_sahi{s}"] = antigo_overall(antigo_por_seq(mot_dir, f"mot_summary_hardmining_sahi{s}.csv"))
    sv2 = base.read_csv(mot_dir / "sahi_summary_v2.csv")
    if sv2 is not None:
        for _, r in sv2[sv2["Model"] == "hardmining"].iterrows():
            key = f"hem_sahi{int(r['SAHI'])}"
            res.setdefault(key, {})
            res[key].update({"MOTA_tabela": float(r["MOTA"]), "IDF1_tabela": float(r["IDF1"]), "IDs_tabela": int(r["IDs"])})
    for fname in ("flit_nsa_sweep_summary.csv", "flit_nsa_sweep_summary_v2.csv"):
        df = base.read_csv(mot_dir / fname)
        if df is None:
            continue
        for _, r in df.iterrows():
            res[f"gm_nsa_g{int(round(float(r['gamma']) * 100)):03d}"] = {
                "MOTA": float(r["MOTA(%)"]), "IDF1": float(r["IDF1(%)"]), "IDs": int(r["IDs"])}
    lb = base.read_csv(mot_dir / "multitracker_leaderboard.csv")
    if lb is not None:
        for _, r in lb.iterrows():
            res[f"lb_{r['tracker']}"] = {"MOTA": float(r["MOTA(%)"]), "IDF1": float(r["IDF1(%)"]), "IDs": int(r["IDs"])}
    return res


def resumir(cfg, out, seqs, smoke):
    cf = configs(cfg)
    old = antigos(cfg)
    rows, per_seq = [], []
    for name, c in cf.items():
        sel = seqs
        if c["grupo"] == "i_determinismo":
            sel = [s for s in cfg["parte2"]["determinismo_seqs"] if s in seqs] or seqs[:2]
        got = coletar(out, name, sel)
        if not got:
            continue
        for r in got:
            per_seq.append({k: r.get(k) for k in ("config", "seq", "quadros", "mota", "idf1", "num_switches",
                                                  "num_false_positives", "num_misses", "num_objects",
                                                  "num_predictions", "idtp")})
        o = overall_of(got)
        a = old.get(name, {})
        rows.append({"config": name, "grupo": c["grupo"], "descricao": c["descricao"],
                     "sequencias": len(got), "sequencias_esperadas": len(sel), **o,
                     "MOTA_antigo": a.get("MOTA", a.get("MOTA_tabela", NAN)),
                     "IDF1_antigo": a.get("IDF1", a.get("IDF1_tabela", NAN)),
                     "IDs_antigo": a.get("IDs", a.get("IDs_tabela", NAN)),
                     "gamma": c.get("gm", NAN), "sahi": c.get("slice", NAN)})
    resumo = pd.DataFrame(rows)
    base.write_csv(resumo, out / "rodadas_resumo.csv")
    base.write_csv(pd.DataFrame(per_seq), out / "rodadas_por_seq.csv")
    base.write_csv(determinismo(cfg, out, seqs, smoke), out / "determinismo.csv")

    # BoT-SORT do leaderboard = linha de base de 28/09: conferencia de configuracao
    mpc = base.mp_cfg()
    ref = mpc.get("reference", {}).get("baseline", {})
    import ultralytics

    ycfg = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", "botsort.yaml")
    base.write_json(out / "botsort_config_check.json", {
        "linha_de_base_28_09": {"pesos": str(base.P("{root}/" + str(mpc["weights"][ref.get("weights", "baseline")]))),
                                "tracker": ref.get("tracker_cfg"), "conf": ref.get("conf"),
                                "estado": "modelo novo por sequencia; 2 chamadas de aquecimento + tracker.reset()"},
        "leaderboard_lafusion": {"pesos": "weights/best.pt (src/inference/common.py::WEIGHTS)", "tracker": "botsort.yaml",
                                 "conf": 0.25, "estado": "del predictor.trackers por sequencia"},
        "mesma_configuracao": bool(ref.get("tracker_cfg") == "botsort.yaml" and float(ref.get("conf", -1)) == 0.25
                                   and str(mpc["weights"][ref.get("weights", "baseline")]).endswith("best.pt")),
        "botsort_yaml_em_uso": ycfg,
        "ultralytics": ultralytics.__version__,
    })


# ---------------------------------------------------------------------------
# Orquestracao
# ---------------------------------------------------------------------------

def main() -> int:
    smoke = base.smoke_flag()
    workers = int(base.arg_value("--workers", "2"))
    cfg = base.load_cfg()
    out = base.sub(smoke, "parte2")
    seqs = base.val_sequences(smoke)
    dec_path = out / "sahi_decisao.json"
    falhas = []
    t0 = time.perf_counter()
    for _passada in range(3):          # 2a passada so se a decisao do SAHI sair depois da fila
        dec = base.read_json(dec_path) if dec_path.exists() else None
        if dec is None:
            dec = decidir_sahi(cfg, out, seqs, smoke)       # None se o SAHI 512 ainda nao terminou
        extras = bool(dec is not None and not dec.get("identico_por_sequencia", False))
        q = fila(cfg, seqs, smoke, extras)
        pend = [(n, s) for n, s in q if not paths(out, n, s)["eval"].exists() and (n, s) not in falhas]
        base.log(f"Parte 2: {len(q)} tarefas na fila, {len(pend)} pendentes, {workers} processos, "
                 f"{len(seqs)} sequencias"
                 f"{' (teste rapido: ' + str(cfg['smoke']['max_frames']) + ' quadros)' if smoke else ''}")
        if not pend:
            if dec is None and not dec_path.exists():
                break                                   # SAHI 512 falhou: sem decisao possivel
            if extras or dec is not None:
                break
        _executar(cfg, out, seqs, smoke, workers, pend, dec, falhas, t0)
        if dec is not None:
            break
    resumir(cfg, out, seqs, smoke)
    base.log(f"Parte 2 concluida em {(time.perf_counter() - t0) / 60:.1f} min; falhas: {len(falhas)}")
    if falhas:
        base.write_json(out / "falhas.json", [{"config": n, "seq": s} for n, s in falhas])
    return 1 if falhas else 0


def _executar(cfg, out, seqs, smoke, workers, pend, dec, falhas, t0):
    total = len(pend)
    feitas = 0
    if pend:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker) as ex:
            futs = {ex.submit(tarefa, n, s, smoke): (n, s) for n, s in pend}
            decided = dec is not None
            while futs:
                done, _ = wait(list(futs), return_when=FIRST_COMPLETED)
                for f in done:
                    n, s = futs.pop(f)
                    feitas += 1
                    try:
                        r = f.result()
                        el = time.perf_counter() - t0
                        eta = el / feitas * (total - feitas)
                        base.log(f"  [{feitas}/{total}] {n} {s}: MOTA {100 * r['mota']:.2f} IDF1 {100 * r['idf1']:.2f} "
                                 f"IDs {int(r['num_switches'])} ({el / 60:.1f} min; faltam ~{eta / 60:.0f} min)")
                    except Exception as exc:  # noqa: BLE001
                        falhas.append((n, s))
                        base.log(f"  ERRO {n} {s}: {exc!r}")
                        base.write_text(out / "erros" / f"{n}_{s}.txt", traceback.format_exc() + "\n" + repr(exc))
                if not decided:
                    d = decidir_sahi(cfg, out, seqs, smoke)
                    if d is not None:
                        decided = True
                        base.log(f"  SAHI 512: {d['acao']} (novo {d['novo']}, antigo {d['antigo']})")
                        if not d["identico_por_sequencia"]:
                            for n, s in [(n, s) for n, s in fila(cfg, seqs, smoke, True)
                                         if configs(cfg)[n]["grupo"] == "h_sahi_extra"
                                         and not paths(out, n, s)["eval"].exists()]:
                                futs[ex.submit(tarefa, n, s, smoke)] = (n, s)
                                total += 1


if __name__ == "__main__":
    raise SystemExit(main())
