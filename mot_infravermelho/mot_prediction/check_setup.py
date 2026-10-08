#!/usr/bin/env python3
"""ETAPA 0 - Verificacao do ambiente e dos dados. NAO roda experimento.

Imprime e grava em mot_prediction/check_setup.txt:
  1. versoes das bibliotecas, dispositivo, OpenCV com CSRT/KCF/MOSSE (contrib);
  2. se todos os caminhos do config.yaml (e do paths.txt, se existir) existem;
  3. as 30 sequencias de validacao (split do LAFusion x manifesto da auditoria);
  4. se as trajetorias da linha de base e do HIT-Tracker existem e QUAIS reproduzem
     as metricas publicadas (grava results/setup/trajectory_selection.json);
  5. se o conjunto de avaliacao esta subamostrado (passo 4) e os quadros por sequencia;
  6. a taxa de quadros do Track 3 e do DUT Anti-UAV e os horizontes em quadros e segundos;
  7. se o BoT-SORT esta configurado com ou sem ReID;
  8. o Kalman do SIGE (parametros lidos do proprio uav_common.py);
  9. a referencia visivel do DUT para H4.

Uso:
    python mot_prediction/check_setup.py              # completo (~5-8 min: reavalia as trajetorias)
    python mot_prediction/check_setup.py --rapido     # sem reavaliar metricas (usa a selecao gravada)
Saida: codigo 0 sem problema bloqueante, 1 com problema bloqueante.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from motpred.common import (MP_DIR, ROOT, environment_info, lafusion_common, load_config,  # noqa: E402
                            out_dirs, read_json, rpath, selection_file, set_seeds, sha256_file,
                            sige_rastreadores, uav_common, write_json)

EXPECTED = {"ultralytics": "8.4.113", "numpy": "1.26.4", "pandas": "2.3.3", "scipy": "1.17.1",
            "motmetrics": "1.4.0", "sahi": "0.12.2"}


class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.blocking: list[str] = []
        self.warnings: list[str] = []

    def section(self, title: str) -> None:
        self.add("")
        self.add("=" * 78)
        self.add(title)
        self.add("=" * 78)

    def add(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def ok(self, text: str) -> None:
        self.add(f"  [ok]    {text}")

    def warn(self, text: str) -> None:
        self.add(f"  [aviso] {text}")
        self.warnings.append(text)

    def fail(self, text: str) -> None:
        self.add(f"  [ERRO]  {text}")
        self.blocking.append(text)

    def info(self, text: str) -> None:
        self.add(f"          {text}")


# ---------------------------------------------------------------------------
# 1. Ambiente
# ---------------------------------------------------------------------------

def check_environment(cfg, rep: Report) -> dict:
    rep.section("1. AMBIENTE")
    env = environment_info()
    rep.ok(f"Python {env['python']} ({env['executavel']})")
    rep.ok(f"Plataforma: {env['plataforma']}  CPU: {env.get('cpu', 'desconhecida')}")
    packages = env.get("pacotes", {})
    for name, expected in EXPECTED.items():
        got = packages.get(name)
        if got is None:
            rep.fail(f"{name}: AUSENTE (esperado {expected})")
        elif got != expected:
            rep.warn(f"{name} {got} (o LAFusion usou {expected}; diferencas mudam as ultimas casas)")
        else:
            rep.ok(f"{name} {got}")
    for name in ("torch", "torchvision", "matplotlib", "PyYAML", "lap", "lapx"):
        if name in packages:
            rep.ok(f"{name} {packages[name]}")
    try:
        import torch

        mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        rep.ok(f"CUDA={torch.cuda.is_available()}  MPS={mps}")
        if not mps and not torch.cuda.is_available():
            rep.warn("sem MPS/CUDA: o detector rodara na CPU; FPS nao comparavel ao do artigo")
    except ImportError:
        rep.fail("PyTorch ausente")

    opencv = sorted(k for k in packages if k.startswith("opencv"))
    rep.add("")
    rep.ok(f"OpenCV (cv2) {env.get('cv2_version')}  pacotes instalados: {', '.join(opencv) or 'nenhum'}")
    if len(opencv) > 1:
        rep.fail("mais de um pacote OpenCV instalado; eles se sobrepoem. Deixe so o "
                 "opencv-contrib-python (README_RUN.md, passo 1)")
    if "opencv-contrib-python" not in opencv:
        rep.warn("opencv-contrib-python nao aparece nos metadados; o CSRT fica no contrib")

    sige04 = sige_rastreadores(cfg)
    available = {}
    for name in ("csrt", "kcf", "mosse", "medianflow"):
        try:
            sige04.make_local_tracker(name)
            available[name] = True
        except Exception as error:  # noqa: BLE001
            available[name] = False
            available[f"{name}_erro"] = str(error).splitlines()[0]
    line = ", ".join(f"{k.upper()}={'sim' if v else 'NAO'}" for k, v in available.items() if not k.endswith("_erro"))
    rep.info(f"rastreadores locais (make_local_tracker do SIGE): {line}")
    if not available.get("csrt"):
        rep.fail("CSRT indisponivel: instale opencv-contrib-python (Experimento B depende dele)")
    for name in ("kcf", "mosse"):
        if not available.get(name):
            rep.warn(f"{name.upper()} indisponivel: a varredura correspondente vai falhar")

    # teste funcional em caixa pequena (alvos do Track 3 tem ~10-25 px)
    if available.get("csrt"):
        import cv2

        rng = np.random.default_rng(int(cfg["seed"]))
        h, w = int(cfg["track3"]["frame_h"]), int(cfg["track3"]["frame_w"])
        frame = rng.integers(0, 60, size=(h, w, 3), dtype=np.uint8)
        frame[250:259, 300:312] = 230
        moved = np.roll(frame, 2, axis=1)
        try:
            tracker = sige04.make_local_tracker("csrt")
            tracker.init(frame, (300, 250, 12, 9))
            ok, box = tracker.update(moved)
            shown = tuple(int(v) for v in box) if box is not None else None
            rep.ok(f"CSRT em caixa 12x9 px: init+update sem erro (sucesso={ok}, caixa={shown})")
            available["csrt_caixa_pequena"] = True
        except cv2.error as error:
            rep.warn(f"CSRT falhou em caixa 12x9 px: {str(error).splitlines()[0]} "
                     "(o hibrido trata como falha do rastreador e conta em 'exceptions')")
            available["csrt_caixa_pequena"] = False
    env["rastreadores"] = available
    return env


# ---------------------------------------------------------------------------
# 2. Caminhos
# ---------------------------------------------------------------------------

def check_paths(cfg, rep: Report) -> None:
    rep.section("2. CAMINHOS (config.yaml e paths.txt)")
    required = ["lafusion_root", "sige_repo", "track3_label_root", "track3_video_root", "mot_eval_dir"]
    for key, value in cfg["paths"].items():
        path = rpath(value)
        if key in ("results", "logs"):
            rep.ok(f"{key}: {path} (criado se preciso)")
            continue
        if path is not None and path.exists():
            rep.ok(f"{key}: {path}")
        elif key in required:
            rep.fail(f"{key}: NAO EXISTE -> {path}")
        else:
            rep.warn(f"{key}: nao existe -> {path}")
    for key, value in cfg["weights"].items():
        path = rpath(value)
        if path is not None and path.exists():
            rep.ok(f"peso '{key}': {path}  ({path.stat().st_size / 1e6:.1f} MB, sha256 {sha256_file(path)[:16]}...)")
        elif key in ("baseline",):
            rep.fail(f"peso '{key}' NAO EXISTE: {path}")
        else:
            rep.warn(f"peso '{key}' nao existe: {path}")

    common = lafusion_common(cfg)
    for attr, key in (("LABEL_ROOT", "track3_label_root"), ("VIDEO_ROOT", "track3_video_root")):
        theirs = Path(getattr(common, attr)).resolve()
        ours = rpath(cfg["paths"][key]).resolve()
        if theirs == ours:
            rep.ok(f"src/inference/common.py::{attr} = config.paths.{key}")
        else:
            rep.fail(f"src/inference/common.py::{attr} ({theirs}) difere de config.paths.{key} ({ours}); "
                     "o split de validacao vem do common.py")

    paths_txt = rpath(cfg["paths"].get("paths_txt"))
    rep.add("")
    if paths_txt is None or not paths_txt.exists():
        rep.warn(f"paths.txt nao encontrado em {paths_txt}; os caminhos acima (config.yaml) foram usados")
        return
    rep.ok(f"paths.txt: {paths_txt}")
    for number, raw in enumerate(paths_txt.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        candidates = re.findall(r"(~?[\w./\\-]*[/\\][\w./\\ -]*)", text)
        for candidate in candidates:
            candidate = candidate.strip().strip("\"'")
            path = rpath(candidate)
            if path is not None and path.exists():
                rep.ok(f"paths.txt linha {number}: {candidate}")
            else:
                rep.warn(f"paths.txt linha {number}: NAO EXISTE -> {candidate}")


# ---------------------------------------------------------------------------
# 3-5. Sequencias, subamostragem e taxa de quadros
# ---------------------------------------------------------------------------

def check_sequences(cfg, rep: Report) -> list[str]:
    rep.section("3. SEQUENCIAS DE VALIDACAO")
    from motpred.data import val_sequences

    seqs = val_sequences(cfg)
    rep.ok(f"get_val_sequences(): {len(seqs)} sequencias")
    rep.info(", ".join(seqs))
    manifest_path = rpath(cfg["paths"]["manifest"])
    if manifest_path and manifest_path.exists():
        manifest = read_json(manifest_path)
        listed = manifest.get("split", {}).get("sequences", [])
        if sorted(listed) == sorted(seqs):
            rep.ok("identicas as do manifesto da auditoria (audit/validated_runs/run_manifest.json)")
        else:
            rep.fail("as sequencias diferem do manifesto da auditoria")
    if len(seqs) != 30:
        rep.warn(f"esperadas 30 sequencias; encontradas {len(seqs)}")
    return seqs


def check_subsampling(cfg, rep: Report, seqs: list[str], baseline_suffix: str | None) -> dict:
    rep.section("5. SUBAMOSTRAGEM E QUADROS POR SEQUENCIA")
    from motpred.data import gt_path, load_mot, trajectory_path, video_path, video_properties

    yolo_val = rpath(cfg["paths"]["track3_yolo_val_images"])
    rows = []
    for seq in seqs:
        row = {"seq": seq}
        gt = load_mot(cfg, gt_path(cfg, seq))
        frames = np.unique(gt["frame"].to_numpy())
        row["gt_quadros"] = int(frames.size)
        row["gt_min"] = int(frames.min()) if frames.size else 0
        row["gt_max"] = int(frames.max()) if frames.size else 0
        row["gt_passo_max"] = int(np.diff(frames).max()) if frames.size > 1 else 0
        row["gt_passo_min"] = int(np.diff(frames).min()) if frames.size > 1 else 0
        props = video_properties(video_path(cfg, seq))
        row["video_quadros"] = props.get("quadros", 0)
        row["video_fps"] = props.get("fps", float("nan"))
        row["video_res"] = f"{props.get('largura', 0)}x{props.get('altura', 0)}"
        if baseline_suffix:
            traj = trajectory_path(cfg, baseline_suffix, seq)
            if traj.exists():
                tf = np.unique(load_mot(cfg, traj)["frame"].to_numpy())
                row["traj_max"] = int(tf.max()) if tf.size else 0
                row["traj_passo_min"] = int(np.diff(tf).min()) if tf.size > 1 else 0
        if yolo_val is not None and yolo_val.is_dir():
            row["yolo_val_imgs"] = sum(1 for _ in yolo_val.glob(f"{seq}*"))
        rows.append(row)
    table = pd.DataFrame(rows)
    for text in table.to_string(index=False).splitlines():
        rep.info(text)

    full_rate = bool((table["gt_passo_min"] == 1).all()) if len(table) else False
    rep.add("")
    if full_rate:
        rep.ok("AVALIACAO SEM SUBAMOSTRAGEM: o gt.txt tem quadros consecutivos (passo 1) em todas as "
               "sequencias; o rastreamento roda no video completo (run_tracker_on_val.py)")
    else:
        rep.warn("o gt.txt NAO tem quadros consecutivos em todas as sequencias: verificar subamostragem")
    if "traj_passo_min" in table:
        if (table["traj_passo_min"].fillna(1) == 1).all():
            rep.ok("as trajetorias salvas tambem sao quadro a quadro")
        else:
            rep.warn("trajetorias com passo > 1 em alguma sequencia")
    if "yolo_val_imgs" in table and table["yolo_val_imgs"].sum() > 0:
        ratio = table["gt_quadros"].sum() / max(1, table["yolo_val_imgs"].sum())
        rep.info(f"conjunto de DETECCAO dataset_yolo/images/val: {int(table['yolo_val_imgs'].sum())} imagens "
                 f"(~1/{ratio:.1f} dos quadros: e o stride 4 do split_train_val.py; NAO e usado no rastreamento)")
    rep.info(f"total: {int(table['gt_quadros'].sum())} quadros anotados em {len(table)} sequencias")
    return {"sem_subamostragem": full_rate, "tabela": table.to_dict(orient="records"),
            "quadros_total": int(table["gt_quadros"].sum()),
            "fps_track3": sorted({round(float(v), 3) for v in table["video_fps"] if np.isfinite(v)})}


def check_frame_rates(cfg, rep: Report, sub: dict) -> dict:
    rep.section("6. TAXA DE QUADROS E HORIZONTES")
    fps_values = sub.get("fps_track3", [])
    declared = float(cfg["track3"]["fps_declared"])
    if fps_values:
        rep.ok(f"Track 3: fps medido nos .mp4 = {fps_values} (declarado no config: {declared})")
        fps_track3 = fps_values[0] if len(fps_values) == 1 else declared
        if len(fps_values) > 1:
            rep.warn("fps diferentes entre videos; usando o declarado")
    else:
        fps_track3 = declared
        rep.warn(f"fps do Track 3 nao medido; usando o declarado ({declared})")
    step = 1 if sub.get("sem_subamostragem") else 4
    fps_dut = cfg["dut"].get("fps_declared")
    dut_note = "declarado no config" if fps_dut else "desconhecido"
    for key in ("images_dir", "gt_dir"):
        base = rpath(cfg["dut"].get(key))
        if base is not None and base.exists():
            videos = [p for p in base.rglob("*") if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")][:5]
            if videos:
                from motpred.data import video_properties

                measured = video_properties(videos[0]).get("fps")
                if measured and measured > 0:
                    fps_dut = measured
                    dut_note = f"medido em {videos[0].name}"
    if fps_dut:
        rep.ok(f"DUT Anti-UAV: {fps_dut} fps ({dut_note})")
    else:
        rep.warn("DUT Anti-UAV: taxa de quadros nao determinavel (o dataset e distribuido como quadros JPEG); "
                 "informe dut.fps_declared para converter os horizontes do SIGE em segundos")
    rep.add("")
    rep.info("Horizonte | Track 3 (quadros originais) | Track 3 (s) | DUT (s)")
    table = []
    for horizon in cfg["prediction"]["horizons"]:
        original = horizon * step
        seconds = original / fps_track3 if fps_track3 else float("nan")
        dut_s = horizon / float(fps_dut) if fps_dut else float("nan")
        table.append({"H": horizon, "quadros_originais_track3": original, "segundos_track3": seconds,
                      "segundos_dut": dut_s})
        dut_text = f"{dut_s:.2f}" if np.isfinite(dut_s) else "?"
        rep.info(f"   H{horizon:<3}  | {original:>5}                        | {seconds:>6.2f}      | {dut_text}")
    return {"fps_track3": fps_track3, "fps_dut": fps_dut, "passo_avaliacao": step, "horizontes": table}


# ---------------------------------------------------------------------------
# 6. BoT-SORT
# ---------------------------------------------------------------------------

def check_botsort(cfg, rep: Report) -> dict:
    rep.section("7. CONFIGURACAO DO BoT-SORT")
    import yaml
    from ultralytics import __file__ as ultra_file

    path = Path(ultra_file).parent / "cfg" / "trackers" / "botsort.yaml"
    if not path.exists():
        rep.fail(f"botsort.yaml nao encontrado em {path}")
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    reid = bool(data.get("with_reid"))
    rep.ok(f"{path}")
    rep.ok(f"with_reid = {data.get('with_reid')}  ->  BoT-SORT {'COM' if reid else 'SEM'} ReID")
    for key in ("gmc_method", "track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer",
                "match_thresh", "fuse_score", "proximity_thresh", "appearance_thresh"):
        rep.info(f"{key} = {data.get(key)}")
    rep.info("run_tracker_on_val.py (linha de base) e run_sahi_flit_nsa.py (HIT-Tracker) usam 'botsort.yaml'.")
    if reid:
        rep.warn("with_reid=True: o README do LAFusion documenta que ReID desliga o GMC nesta versao")
    return {"botsort_yaml": str(path), **{k: data.get(k) for k in data}}


# ---------------------------------------------------------------------------
# 7. Trajetorias salvas e selecao
# ---------------------------------------------------------------------------

def evaluate_candidate(cfg, seqs, suffix, published: pd.DataFrame | None) -> dict:
    from motpred.data import load_gt, load_mot, trajectory_path
    from motpred.mot_eval import evaluate_mot_sequence, overall

    tol = float(cfg["trajectories"]["per_sequence_tolerance"])
    rows = []
    per_seq_ok = True
    idf1_ok = True
    for seq in seqs:
        gt_df, _ = load_gt(cfg, seq)
        pred = load_mot(cfg, trajectory_path(cfg, suffix, seq))
        metrics, _ = evaluate_mot_sequence(gt_df, pred, seq, max_iou=float(cfg["track3"]["iou_threshold"]))
        idf1_ok &= metrics["idf1_confere"]
        match = None
        if published is not None and seq in published.index:
            ref = published.loc[seq]
            match = (abs(metrics["mota"] - float(ref["mota"])) <= tol
                     and abs(metrics["idf1"] - float(ref["idf1"])) <= tol
                     and int(round(metrics["num_switches"])) == int(round(float(ref["num_switches"]))))
            per_seq_ok &= bool(match)
        else:
            per_seq_ok = False
        rows.append({**metrics, "seq": seq, "confere_publicado": match})
    agg = overall(rows)
    return {"overall": agg, "confere_por_sequencia": bool(per_seq_ok), "idf1_reconstruido_confere": bool(idf1_ok),
            "por_sequencia": [{k: r[k] for k in ("seq", "mota", "idf1", "num_switches", "confere_publicado")}
                              for r in rows]}


def check_trajectories(cfg, rep: Report, seqs: list[str], quick: bool) -> dict:
    rep.section("4. TRAJETORIAS SALVAS (EXPERIMENTO A) E SELECAO")
    from motpred.data import trajectory_path

    sel_path = selection_file(cfg)
    previous = read_json(sel_path) if sel_path.exists() else None
    manifest_path = rpath(cfg["paths"]["manifest"])
    manifest_runs = {}
    if manifest_path and manifest_path.exists():
        for run in read_json(manifest_path).get("runs", []):
            manifest_runs[run.get("run")] = run

    selection = {"gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"), "rastreadores": {}}
    tol_pp = float(cfg["trajectories"]["overall_tolerance_pp"])
    for tracker in ("baseline", "hit"):
        tcfg = cfg["trajectories"][tracker]
        pub_path = rpath(tcfg["published_summary"])
        published = pd.read_csv(pub_path, index_col=0) if pub_path and pub_path.exists() else None
        target = tcfg["published_overall"]
        rep.add("")
        rep.ok(f"{tracker}: publicado MOTA {target['mota']}% / IDF1 {target['idf1']}% / IDs {target['ids']} "
               f"({pub_path.name if pub_path else '-'})")
        candidates_info = []
        chosen = None
        for suffix in tcfg["candidates"]:
            files = [trajectory_path(cfg, suffix, s) for s in seqs]
            present = [p for p in files if p.exists()]
            info = {"sufixo": suffix, "arquivos": len(present), "esperados": len(files)}
            rep.info(f"candidato '{suffix}': {len(present)}/{len(files)} arquivos UAV*_{suffix}.txt")
            if len(present) != len(files):
                candidates_info.append(info)
                continue
            hashes = {s: sha256_file(p) for s, p in zip(seqs, files)}
            info["sha256"] = hashes
            run = manifest_runs.get(suffix)
            if run and run.get("prediction_sha256"):
                same = all(run["prediction_sha256"].get(s) == h for s, h in hashes.items())
                info["igual_ao_manifesto"] = same
                rep.info(f"   sha256 {'IGUAL' if same else 'DIFERENTE'} ao manifesto da auditoria (25/08/2026)")
            cached = None
            if previous:
                for old in previous.get("rastreadores", {}).get(tracker, {}).get("candidatos", []):
                    if old.get("sufixo") == suffix and old.get("sha256") == hashes and "overall" in old:
                        cached = old
            if cached is not None:
                info.update({k: cached[k] for k in ("overall", "confere_por_sequencia",
                                                     "idf1_reconstruido_confere", "por_sequencia") if k in cached})
                rep.info("   metricas reaproveitadas da verificacao anterior (arquivos identicos)")
            elif not quick:
                t0 = time.perf_counter()
                info.update(evaluate_candidate(cfg, seqs, suffix, published))
                rep.info(f"   reavaliado em {time.perf_counter() - t0:.0f} s")
            if "overall" in info:
                agg = info["overall"]
                d_mota = abs(agg["mota"] * 100 - float(target["mota"]))
                d_idf1 = abs(agg["idf1"] * 100 - float(target["idf1"]))
                info["confere_overall"] = bool(d_mota <= tol_pp and d_idf1 <= tol_pp
                                               and int(agg["num_switches"]) == int(target["ids"]))
                rep.info(f"   recalculado: MOTA {agg['mota'] * 100:.2f}% / IDF1 {agg['idf1'] * 100:.2f}% / "
                         f"IDs {int(agg['num_switches'])}  | por sequencia = publicado: "
                         f"{'SIM' if info.get('confere_por_sequencia') else 'NAO'}  | overall confere: "
                         f"{'SIM' if info['confere_overall'] else 'NAO'}  | IDF1 reconstruido = motmetrics: "
                         f"{'SIM' if info.get('idf1_reconstruido_confere') else 'NAO'}")
            candidates_info.append(info)

        for rule in ("confere_por_sequencia", "confere_overall"):
            for info in candidates_info:
                if chosen is None and info.get(rule):
                    chosen = dict(info, criterio=rule)
        if chosen is None:
            complete = [i for i in candidates_info if i["arquivos"] == i["esperados"]]
            if complete:
                chosen = dict(complete[0], criterio="nao_reproduz_publicado" if "overall" in complete[0]
                              else "nao_verificado")
                if chosen["criterio"] == "nao_reproduz_publicado":
                    rep.warn(f"{tracker}: nenhum candidato reproduz o artigo; usando '{chosen['sufixo']}' "
                             "(registrado em ASSUMPTIONS.md A-02; o resumo informa as metricas efetivas)")
                else:
                    rep.warn(f"{tracker}: selecao sem verificacao de metricas (--rapido)")
            else:
                chosen = {"sufixo": None, "criterio": "gerar"}
                rep.warn(f"{tracker}: nenhuma trajetoria completa; a etapa 2 vai gera-las com a configuracao "
                         "do artigo (motpred/runners.py)")
        else:
            rep.ok(f"{tracker}: selecionado '{chosen['sufixo']}' (criterio: {chosen['criterio']})")
        selection["rastreadores"][tracker] = {
            "selecionado": chosen.get("sufixo"), "criterio": chosen.get("criterio"),
            "overall": chosen.get("overall"), "candidatos": candidates_info,
        }
    write_json(sel_path, selection)
    rep.info(f"selecao gravada em {sel_path}")
    return selection


# ---------------------------------------------------------------------------
# 8-9. SIGE e DUT
# ---------------------------------------------------------------------------

def check_sige(cfg, rep: Report) -> dict:
    rep.section("8. KALMAN E PROTOCOLO DO SIGE (uav_common.py)")
    uc = uav_common(cfg)
    rep.ok(f"uav_common importado de {Path(uc.__file__).resolve()}")
    horizons = tuple(int(h) for h in cfg["prediction"]["horizons"])
    if tuple(uc.HORIZONS) == horizons:
        rep.ok(f"horizontes do config = uav_common.HORIZONS = {uc.HORIZONS}")
    else:
        rep.fail(f"horizontes do config {horizons} diferem de uav_common.HORIZONS {uc.HORIZONS}")
    kalman = uc.init_kalman(100.0, 50.0)
    q = np.diag(kalman.processNoiseCov).tolist()
    r = np.diag(kalman.measurementNoiseCov).tolist()
    p0 = np.diag(kalman.errorCovPost).tolist()
    rep.ok(f"init_kalman: Q=diag{[round(v, 4) for v in q]}  R=diag{[round(v, 4) for v in r]}  "
           f"P0=diag{[round(v, 4) for v in p0]}")
    rep.info(f"F = {kalman.transitionMatrix.tolist()}")
    rep.info(f"H = {kalman.measurementMatrix.tolist()}")
    rep.info("forward(): amortecimento aplicado a cada passo da predicao (uav_common.py)")
    return {"Q": q, "R": r, "P0": p0}


def check_dut(cfg, rep: Report) -> dict:
    rep.section("9. REFERENCIA VISIVEL PARA H4 (DUT Anti-UAV)")
    from motpred.dut_sige import dut_configured

    enabled, reason = dut_configured(cfg)
    weights = rpath(cfg["weights"]["sige_dut"])
    if weights and weights.exists():
        rep.ok(f"pesos do SIGE: {weights}")
    else:
        rep.warn(f"pesos do SIGE nao encontrados: {weights}")
    if not enabled:
        rep.warn(f"H4 sem referencia visivel: {reason}. Sem ela, H4 sera declarada inconclusiva.")
        return {"habilitado": False, "motivo": reason}
    from motpred.dut_sige import find_dut_ground_truth

    uc = uav_common(cfg)
    images = rpath(cfg["dut"]["images_dir"])
    videos = uc.list_videos(images)
    frames = sum(len(uc.frames_of_video(images, v)) for v in videos)
    rep.ok(f"{len(videos)} sequencias, {frames} quadros em {images}")
    if len(videos) != 20 or frames != 24804:
        rep.warn("o SIGE usou 20 sequencias e 24.804 quadros")
    kind, source = find_dut_ground_truth(cfg, uc)
    if kind == "oficial":
        rep.ok(f"GT oficial: {len(source)} arquivos videoNN_gt.txt")
        missing = [v for v in videos if v not in source]
    elif kind == "rotulos_yolo":
        n_labels = sum(1 for _ in source.glob("video*_*.txt"))
        rep.ok(f"GT pelos rotulos YOLO do PASSO 2 do SIGE: {n_labels} arquivos em {source} "
               "(alternativa aceita pelo 05_avaliar.py; convertidos para pixels com a resolucao de cada sequencia)")
        missing = [v for v in videos if next(source.glob(f"{v}_*.txt"), None) is None]
    else:
        rep.warn(f"H4 sem ground truth do DUT: {source}. Sem ele, H4 sera declarada inconclusiva.")
        return {"habilitado": False, "motivo": source}
    if missing:
        rep.warn(f"sem GT para: {', '.join(missing)}")
    return {"habilitado": True, "sequencias": len(videos), "quadros": frames, "fonte_gt": kind}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--rapido", action="store_true", help="nao reavalia as metricas das trajetorias")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(int(cfg["seed"]))
    rep = Report()
    rep.add(f"check_setup.py  --  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rep.add(f"raiz do projeto: {ROOT}")
    rep.add(f"config: {cfg['_config_path']} (sha256 {cfg['_config_sha256'][:16]}...)")

    status = {}
    sections = [
        ("ambiente", lambda: check_environment(cfg, rep)),
        ("caminhos", lambda: check_paths(cfg, rep)),
    ]
    for name, fn in sections:
        try:
            status[name] = fn()
        except SystemExit as error:
            rep.fail(f"{name}: {error}")
        except Exception as error:  # noqa: BLE001
            rep.fail(f"{name}: {type(error).__name__}: {error}")

    seqs = []
    try:
        seqs = check_sequences(cfg, rep)
    except Exception as error:  # noqa: BLE001
        rep.fail(f"sequencias: {type(error).__name__}: {error}")

    selection = {}
    if seqs:
        try:
            selection = check_trajectories(cfg, rep, seqs, args.rapido)
        except Exception as error:  # noqa: BLE001
            rep.fail(f"trajetorias: {type(error).__name__}: {error}")
        baseline_suffix = (selection.get("rastreadores", {}).get("baseline", {}) or {}).get("selecionado")
        try:
            status["subamostragem"] = check_subsampling(cfg, rep, seqs, baseline_suffix)
            status["taxas"] = check_frame_rates(cfg, rep, status["subamostragem"])
        except Exception as error:  # noqa: BLE001
            rep.fail(f"subamostragem/taxa de quadros: {type(error).__name__}: {error}")

    for name, fn in (("botsort", lambda: check_botsort(cfg, rep)), ("sige", lambda: check_sige(cfg, rep)),
                     ("dut", lambda: check_dut(cfg, rep))):
        try:
            status[name] = fn()
        except Exception as error:  # noqa: BLE001
            rep.fail(f"{name}: {type(error).__name__}: {error}")

    rep.section("RESUMO")
    rep.add(f"  problemas BLOQUEANTES: {len(rep.blocking)}")
    for item in rep.blocking:
        rep.add(f"    - {item}")
    rep.add(f"  avisos: {len(rep.warnings)}")
    for item in rep.warnings:
        rep.add(f"    - {item}")

    text = "\n".join(rep.lines) + "\n"
    (MP_DIR / "check_setup.txt").write_text(text, encoding="utf-8")
    dirs = out_dirs(cfg)
    dirs["setup"].mkdir(parents=True, exist_ok=True)
    (dirs["setup"] / "check_setup.txt").write_text(text, encoding="utf-8")
    env = status.get("ambiente") or environment_info()
    sub = status.get("subamostragem", {})
    write_json(dirs["setup"] / "setup_status.json", {
        "ambiente": env,
        "config_sha256": cfg["_config_sha256"],
        "sem_subamostragem": sub.get("sem_subamostragem"),
        "quadros_total": sub.get("quadros_total"),
        "taxas": status.get("taxas"),
        "botsort": status.get("botsort"),
        "kalman_sige": status.get("sige"),
        "dut": status.get("dut"),
        "bloqueantes": rep.blocking,
        "avisos": rep.warnings,
    })
    return 1 if rep.blocking else 0


if __name__ == "__main__":
    sys.exit(main())
