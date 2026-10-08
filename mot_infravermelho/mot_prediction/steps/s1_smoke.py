#!/usr/bin/env python3
"""ETAPA 1 - Teste rapido em UMA sequencia (config: smoke.sequence) + estimativa de tempo.

Verificacoes (as marcadas [B] interrompem a noite se falharem):
  [B] fidelidade do Kalman: a reimplementacao por trilha reproduz
      uav_common.prediction_errors bit a bit (alvo unico do GT e uma trilha real com lacunas);
  [B] o avaliador reproduz as metricas publicadas da sequencia (quando o check_setup
      selecionou trajetorias que reproduzem o artigo);
      o IDF1 reconstruido pelo mapeamento global confere com o py-motmetrics (aviso);
  Execucao cronometrada de cada peca da noite na sequencia de teste:
      Experimento A; linha de base (todo quadro); hibrido principal; KCF e MOSSE
      (primeiros quadros); HIT-Tracker (primeiros quadros); DUT (uma sequencia, se configurado).
No fim, extrapola o tempo das etapas 2, 3 e 4 para as 30 sequencias.
Saidas em results/smoke/ (nao se misturam com os resultados da noite).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from motpred.common import (deep_merge, load_config, log, mark_done, out_dirs, read_json, rpath,  # noqa: E402
                            selected_trajectories, set_seeds, sige_rastreadores, uav_common, write_json)
from motpred.data import load_gt, load_mot, val_sequences  # noqa: E402
from motpred.dut_sige import dut_configured  # noqa: E402
from motpred.mot_eval import evaluate_mot_sequence  # noqa: E402
from motpred.pipeline import (DutContext, process_baseline_ref, process_dut_video, process_expa_sequence,  # noqa: E402
                              process_hit_ref, process_hybrid, published_traj)
from motpred.prediction import fidelity_check, track_centers  # noqa: E402
from motpred.runners import HitReference, HybridFactory, variant_paths  # noqa: E402


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name: str, ok: bool, blocking: bool, detail: str) -> None:
        self.items.append({"verificacao": name, "ok": bool(ok), "bloqueante": bool(blocking), "detalhe": detail})
        tag = "ok" if ok else ("ERRO" if blocking else "aviso")
        log(f"[{tag}] {name}: {detail}")

    @property
    def blocking_failures(self):
        return [i for i in self.items if i["bloqueante"] and not i["ok"]]


def fit_time_model(frames: pd.DataFrame) -> dict:
    """tempo_ms = b*ativos + c*deteccao*ativos + d*deteccao (sem intercepto).

    b: atualizacao do rastreador local por trilha; c: reinicializacao por trilha nos
    quadros de redeteccao; d: a chamada do detector. Sem intercepto porque numa unica
    sequencia o numero de trilhas quase nao varia e o intercepto ficaria confundido com b.
    """
    df = frames.dropna(subset=["time_ms", "n_active", "detect"])
    active = df["n_active"].to_numpy(float)
    detect = df["detect"].to_numpy(float)
    if len(df) < 10 or active.sum() == 0:
        return {"b": float(frames["time_ms"].mean()) / max(1.0, float(frames["n_active"].mean())), "c": 0.0, "d": 0.0}
    X = np.column_stack([active, detect * active, detect])
    coef, *_ = np.linalg.lstsq(X, df["time_ms"].to_numpy(float), rcond=None)
    return {"b": float(coef[0]), "c": float(coef[1]), "d": float(coef[2])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(int(cfg["seed"]))
    dirs = out_dirs(cfg)
    smoke = dirs["smoke"]
    seq = str(cfg["smoke"]["sequence"])
    selection = selected_trajectories(cfg)
    uc = uav_common(cfg)
    horizons = tuple(int(h) for h in cfg["prediction"]["horizons"])
    dampings = tuple(float(d) for d in cfg["prediction"]["dampings"])
    checks = Checks()
    timing = {}
    seqs = val_sequences(cfg)
    if seq not in seqs:
        raise SystemExit(f"smoke.sequence = {seq} nao esta entre as sequencias de validacao")
    gt_df, gt_index = load_gt(cfg, seq)
    log(f"Teste rapido na sequencia {seq}: densidade media {gt_index.mean_density:.1f} alvos/quadro")

    # ------------------------------------------------ 1. fidelidade do Kalman
    g_long = max(gt_index.by_id, key=lambda g: len(gt_index.by_id[g]))
    gt_centers = {f: (c[0], c[1]) for f, c in gt_index.by_id[g_long].items()}
    fid = fidelity_check(uc, gt_centers, gt_centers, horizons, dampings)
    checks.add("Kalman identico ao SIGE (trajetoria do GT)", fid["igual"], True,
               f"{fid['n_valores']} erros comparados, maior diferenca {fid['max_dif_abs']:.3g}")

    base_path = published_traj(cfg, selection, "baseline", seq, dirs["exp_a"])
    if base_path.exists():
        pred_df = load_mot(cfg, base_path)
        _, corr = evaluate_mot_sequence(gt_df, pred_df, seq)
        tracks = track_centers(pred_df)
        mapped = [h for h in tracks if h in corr["idf1"]]
        if mapped:
            h_long = max(mapped, key=lambda h: len(tracks[h]))
            g = corr["idf1"][h_long]
            target = {f: (c[0], c[1]) for f, c in gt_index.by_id[g].items()}
            fid = fidelity_check(uc, tracks[h_long], target, horizons, dampings)
            frames = sorted(tracks[h_long])
            gaps = sum(1 for a, b in zip(frames[:-1], frames[1:]) if b - a > 1)
            checks.add("Kalman identico ao SIGE (trilha real da linha de base, com lacunas)", fid["igual"], True,
                       f"trilha {h_long} ({len(frames)} quadros, {gaps} lacunas), {fid['n_valores']} erros, "
                       f"maior diferenca {fid['max_dif_abs']:.3g}")

    # ------------------------------------------------ 2. avaliador x publicado
    for tracker in ("baseline", "hit"):
        entry = selection["rastreadores"][tracker]
        path = published_traj(cfg, selection, tracker, seq, dirs["exp_a"])
        if not path.exists():
            checks.add(f"trajetoria {tracker}", False, False, f"{path} ausente (sera gerada na etapa 2)")
            continue
        metrics, _ = evaluate_mot_sequence(gt_df, load_mot(cfg, path), seq)
        checks.add(f"IDF1 reconstruido = py-motmetrics ({tracker})", metrics["idf1_confere"], False,
                   f"{metrics['idf1_reconstruido']:.6f} x {metrics['idf1']:.6f}")
        pub_path = rpath(cfg["trajectories"][tracker]["published_summary"])
        if pub_path and pub_path.exists():
            pub = pd.read_csv(pub_path, index_col=0)
            if seq in pub.index:
                ref = pub.loc[seq]
                same = (abs(metrics["mota"] - float(ref["mota"])) < 1e-6
                        and abs(metrics["idf1"] - float(ref["idf1"])) < 1e-6
                        and int(metrics["num_switches"]) == int(ref["num_switches"]))
                blocking = entry["criterio"] == "confere_por_sequencia"
                checks.add(f"avaliador reproduz o publicado ({tracker}, {seq})", same, blocking,
                           f"MOTA {metrics['mota']:.4f} x {float(ref['mota']):.4f}; IDF1 {metrics['idf1']:.4f} x "
                           f"{float(ref['idf1']):.4f}; IDs {int(metrics['num_switches'])} x {int(ref['num_switches'])}")

    # ------------------------------------------------ 3. Experimento A na sequencia
    t0 = time.perf_counter()
    info_a = process_expa_sequence(cfg, selection, seq, smoke / "exp_a")
    timing["exp_a_seq_s"] = time.perf_counter() - t0 if info_a is not None else read_json(
        smoke / "exp_a" / "done" / f"{seq}.json").get("tempo_s", math.nan)
    log(f"Experimento A em {seq}: {timing['exp_a_seq_s']:.1f} s")

    # ------------------------------------------------ 4-5. linha de base e hibrido
    base_b = smoke / "exp_b"
    process_baseline_ref(cfg, seq, base_b)
    ref_frames = pd.read_csv(variant_paths(base_b, cfg["reference"]["baseline"]["name"], seq)["frames"])
    timing["baseline_ms_quadro"] = float(ref_frames["time_ms"].mean())
    done_ref = read_json(variant_paths(base_b, cfg["reference"]["baseline"]["name"], seq)["done"])
    timing["baseline_eval_s"] = float(done_ref.get("tempo_avaliacao_s", 0.0))
    log(f"linha de base (todo quadro): {timing['baseline_ms_quadro']:.1f} ms/quadro "
        f"({1000 / timing['baseline_ms_quadro']:.1f} FPS)")

    hcfg = cfg["hybrid"]
    factory = HybridFactory(cfg, hcfg)
    process_hybrid(cfg, factory, seq, base_b, base_path if base_path.exists() else None)
    hp = variant_paths(base_b, hcfg["name"], seq)
    hyb_frames = pd.read_csv(hp["frames"])
    done_h = read_json(hp["done"])
    timing["hibrido_ms_quadro"] = float(hyb_frames["time_ms"].mean())
    timing["hibrido_eval_s"] = float(done_h.get("tempo_avaliacao_s", 0.0))
    timing["hibrido_frac_deteccao"] = float(hyb_frames["detect"].mean())
    timing["hibrido_ativos_por_gt"] = float(hyb_frames["n_active"].mean() / max(1e-9, hyb_frames["n_gt"].mean()))
    model = fit_time_model(hyb_frames)
    timing["hibrido_modelo_tempo"] = model
    hyb_mot = read_json(hp["eval_dir"] / f"{seq}_mot.json")
    log(f"hibrido: {timing['hibrido_ms_quadro']:.1f} ms/quadro ({1000 / timing['hibrido_ms_quadro']:.1f} FPS), "
        f"MOTA {hyb_mot['mot_mota'] * 100:.1f}%, IDF1 {hyb_mot['mot_idf1'] * 100:.1f}%, "
        f"deteccao em {timing['hibrido_frac_deteccao'] * 100:.1f}% dos quadros; "
        f"custo: {model['b']:.1f} ms por trilha por quadro + {model['c']:.1f} ms por trilha reancorada "
        f"+ {model['d']:.1f} ms por chamada do detector")
    exc = int(hyb_frames["exceptions"].sum())
    checks.add("hibrido sem excecoes do OpenCV", exc == 0, False,
               f"{exc} excecoes do rastreador local (tratadas como falha)")
    checks.add("hibrido produziu trilhas", hyb_mot["mot_num_predictions"] > 0, True,
               f"{int(hyb_mot['mot_num_predictions'])} caixas, {done_h.get('trilhas_criadas')} trilhas")

    # ------------------------------------------------ 6. KCF e MOSSE (primeiros quadros)
    n_sweep = int(cfg["smoke"]["sweep_frames"])
    csrt_ref = float(hyb_frames["time_ms"].iloc[:n_sweep].mean())
    sige04 = sige_rastreadores(cfg)
    for local in ("kcf", "mosse"):
        try:
            sige04.make_local_tracker(local)
        except Exception as error:  # noqa: BLE001
            checks.add(f"rastreador {local.upper()} disponivel", False, False, str(error).splitlines()[0])
            continue
        vcfg = deep_merge(hcfg, {"name": f"smoke_{local}", "local_tracker": local})
        fac = HybridFactory(cfg, vcfg)
        process_hybrid(cfg, fac, seq, base_b, None, max_frames=n_sweep, evaluate=False)
        frames = pd.read_csv(variant_paths(base_b, vcfg["name"], seq)["frames"])
        timing[f"razao_{local}_csrt"] = float(frames["time_ms"].mean()) / csrt_ref if csrt_ref else math.nan
        log(f"{local.upper()}: {frames['time_ms'].mean():.1f} ms/quadro nos {n_sweep} primeiros quadros "
            f"(CSRT: {csrt_ref:.1f})")

    # ------------------------------------------------ 7. HIT-Tracker (primeiros quadros)
    n_hit = int(cfg["smoke"]["hit_frames"])
    try:
        with HitReference(cfg, cfg["reference"]["hit"]) as hit:
            process_hit_ref(cfg, hit, seq, base_b, max_frames=n_hit, evaluate=False)
        frames = pd.read_csv(variant_paths(base_b, cfg["reference"]["hit"]["name"], seq)["frames"])
        timing["hit_ms_quadro"] = float(frames["time_ms"].mean())
        log(f"HIT-Tracker: {timing['hit_ms_quadro']:.1f} ms/quadro nos {n_hit} primeiros quadros")
        checks.add("HIT-Tracker (SAHI + GM + NSA) executa", True, False, "ok")
    except Exception as error:  # noqa: BLE001
        checks.add("HIT-Tracker (SAHI + GM + NSA) executa", False, False, f"{type(error).__name__}: {error}")

    # ------------------------------------------------ 8. DUT (uma sequencia)
    enabled, reason = dut_configured(cfg)
    if enabled:
        try:
            ctx = DutContext(cfg)
            video = str(cfg["smoke"]["dut_video"])
            if video not in ctx.videos():
                video = ctx.videos()[0]
            t0 = time.perf_counter()
            record = process_dut_video(cfg, ctx, video, smoke / "h4_dut")
            if record is None:
                record = read_json(smoke / "h4_dut" / "eval" / f"{video}_sige.json")
            timing["dut_ms_quadro"] = float(record["mean_time_ms"])
            timing["dut_seq_s"] = time.perf_counter() - t0
            timing["dut_quadros_total"] = int(sum(len(ctx.uc.frames_of_video(ctx.images, v)) for v in ctx.videos()))
            checks.add("hibrido do SIGE no DUT executa", True, False,
                       f"{video} (GT: {ctx.gt_kind}): SR@0.5={record.get('sige_SR@0.5', float('nan')):.3f}, "
                       f"H30 persistente={record.get('g1.00_H30_media', float('nan')):.1f}px, "
                       f"{record['fps']:.1f} FPS")
        except (Exception, SystemExit) as error:  # noqa: BLE001 - H4 e secundaria: nao derruba o teste
            checks.add("hibrido do SIGE no DUT executa", False, False, f"{type(error).__name__}: {error}")
    else:
        checks.add("referencia visivel de H4 (DUT)", False, False, reason)

    # ------------------------------------------------ 9. estimativa
    totals = {"quadros": 0, "gt_objetos": 0}
    sum_gt_per_seq = {}
    for s in seqs:
        _, idx = load_gt(cfg, s)
        n_frames = max(idx.last_frame, 1)
        sum_gt = sum(idx.count_by_frame.values())
        sum_gt_per_seq[s] = (n_frames, sum_gt)
        totals["quadros"] += n_frames
        totals["gt_objetos"] += sum_gt
    smoke_gt = max(1, sum(gt_index.count_by_frame.values()))
    ratio_active = timing["hibrido_ativos_por_gt"]

    def hybrid_minutes(n_red: int, factor: float = 1.0) -> float:
        p_det = min(1.0, timing["hibrido_frac_deteccao"] * float(hcfg["n_red"]) / float(n_red))
        total_ms = 0.0
        for n_frames, sum_gt in sum_gt_per_seq.values():
            total_ms += ((model["b"] + p_det * model["c"]) * ratio_active * sum_gt
                         + p_det * model["d"] * n_frames)
        eval_s = timing["hibrido_eval_s"] * totals["gt_objetos"] / smoke_gt
        return (factor * total_ms / 1000.0 + eval_s) / 60.0

    est = {}
    est["etapa2_experimento_A"] = timing["exp_a_seq_s"] * totals["gt_objetos"] / smoke_gt / 60.0
    est["etapa3_linha_de_base_todo_quadro"] = (timing["baseline_ms_quadro"] * totals["quadros"] / 1000.0
                                               + timing["baseline_eval_s"] * totals["gt_objetos"] / smoke_gt) / 60.0
    est["etapa3_hibrido_principal"] = hybrid_minutes(int(hcfg["n_red"]))
    if "dut_ms_quadro" in timing:
        est["etapa3_dut_H4"] = timing["dut_ms_quadro"] * timing["dut_quadros_total"] / 1000.0 / 60.0 * 1.15
    for variant in cfg["sweeps"]["variants"]:
        if not variant.get("enabled"):
            continue
        vcfg = deep_merge(hcfg, {k: v for k, v in variant.items() if k != "enabled"})
        factor = 1.0
        if vcfg["local_tracker"] in ("kcf", "mosse"):
            factor = timing.get(f"razao_{vcfg['local_tracker']}_csrt", 1.0)
        if vcfg.get("sahi", {}).get("enabled"):
            factor *= 1.5
        est[f"etapa4_{vcfg['name']}"] = hybrid_minutes(int(vcfg["n_red"]), factor)
    if cfg["sweeps"].get("hit_reference_timing") and "hit_ms_quadro" in timing:
        est["etapa4_hit_todo_quadro"] = (timing["hit_ms_quadro"] * totals["quadros"] / 1000.0
                                         + timing["baseline_eval_s"] * totals["gt_objetos"] / smoke_gt) / 60.0
    est = {k: round(float(v), 1) for k, v in est.items()}
    main_total = sum(v for k, v in est.items() if not k.startswith("etapa4"))
    sweep_total = sum(v for k, v in est.items() if k.startswith("etapa4"))
    finish_main = datetime.now() + timedelta(minutes=main_total + 2)
    finish_all = datetime.now() + timedelta(minutes=main_total + sweep_total + 2)

    print()
    print("=" * 78)
    print("ESTIMATIVA DE TEMPO DAS PROXIMAS ETAPAS (30 sequencias, extrapolado desta maquina)")
    print("=" * 78)
    for key, minutes in est.items():
        print(f"  {key:<45s} {minutes:8.1f} min")
    print("-" * 78)
    print(f"  etapas 2 e 3 (sem varreduras)                 {main_total:8.1f} min  "
          f"(~{main_total / 60:.1f} h; termino previsto {finish_main:%d/%m %H:%M} se iniciar agora)")
    print(f"  etapa 4 (varreduras)                          {sweep_total:8.1f} min")
    print(f"  total                                         {main_total + sweep_total:8.1f} min  "
          f"(~{(main_total + sweep_total) / 60:.1f} h; termino previsto {finish_all:%d/%m %H:%M})")
    print("  Obs.: extrapolacao linear pela densidade de cada sequencia; o calor da maquina e o")
    print("  CSRT em enxames densos podem aumentar o tempo real em 20-40%.")
    print("=" * 78)

    write_json(smoke / "estimate.json", {"minutos": est, "total_sem_varreduras_min": main_total,
                                         "total_varreduras_min": sweep_total, "totais": totals,
                                         "tempos_medidos": timing})
    write_json(smoke / "checks.json", {"sequencia": seq, "verificacoes": checks.items})
    lines = [f"# Teste rapido ({seq})", "", "| Verificacao | Resultado | Bloqueante | Detalhe |", "|---|---|---|---|"]
    for item in checks.items:
        lines.append(f"| {item['verificacao']} | {'ok' if item['ok'] else 'FALHOU'} | "
                     f"{'sim' if item['bloqueante'] else 'nao'} | {item['detalhe']} |")
    lines += ["", "## Estimativa (min)", "", "| Parte | Minutos |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in est.items()]
    lines += [f"| **etapas 2-3** | **{main_total:.1f}** |", f"| **etapa 4** | **{sweep_total:.1f}** |"]
    (smoke / "smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failures = checks.blocking_failures
    if failures:
        log("TESTE RAPIDO REPROVADO: " + "; ".join(f"{f['verificacao']} ({f['detalhe']})" for f in failures))
        return 1
    mark_done(smoke / "_DONE_smoke.json", {"sequencia": seq, "estimativa_min": est})
    log("Teste rapido aprovado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
