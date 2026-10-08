#!/usr/bin/env python3
"""ETAPA 3 - Experimento B principal + referencia visivel de H4.

Parte 1 (por sequencia, intercalado para que as duas medicoes de tempo vejam a
maquina no mesmo estado):
    a) linha de base com deteccao em todo quadro (YOLO26n + BoT-SORT), tempo por quadro;
    b) hibrido MOT, N_red = 15, CSRT, custo GM do HIT-Tracker; avaliacao MOT,
       erro de predicao com identidade, suporte comum com a linha de base
       publicada e deriva do CSRT (H4).
Parte 2: hibrido do SIGE instrumentado no DUT Anti-UAV (so se dut.* estiver no config).

Retomavel por sequencia (results/exp_b/done/<variante>/<seq>.json).

    python mot_prediction/steps/s3_exp_b.py
    python mot_prediction/steps/s3_exp_b.py --only-seq UAV005 --sem-dut
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motpred.common import (is_done, load_config, log, mark_done, out_dirs, selected_trajectories,  # noqa: E402
                            set_seeds, write_json)
from motpred.data import val_sequences  # noqa: E402
from motpred.dut_sige import dut_configured  # noqa: E402
from motpred.pipeline import (DutContext, process_baseline_ref, process_dut_video, process_hybrid,  # noqa: E402
                              published_traj, stage_marker)
from motpred.runners import HybridFactory  # noqa: E402


def _fmt(info: dict | None) -> str:
    if info is None:
        return "ja processada (pulando)"
    return f"{info['quadros']} quadros, {info['fps']:.1f} FPS, {info['tempo_total_s']:.0f} s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--only-seq", nargs="*", default=None)
    parser.add_argument("--sem-dut", action="store_true", help="nao roda a parte 2 (DUT)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(int(cfg["seed"]))
    selection = selected_trajectories(cfg)
    dirs = out_dirs(cfg)
    base_b, base_a, base_h4 = dirs["exp_b"], dirs["exp_a"], dirs["h4_dut"]
    hcfg = cfg["hybrid"]
    main_marker = stage_marker(base_b, hcfg["name"])

    seqs = val_sequences(cfg)
    if args.only_seq:
        seqs = [s for s in seqs if s in set(args.only_seq)]

    # ------------------------------------------------------------ parte 1
    if is_done(main_marker) and not args.only_seq:
        log(f"Experimento B principal ja concluido ({main_marker}).")
    else:
        log(f"Hibrido '{hcfg['name']}': N_red={hcfg['n_red']}, {hcfg['local_tracker'].upper()}, "
            f"pesos '{hcfg['detector_weights']}', gamma GM={hcfg['association']['gamma']}")
        factory = HybridFactory(cfg, hcfg)
        log(f"detector: {factory.weights}  dispositivo: {factory.device}")
        start = time.perf_counter()
        for number, seq in enumerate(seqs, start=1):
            info = process_baseline_ref(cfg, seq, base_b)
            log(f"[{number}/{len(seqs)}] {seq} linha de base (todo quadro): {_fmt(info)}")
            ref = published_traj(cfg, selection, "baseline", seq, base_a)
            info = process_hybrid(cfg, factory, seq, base_b, ref)
            log(f"[{number}/{len(seqs)}] {seq} hibrido: {_fmt(info)}")
        if not args.only_seq:
            mark_done(main_marker, {"sequencias": len(seqs), "tempo_s": time.perf_counter() - start})

    # ------------------------------------------------------------ parte 2
    h4_marker = stage_marker(base_h4, "dut")
    if args.sem_dut:
        log("Parte 2 (DUT) pulada por --sem-dut.")
        return 0
    enabled, reason = dut_configured(cfg)
    if not enabled:
        log(f"Referencia visivel de H4 NAO executada: {reason}.")
        write_json(base_h4 / "_SKIPPED.json", {"motivo": reason})
        return 0
    if is_done(h4_marker):
        log(f"Referencia visivel de H4 ja concluida ({h4_marker}).")
        return 0
    ctx = DutContext(cfg)
    videos = ctx.videos()
    start = time.perf_counter()
    for number, video in enumerate(videos, start=1):
        record = process_dut_video(cfg, ctx, video, base_h4)
        if record is None:
            log(f"[{number}/{len(videos)}] DUT {video}: ja processado (pulando)")
        else:
            log(f"[{number}/{len(videos)}] DUT {video}: {record['frames']} quadros, {record['fps']:.1f} FPS, "
                f"SR@0.5={record.get('sige_SR@0.5', float('nan')):.3f}")
    mark_done(h4_marker, {"videos": len(videos), "tempo_s": time.perf_counter() - start})
    return 0


if __name__ == "__main__":
    sys.exit(main())
