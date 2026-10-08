#!/usr/bin/env python3
"""ETAPA 4 - Varreduras opcionais do Experimento B.

Cada variante habilitada em config.yaml (sweeps.variants) herda a secao 'hybrid'
e sobrescreve so o que declara (N_red, rastreador local, pesos, SAHI, ciclo de
vida). Depois, se sweeps.hit_reference_timing = true, o HIT-Tracker com tempo por
quadro (referencia de custo com deteccao em todo quadro).

Retomavel por variante e por sequencia.

    python mot_prediction/steps/s4_sweeps.py
    python mot_prediction/steps/s4_sweeps.py --variant hybrid_kcf_n15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motpred.common import (deep_merge, is_done, load_config, log, mark_done, out_dirs,  # noqa: E402
                            selected_trajectories, set_seeds)
from motpred.data import val_sequences  # noqa: E402
from motpred.pipeline import process_hit_ref, process_hybrid, published_traj, stage_marker  # noqa: E402
from motpred.runners import HitReference, HybridFactory  # noqa: E402


def variant_config(cfg: dict, variant: dict) -> dict:
    override = {k: v for k, v in variant.items() if k not in ("enabled",)}
    return deep_merge(cfg["hybrid"], override)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--variant", nargs="*", default=None, help="so estas variantes (nomes do config)")
    parser.add_argument("--only-seq", nargs="*", default=None)
    parser.add_argument("--sem-hit", action="store_true", help="nao roda a referencia do HIT-Tracker")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(int(cfg["seed"]))
    selection = selected_trajectories(cfg)
    dirs = out_dirs(cfg)
    base_b, base_a = dirs["exp_b"], dirs["exp_a"]
    seqs = val_sequences(cfg)
    if args.only_seq:
        seqs = [s for s in seqs if s in set(args.only_seq)]

    variants = [v for v in cfg["sweeps"]["variants"] if v.get("enabled", False)]
    if args.variant:
        variants = [v for v in cfg["sweeps"]["variants"] if v["name"] in set(args.variant)]
    for variant in variants:
        hcfg = variant_config(cfg, variant)
        marker = stage_marker(base_b, hcfg["name"])
        if is_done(marker) and not args.only_seq:
            log(f"Variante {hcfg['name']} ja concluida (pulando).")
            continue
        log(f"Variante {hcfg['name']}: N_red={hcfg['n_red']}, {hcfg['local_tracker'].upper()}, "
            f"pesos '{hcfg['detector_weights']}', SAHI={hcfg['sahi']['enabled']}")
        factory = HybridFactory(cfg, hcfg)
        start = time.perf_counter()
        for number, seq in enumerate(seqs, start=1):
            ref = published_traj(cfg, selection, "baseline", seq, base_a)
            info = process_hybrid(cfg, factory, seq, base_b, ref)
            text = "ja processada (pulando)" if info is None else (
                f"{info['quadros']} quadros, {info['fps']:.1f} FPS, {info['tempo_total_s']:.0f} s")
            log(f"[{number}/{len(seqs)}] {hcfg['name']} {seq}: {text}")
        if not args.only_seq:
            mark_done(marker, {"sequencias": len(seqs), "tempo_s": time.perf_counter() - start})

    if cfg["sweeps"].get("hit_reference_timing") and not args.sem_hit and not args.variant:
        rcfg = cfg["reference"]["hit"]
        marker = stage_marker(base_b, rcfg["name"])
        if is_done(marker) and not args.only_seq:
            log("Referencia do HIT-Tracker ja concluida (pulando).")
        else:
            start = time.perf_counter()
            with HitReference(cfg, rcfg) as hit:
                for number, seq in enumerate(seqs, start=1):
                    info = process_hit_ref(cfg, hit, seq, base_b)
                    text = "ja processada (pulando)" if info is None else (
                        f"{info['quadros']} quadros, {info['fps']:.1f} FPS")
                    log(f"[{number}/{len(seqs)}] HIT-Tracker (todo quadro) {seq}: {text}")
            if not args.only_seq:
                mark_done(marker, {"sequencias": len(seqs), "tempo_s": time.perf_counter() - start})
    return 0


if __name__ == "__main__":
    sys.exit(main())
