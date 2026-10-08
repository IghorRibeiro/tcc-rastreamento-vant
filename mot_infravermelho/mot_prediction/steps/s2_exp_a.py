#!/usr/bin/env python3
"""ETAPA 2 - Experimento A: predicao sobre as trajetorias MOT ja salvas.

Para cada uma das 30 sequencias: linha de base e HIT-Tracker (arquivos escolhidos
pelo check_setup), metricas MOT, Kalman do SIGE por trilha (persistente e
amortecido), erro com identidade (pi_CLEAR e pi_ID) e suporte comum.
Retomavel: uma sequencia com results/exp_a/done/<seq>.json e pulada.

    python mot_prediction/steps/s2_exp_a.py
    python mot_prediction/steps/s2_exp_a.py --only-seq UAV005 UAV007
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motpred.common import (is_done, load_config, log, mark_done, out_dirs, selected_trajectories,  # noqa: E402
                            set_seeds)
from motpred.data import val_sequences  # noqa: E402
from motpred.pipeline import process_expa_sequence, stage_marker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--only-seq", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(int(cfg["seed"]))
    selection = selected_trajectories(cfg)
    base = out_dirs(cfg)["exp_a"]
    marker = stage_marker(base, "exp_a")
    if is_done(marker) and not args.only_seq:
        log(f"Experimento A ja concluido ({marker}); nada a fazer.")
        return 0

    seqs = val_sequences(cfg)
    if args.only_seq:
        seqs = [s for s in seqs if s in set(args.only_seq)]
    for tracker in ("baseline", "hit"):
        entry = selection["rastreadores"][tracker]
        log(f"{tracker}: trajetorias '{entry['selecionado']}' (criterio {entry['criterio']})")

    start = time.perf_counter()
    for number, seq in enumerate(seqs, start=1):
        info = process_expa_sequence(cfg, selection, seq, base)
        if info is None:
            log(f"[{number}/{len(seqs)}] {seq}: ja processada (pulando)")
            continue
        b, h = info["baseline"], info["hit"]
        log(f"[{number}/{len(seqs)}] {seq}: H30 persistente base={b['H30_px_persistente']:.1f}px "
            f"(IDF1 {b['IDF1'] * 100:.1f}%)  HIT={h['H30_px_persistente']:.1f}px (IDF1 {h['IDF1'] * 100:.1f}%)  "
            f"[{info['tempo_s']:.0f} s]")

    if not args.only_seq:
        mark_done(marker, {"sequencias": len(seqs), "tempo_s": time.perf_counter() - start})
        log(f"Experimento A concluido: {marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
