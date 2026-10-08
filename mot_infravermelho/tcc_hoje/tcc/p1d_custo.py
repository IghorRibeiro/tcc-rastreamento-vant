"""Parte 1d -- custo x densidade com os tempos por quadro de 28/09 (sem medir FPS de novo).

tempo_ms = a + b * n_alvos (MQO com todos os quadros da variante; n_alvos = n_gt do quadro).
IC das inclinacoes e da diferenca hibrido - linha de base: bootstrap por sequencia
(o mesmo sorteio de sequencias para todas as variantes = pareado), BCa, 10.000.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import base, estat


def main() -> int:
    smoke = base.smoke_flag()
    cfg = base.load_cfg()
    out = base.sub(smoke, "parte1")
    exp_b = base.P(cfg["paths"]["exp_b"])
    seqs = base.val_sequences(smoke)
    variants = list(cfg["mot"]["variantes"])

    per_seq, suf, bins, faltas = [], [], [], []
    S = {v: [] for v in variants}
    common = []
    frames_by = {}
    for v in variants:
        for s in seqs:
            df = base.read_csv(exp_b / "frames" / v / f"{s}.csv")
            if df is None or df.empty:
                faltas.append(f"{v}/{s}")
                continue
            frames_by[(v, s)] = df[["frame", "time_ms", "n_gt"]].copy()
    for s in seqs:
        if all((v, s) in frames_by for v in variants):
            common.append(s)
    for v in variants:
        allv = []
        for s in common:
            df = frames_by[(v, s)]
            x = df["n_gt"].to_numpy(float)
            y = df["time_ms"].to_numpy(float)
            Si = estat.suficientes(x, y)
            S[v].append(Si)
            ann = x[x > 0]
            per_seq.append({"variante": v, "seq": s, "quadros": int(x.size),
                            "densidade_media": float(ann.mean()) if ann.size else float("nan"),
                            "tempo_medio_ms": float(y.mean()), "fps": 1000.0 / float(y.mean())})
            suf.append({"variante": v, "seq": s, **dict(zip(["n", "sx", "sy", "sxx", "sxy", "syy"], Si.tolist()))})
            allv.append(df)
        if allv:
            cat = pd.concat(allv, ignore_index=True)
            g = cat.groupby("n_gt")["time_ms"].agg(["size", "mean", "median"]).reset_index()
            g.columns = ["n_alvos", "quadros", "tempo_medio_ms", "tempo_mediano_ms"]
            g.insert(0, "variante", v)
            bins.append(g)
    base.write_csv(pd.DataFrame(per_seq), out / "custo_por_seq.csv")
    base.write_csv(pd.DataFrame(suf), out / "custo_suficientes.csv")
    base.write_csv(pd.concat(bins, ignore_index=True) if bins else pd.DataFrame(), out / "custo_por_n_alvos.csv")

    if not common:
        base.log("Parte 1d: sem tempos por quadro em comum")
        return 1
    S = {v: np.vstack(S[v]) for v in variants}
    diffs = [(v, "baseline_ref") for v in variants if v != "baseline_ref"]
    boot = estat.cluster_slopes(S, conf_levels=(0.95,), seed=int(cfg["seed"]), n_resamples=int(cfg["bootstrap_n"]),
                                diffs=diffs)
    rows = []
    for v in variants:
        a, b, r2 = estat.ols_from_S(S[v].sum(axis=0))
        bt = boot[v]
        rows.append({"variante": v, "rotulo": cfg["mot"]["variantes"][v], "tipo": "inclinacao",
                     "intercepto_ms": a, "inclinacao_ms_por_alvo": b, "r2": r2,
                     "ic95_inf": bt["ic"][0.95][0], "ic95_sup": bt["ic"][0.95][1], "p_bootstrap": float("nan"),
                     "quadros": int(S[v][:, 0].sum()), "sequencias": int(S[v].shape[0]),
                     "tempo_medio_ms": float(S[v][:, 2].sum() / S[v][:, 0].sum())})
    for a_, b_ in diffs:
        bt = boot[f"{a_} - {b_}"]
        rows.append({"variante": f"{a_} - {b_}", "rotulo": "diferença de inclinação", "tipo": "diferenca",
                     "intercepto_ms": float("nan"), "inclinacao_ms_por_alvo": bt["estimativa"], "r2": float("nan"),
                     "ic95_inf": bt["ic"][0.95][0], "ic95_sup": bt["ic"][0.95][1], "p_bootstrap": bt["p"],
                     "quadros": float("nan"), "sequencias": bt["K"], "tempo_medio_ms": float("nan")})
    base.write_csv(pd.DataFrame(rows), out / "custo_regressao.csv")
    base.write_json(out / "custo_checagem.json", {"sequencias_em_comum": common, "faltando": faltas})
    base.log("Parte 1d concluida")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
