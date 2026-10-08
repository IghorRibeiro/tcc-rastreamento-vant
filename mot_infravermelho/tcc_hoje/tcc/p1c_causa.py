"""Parte 1c -- causa raiz (descritiva): fatores por sequencia x desempenho da linha de base de 28/09.

Desempenho: baseline_ref recalculado na Parte 1b (IDF1, MOTA, IDs por mil objetos).
Fatores de imagem: os CSVs ja calculados pelo LAFusion em runs/mot_eval (reaproveitados).
Pearson (r, p, IC BCa pareado) e Spearman (rho, p) como checagem. Sem correcao
multipla: e descritivo, como pedido. Tambem: r antigo (com a rodada publicada do BoT-SORT)
e a correlacao entre os proprios fatores (densidade de bordas x contraste local etc.).
"""

from __future__ import annotations

import pandas as pd

from . import base, estat


def main() -> int:
    smoke = base.smoke_flag()
    cfg = base.load_cfg()
    out = base.sub(smoke, "parte1")
    met = base.read_csv(out / "mot_metricas_por_seq.csv")
    if met is None or met.empty:
        base.log("Parte 1c: mot_metricas_por_seq.csv ausente (Parte 1b falhou?)")
        return 1
    bl = met[met["variante"] == "baseline_ref"].copy()
    bl["IDF1"] = 100 * bl["idf1"]
    bl["MOTA"] = 100 * bl["mota"]
    bl["IDs_por_mil"] = 1000 * bl["num_switches"] / bl["num_objects"]
    tab = bl[["seq", "IDF1", "MOTA", "IDs_por_mil", "num_switches", "densidade_media"]].rename(columns={"seq": "sequence"})

    mot_dir = base.P(cfg["paths"]["mot_eval_dir"])
    fontes = {}
    for f in cfg["causa_raiz"]["fatores"]:
        arq = f.get("arquivo")
        if not arq or arq in fontes:
            continue
        df = base.read_csv(mot_dir / arq)
        fontes[arq] = df
        if df is None:
            base.log(f"AVISO: {arq} nao encontrado")
            continue
        cols = ["sequence"] + [x["coluna"] for x in cfg["causa_raiz"]["fatores"]
                               if x.get("arquivo") == arq and x["coluna"] in df.columns]
        tab = tab.merge(df[cols], on="sequence", how="left")

    # desempenho antigo (rodada publicada do BoT-SORT) para o "r antigo"
    old = base.read_csv(mot_dir / "frame_quality_vs_performance.csv")
    if old is not None and "idf1_pct_botsort" in old.columns:
        tab = tab.merge(old[["sequence", "idf1_pct_botsort", "mota_pct_botsort"]], on="sequence", how="left")
    dens_old = base.read_csv(mot_dir / "density_vs_performance.csv")
    if dens_old is not None and "IDF1(%)_botsort" in dens_old.columns and "idf1_pct_botsort" not in tab.columns:
        tab = tab.merge(dens_old[["sequence", "IDF1(%)_botsort"]].rename(columns={"IDF1(%)_botsort": "idf1_pct_botsort"}),
                        on="sequence", how="left")
    base.write_csv(tab, out / "causa_raiz_por_seq.csv")

    seed, nb = int(cfg["seed"]), int(cfg["bootstrap_n"])
    rows = []
    for f in cfg["causa_raiz"]["fatores"]:
        col = f["coluna"]
        if col not in tab.columns:
            continue
        for desfecho in ("IDF1", "MOTA", "IDs_por_mil"):
            pr = estat.correlacao(tab[col], tab[desfecho], "pearson", seed=seed, n_resamples=nb)
            sp = estat.correlacao(tab[col], tab[desfecho], "spearman", seed=seed, n_resamples=nb)
            row = {"fator": col, "rotulo": f["rotulo"], "desfecho": desfecho, "n": pr["n"],
                   "pearson_r": pr["r"], "pearson_p": pr["p"], "pearson_ic95_inf": pr["ic_inf"],
                   "pearson_ic95_sup": pr["ic_sup"], "spearman_rho": sp["r"], "spearman_p": sp["p"],
                   "obs": "; ".join(x for x in (pr["obs"], sp["obs"]) if x)}
            if desfecho == "IDF1" and "idf1_pct_botsort" in tab.columns:
                old_r = estat.correlacao(tab[col], tab["idf1_pct_botsort"], "pearson", seed=seed, n_resamples=nb)
                row["pearson_r_antigo"] = old_r["r"]
            rows.append(row)
    base.write_csv(pd.DataFrame(rows), out / "causa_raiz_correlacoes.csv")

    fat = [f["coluna"] for f in cfg["causa_raiz"]["fatores"] if f["coluna"] in tab.columns]
    frows = []
    for i, a in enumerate(fat):
        for b in fat[i + 1:]:
            pr = estat.correlacao(tab[a], tab[b], "pearson", seed=seed, n_resamples=nb)
            sp = estat.correlacao(tab[a], tab[b], "spearman", seed=seed, n_resamples=nb)
            frows.append({"fator_a": a, "fator_b": b, "n": pr["n"], "pearson_r": pr["r"], "pearson_p": pr["p"],
                          "pearson_ic95_inf": pr["ic_inf"], "pearson_ic95_sup": pr["ic_sup"],
                          "spearman_rho": sp["r"], "spearman_p": sp["p"]})
    base.write_csv(pd.DataFrame(frows), out / "causa_raiz_fatores_entre_si.csv")
    base.log("Parte 1c concluida")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
