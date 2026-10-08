"""Parte 3 -- modulo estatistico unico -> resultados/estatistica_testes.csv (uma linha por teste).

Familias (Holm dentro de cada uma; IC de Bonferroni com nivel 1 - 0,05/m):
  F1 (MOT, erro normalizado, correspondencia CLEAR, sequencia = unidade):
     amortecido - persistente em H30 (linha de base); idem (HIT);
     Spearman IDF1 x H30 normalizado persistente (linha de base); idem (HIT);
     HIT - linha de base em H30 (suporte comum, persistente);
     diferenca de inclinacao tempo~alvos hibrido - linha de base (bootstrap por sequencia).
  F2 (DUT, px, persistente): hibrido CSRT - cada referencia em H30, suporte comum, video = unidade.
  F3 (DUT, px, persistente): CSRT - KCF / MOSSE / MedianFlow em H30, com os MESMOS erros por video
     da tabela de rastreadores locais do TCC (Kalman do SIGE, sot_pred_por_video.csv). Na rodada de
     07/10 o F3 usava os erros por video salvos da execucao publicada da Tabela 2; o repositorio usa
     o Kalman do SIGE para reproduzir os numeros do texto.
  F4 (DUT, px): amortecido - persistente em H30 para cada um dos 9 metodos (Kalman do SIGE).
Diferencas pareadas: Wilcoxon bicaudal + rank-biserial; IC BCa da media das diferencas.
Correlacoes: Spearman (p da scipy) + IC BCa pareado. Inclinacoes: p e IC do bootstrap BCa
por sequencia. Nenhum IC t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import base, estat

NAN = float("nan")


class Familia:
    def __init__(self, nome, descricao):
        self.nome = nome
        self.descricao = descricao
        self.itens = []        # (row, recompute_fn(conf) -> (lo, hi))

    def add(self, row, ic_fn):
        row = {"familia": self.nome, "familia_descricao": self.descricao, **row}
        self.itens.append((row, ic_fn))

    def finalizar(self, alpha):
        rows = [r for r, _ in self.itens]
        m = len(rows)
        adj = estat.holm([r.get("p_bruto", NAN) for r in rows]) if m else []
        out = []
        for (row, fn), p_adj in zip(self.itens, adj):
            row["m_familia"] = m
            row["p_holm"] = float(p_adj)
            level = 1.0 - alpha / m if m else NAN
            row["ic_bonf_nivel"] = level
            lo, hi = fn(level) if fn else (NAN, NAN)
            row["ic_bonf_inf"], row["ic_bonf_sup"] = lo, hi
            row["significativo_holm_5pct"] = bool(np.isfinite(p_adj) and p_adj < alpha)
            out.append(row)
        return out


def pareado(d, seed, nb, **info):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    w = estat.wilcoxon_rb(d)
    lo, hi, obs = estat.bca_mean(d, 0.95, seed, nb)
    row = {**info, "n": int(d.size), "estimativa": float(d.mean()) if d.size else NAN,
           "mediana_dif": float(np.median(d)) if d.size else NAN, "ic95_inf": lo, "ic95_sup": hi,
           "estatistica": w["W"], "nome_estatistica": "W (Wilcoxon)", "p_bruto": w["p"],
           "rank_biserial": w["rank_biserial"], "metodo": "Wilcoxon pareado bicaudal; IC BCa da média das diferenças",
           "obs": "; ".join(x for x in (w["obs"], obs) if x)}
    return row, (lambda level: estat.bca_mean(d, level, seed, nb)[:2])


def spearman(x, y, seed, nb, **info):
    r = estat.correlacao(x, y, "spearman", 0.95, seed, nb)
    row = {**info, "n": r["n"], "estimativa": r["r"], "mediana_dif": NAN, "ic95_inf": r["ic_inf"],
           "ic95_sup": r["ic_sup"], "estatistica": r["r"], "nome_estatistica": "rho (Spearman)", "p_bruto": r["p"],
           "rank_biserial": NAN, "metodo": "Spearman; IC BCa pareado", "obs": r["obs"]}

    def fn(level):
        rr = estat.correlacao(x, y, "spearman", level, seed, nb)
        return rr["ic_inf"], rr["ic_sup"]

    return row, fn


def pivot_seq(df, key, value, a, b):
    """Diferenca a - b pareada pela chave (seq/video)."""
    pa = df[df["_lado"] == a].set_index(key)[value]
    pb = df[df["_lado"] == b].set_index(key)[value]
    j = pa.to_frame("a").join(pb.to_frame("b"), how="inner").dropna()
    return j["a"] - j["b"], j


def main() -> int:
    smoke = base.smoke_flag()
    cfg = base.load_cfg()
    seed, nb, alpha = int(cfg["seed"]), int(cfg["bootstrap_n"]), float(cfg["alpha"])
    out = base.out_dir(smoke)
    p1 = out / "parte1"
    rows = []

    # ---------------- F1 (MOT) ----------------
    f1 = Familia("F1", "MOT (Track 3): amortecimento, identidade x predição, HIT x linha de base, custo")
    pred = base.read_csv(p1 / "mot_pred_por_seq.csv")
    met = base.read_csv(p1 / "mot_metricas_por_seq.csv")
    sup = base.read_csv(p1 / "mot_suporte_comum_por_seq.csv")
    labels = cfg["mot"]["variantes"]
    if pred is not None and not pred.empty:
        h30 = pred[pred["H"] == 30]
        for v in ("baseline_ref", "hit_ref"):
            sel = h30[h30["variante"] == v].copy()
            sel["_lado"] = sel["damping"].map({1.0: "pers", 0.95: "amort"})
            d, _ = pivot_seq(sel, "seq", "mean_norm", "amort", "pers")
            row, fn = pareado(d, seed, nb, teste=f"Amortecido − persistente, H30, {labels[v]}",
                              variavel="erro normalizado H30 (média por sequência)", unidade="fração da diagonal",
                              comparacao="γ=0,95 − γ=1,0")
            f1.add(row, fn)
        if met is not None:
            for v in ("baseline_ref", "hit_ref"):
                e = h30[(h30["variante"] == v) & (h30["damping"] == 1.0)].set_index("seq")["mean_norm"]
                i = met[met["variante"] == v].set_index("seq")["idf1"]
                j = e.to_frame("e").join(i.to_frame("i"), how="inner").dropna()
                row, fn = spearman(j["i"].to_numpy(), j["e"].to_numpy(), seed, nb,
                                   teste=f"Spearman IDF1 × H30 normalizado (persistente), {labels[v]}",
                                   variavel="IDF1 por sequência × erro normalizado H30", unidade="ρ",
                                   comparacao="correlação")
                f1.add(row, fn)
    if sup is not None and not sup.empty:
        s = sup[(sup["a"] == "hit_ref") & (sup["b"] == "baseline_ref") & (sup["H"] == 30) & (sup["damping"] == 1.0)]
        d = (s["mean_norm_a"] - s["mean_norm_b"]).to_numpy()
        row, fn = pareado(d, seed, nb, teste="HIT − linha de base, H30 (suporte comum, persistente)",
                          variavel="erro normalizado H30 no suporte comum (média por sequência)",
                          unidade="fração da diagonal", comparacao="HIT − linha de base")
        f1.add(row, fn)
    suf = base.read_csv(p1 / "custo_suficientes.csv")
    if suf is not None and not suf.empty:
        S = {}
        seqs = None
        for v in ("hybrid_csrt_n15", "baseline_ref"):
            sv = suf[suf["variante"] == v].set_index("seq")
            seqs = sv.index if seqs is None else seqs.intersection(sv.index)
            S[v] = sv
        if seqs is not None and len(seqs):
            Sm = {v: S[v].loc[list(seqs), ["n", "sx", "sy", "sxx", "sxy", "syy"]].to_numpy(float) for v in S}
            key = "hybrid_csrt_n15 - baseline_ref"
            bt = estat.cluster_slopes(Sm, conf_levels=(0.95,), seed=seed, n_resamples=nb,
                                      diffs=[("hybrid_csrt_n15", "baseline_ref")])[key]
            row = {"teste": "Inclinação tempo×alvos: híbrido − linha de base", "variavel": "ms por alvo por quadro",
                   "unidade": "ms/alvo", "comparacao": "híbrido − linha de base", "n": bt["K"],
                   "estimativa": bt["estimativa"], "mediana_dif": NAN, "ic95_inf": bt["ic"][0.95][0],
                   "ic95_sup": bt["ic"][0.95][1], "estatistica": NAN, "nome_estatistica": "—",
                   "p_bruto": bt["p"], "rank_biserial": NAN,
                   "metodo": "bootstrap por sequência (pareado), BCa; p bicaudal do bootstrap", "obs": ""}

            def fn_slope(level, Sm=Sm, key=key):
                b2 = estat.cluster_slopes(Sm, conf_levels=(level,), seed=seed, n_resamples=nb,
                                          diffs=[("hybrid_csrt_n15", "baseline_ref")])[key]
                return b2["ic"][level]

            f1.add(row, fn_slope)
    rows += f1.finalizar(alpha)

    # ---------------- F2 (DUT, suporte comum) ----------------
    f2 = Familia("F2", "DUT: híbrido CSRT × cada referência, H30 em suporte comum (persistente)")
    ssup = base.read_csv(p1 / "sot_suporte_comum_por_video.csv")
    if ssup is not None and not ssup.empty:
        for ref in cfg["sot"]["referencias"]:
            s = ssup[(ssup["referencia"] == ref) & (ssup["H"] == 30) & (ssup["damping"] == 1.0)]
            if s.empty:
                continue
            d = (s["mean_px_hibrido"] - s["mean_px_ref"]).to_numpy()
            row, fn = pareado(d, seed, nb, teste=f"Híbrido CSRT − {ref}, H30", variavel="erro H30 em suporte comum (média por vídeo)",
                              unidade="px", comparacao=f"CSRT − {ref}")
            f2.add(row, fn)
    rows += f2.finalizar(alpha)

    # ---------------- F3 (DUT, rastreadores locais) ----------------
    # Mesmos erros por video da tabela de rastreadores locais do TCC: Kalman do SIGE
    # (uav_common.py), persistente (gamma = 1), H30, media por video (sot_pred_por_video.csv).
    f3 = Familia("F3", "DUT: CSRT × KCF, MOSSE e MedianFlow em H30 (persistente; erros por vídeo do Kalman do SIGE, "
                       "os mesmos da tabela de rastreadores locais)")
    sp = base.read_csv(p1 / "sot_pred_por_video.csv")
    if sp is not None and not sp.empty:
        h30p = sp[(sp["H"] == 30) & (sp["damping"] == 1.0)]
        csrt = h30p[h30p["metodo"] == "CSRT"].set_index("video")["mean_px"]
        for other in ("KCF", "MOSSE", "MedianFlow"):
            o = h30p[h30p["metodo"] == other].set_index("video")["mean_px"]
            j = csrt.to_frame("a").join(o.to_frame("b"), how="inner").dropna()
            orig = bool(sp[sp["metodo"] == other]["trajetoria_original"].astype(str).str.lower().eq("true").all())
            row, fn = pareado((j["a"] - j["b"]).to_numpy(), seed, nb, teste=f"CSRT − {other}, H30",
                              variavel="erro H30 por vídeo (Kalman do SIGE, tabela de rastreadores locais)",
                              unidade="px", comparacao=f"CSRT − {other}",
                              obs_trajetoria="original" if orig else "reexecução (original não encontrada)")
            f3.add(row, fn)
    rows += f3.finalizar(alpha)

    # ---------------- F4 (DUT, amortecimento) ----------------
    f4 = Familia("F4", "DUT: amortecido − persistente em H30 para os 9 métodos (Kalman do SIGE)")
    sp = base.read_csv(p1 / "sot_pred_por_video.csv")
    if sp is not None and not sp.empty:
        for metodo in cfg["sot"]["metodos"]:
            s = sp[(sp["metodo"] == metodo) & (sp["H"] == 30)].copy()
            if s.empty:
                continue
            s["_lado"] = s["damping"].map({1.0: "pers", 0.95: "amort"})
            d, _ = pivot_seq(s, "video", "mean_px", "amort", "pers")
            orig = bool(s["trajetoria_original"].astype(str).str.lower().eq("true").all())
            row, fn = pareado(d.to_numpy(), seed, nb, teste=f"Amortecido − persistente, H30, {metodo}",
                              variavel="erro H30 por vídeo", unidade="px", comparacao="γ=0,95 − γ=1,0",
                              obs_trajetoria="original" if orig else "reexecução (original não encontrada)")
            f4.add(row, fn)
    rows += f4.finalizar(alpha)

    df = pd.DataFrame(rows)
    cols = ["familia", "teste", "comparacao", "variavel", "unidade", "n", "estimativa", "mediana_dif",
            "ic95_inf", "ic95_sup", "ic_bonf_nivel", "ic_bonf_inf", "ic_bonf_sup", "nome_estatistica",
            "estatistica", "p_bruto", "p_holm", "m_familia", "significativo_holm_5pct", "rank_biserial",
            "metodo", "obs", "obs_trajetoria", "familia_descricao"]
    df = df[[c for c in cols if c in df.columns]]
    base.write_csv(df, out / "estatistica_testes.csv")
    base.log(f"Parte 3: {len(df)} testes gravados em estatistica_testes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
