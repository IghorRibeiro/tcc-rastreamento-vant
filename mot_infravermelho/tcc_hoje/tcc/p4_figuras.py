"""Parte 4 -- figuras em pt-BR, lendo SOMENTE os CSVs de resultados/.

Padrao: texto em portugues, virgula decimal, fonte serifada tipo Times, PDF vetorial
para 16 cm de largura, paleta Okabe-Ito (segura para daltonismo), sem titulo na figura.
Tambem grava um PNG de conferencia ao lado de cada PDF.
"""

from __future__ import annotations

import numpy as np

from . import base

NAN = float("nan")
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
LARGURA = 16 / 2.54
CORES_VAR = {"baseline_ref": OKABE[0], "hit_ref": OKABE[1], "hybrid_csrt_n15": OKABE[2]}
MARC_VAR = {"baseline_ref": "o", "hit_ref": "s", "hybrid_csrt_n15": "^"}


def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Nimbus Roman No9 L", "TeX Gyre Termes",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": True,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    return plt


def br(v, nd=2):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{nd}f}".replace(".", ",").replace("-", "−")


def _tick(v, _pos):
    s = f"{v:.6g}"
    if "e" in s:
        s = f"{v:.2g}"
    return s.replace(".", ",").replace("-", "−")


def virgula(ax, x=True, y=True):
    from matplotlib.ticker import FuncFormatter

    if x:
        ax.xaxis.set_major_formatter(FuncFormatter(_tick))
    if y:
        ax.yaxis.set_major_formatter(FuncFormatter(_tick))


def letra(ax, s):
    ax.text(0.01, 0.99, s, transform=ax.transAxes, ha="left", va="top", fontsize=9)


def salvar(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    import matplotlib.pyplot as plt

    plt.close(fig)
    base.log(f"  figura: {path.with_suffix('.pdf').name}")


# ---------------------------------------------------------------------------

def fig_sige_amortecimento(plt, out, figs):
    res = base.read_csv(out / "parte1" / "sot_resumo_metodos.csv")
    if res is None or res.empty:
        return
    cfg = base.load_cfg()
    hib = cfg["sot"]["hibrido"]
    hs = [int(h) for h in cfg["sot"]["horizons"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARGURA, 2.6), gridspec_kw={"width_ratios": [1, 1.6]})
    r = res[res["metodo"] == hib]
    if len(r):
        r = r.iloc[0]
        for tag, lab, col, mk in (("pers", "Persistente (γ = 1)", OKABE[0], "o"),
                                  ("amort", "Amortecido (γ = 0,95)", OKABE[1], "s")):
            y = np.array([r[f"H{h}_px_{tag}"] for h in hs], dtype=float)
            lo = np.array([r.get(f"H{h}_px_{tag}_ic_inf", NAN) for h in hs], dtype=float)
            hi = np.array([r.get(f"H{h}_px_{tag}_ic_sup", NAN) for h in hs], dtype=float)
            yerr = np.vstack([np.where(np.isfinite(lo), y - lo, 0), np.where(np.isfinite(hi), hi - y, 0)])
            a1.errorbar(hs, y, yerr=yerr, color=col, marker=mk, ms=4, lw=1.2, capsize=2.5, label=lab)
        a1.set_xlabel("Horizonte H (quadros)")
        a1.set_ylabel("Erro de predição (px)")
        a1.set_xticks(hs)
        a1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92))
        virgula(a1)
        letra(a1, "(a)")
    ordem = cfg["sot"]["referencias"] + cfg["sot"]["locais"]
    rr = res.set_index("metodo").reindex([m for m in ordem if m in set(res["metodo"])])
    x = np.arange(len(rr))
    wbar = 0.38
    a2.bar(x - wbar / 2, rr["H30_px_pers"], wbar, color=OKABE[0], label="Persistente (γ = 1)")
    a2.bar(x + wbar / 2, rr["H30_px_amort"], wbar, color=OKABE[1], label="Amortecido (γ = 0,95)")
    for xi, (_, row) in zip(x, rr.iterrows()):
        if np.isfinite(row.get("reducao_H30_pct", NAN)):
            a2.text(xi, max(row["H30_px_pers"], row["H30_px_amort"]) * 1.02, f"{br(-row['reducao_H30_pct'], 0)}%",
                    ha="center", va="bottom", fontsize=6.5)
    a2.set_xticks(x)
    a2.set_xticklabels([("Híbrido CSRT" if m == hib else m) for m in rr.index], rotation=35, ha="right")
    a2.set_ylabel("Erro em H30 (px)")
    a2.legend(loc="upper right")
    virgula(a2, x=False)
    letra(a2, "(b)")
    fig.tight_layout()
    salvar(fig, figs / "sige_amortecimento")


def fig_correlacoes(plt, out, figs):
    tab = base.read_csv(out / "parte1" / "causa_raiz_por_seq.csv")
    cor = base.read_csv(out / "parte1" / "causa_raiz_correlacoes.csv")
    if tab is None or tab.empty:
        return
    cfg = base.load_cfg()
    rot = {f["coluna"]: f["rotulo"] for f in cfg["causa_raiz"]["fatores"]}
    cols = [c for c in cfg["causa_raiz"]["figura"] if c in tab.columns]
    fig, axes = plt.subplots(2, 2, figsize=(LARGURA, 4.9))
    for ax, col, lt in zip(axes.flat, cols, "abcd"):
        x = tab[col].to_numpy(float)
        y = tab["IDF1"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=14, color=OKABE[0], edgecolor="white", linewidth=0.4, zorder=3)
        if ok.sum() >= 3:
            b, a = np.polyfit(x[ok], y[ok], 1)
            xs = np.linspace(x[ok].min(), x[ok].max(), 50)
            ax.plot(xs, a + b * xs, color=OKABE[1], lw=1.1)
        r = NAN
        if cor is not None:
            sel = cor[(cor["fator"] == col) & (cor["desfecho"] == "IDF1")]
            if len(sel):
                r = float(sel.iloc[0]["pearson_r"])
        ax.text(0.97, 0.06, f"r = {br(r, 2)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5)
        ax.set_xlabel(rot.get(col, col))
        ax.set_ylabel("IDF1 da linha de base (%)")
        virgula(ax)
        letra(ax, f"({lt})")
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    salvar(fig, figs / "lafusion_correlacoes")


def fig_gm(plt, out, figs):
    res = base.read_csv(out / "parte2" / "rodadas_resumo.csv")
    if res is None or res.empty:
        return
    g = res[res["grupo"] == "f_gm"].sort_values("gamma")
    if g.empty:
        return
    geral = base.read_csv(out / "parte1" / "mot_metricas_geral.csv")
    bl = None
    if geral is not None and len(geral[geral["variante"] == "baseline_ref"]):
        bl = geral[geral["variante"] == "baseline_ref"].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(LARGURA, 2.4))
    for ax, col, lab, lt in zip(axes, ("MOTA", "IDF1", "IDs"), ("MOTA (%)", "IDF1 (%)", "Trocas de identidade (IDs)"),
                                "abc"):
        ax.plot(g["gamma"], g[col], color=OKABE[0], marker="o", ms=3.5, lw=1.2, label="Nova rodada (estado zerado)")
        old = g[f"{col}_antigo"] if f"{col}_antigo" in g.columns else None
        if old is not None and np.isfinite(old.to_numpy(float)).any():
            ax.plot(g["gamma"], old, color="0.55", marker="o", ms=3.5, mfc="white", lw=0.8, ls="--",
                    label="Varredura antiga")
        sel = g[np.isclose(g["gamma"], 0.55)]
        if len(sel):
            ax.scatter(sel["gamma"], sel[col], s=40, facecolor="none", edgecolor=OKABE[1], lw=1.2, zorder=4,
                       label="γ = 0,55 (HIT-Tracker)")
        if bl is not None and col in bl.index and np.isfinite(float(bl[col])):
            ax.axhline(float(bl[col]), color=OKABE[2], lw=0.9, ls=":", label="Linha de base (sem GM/NSA)")
        ax.set_xlabel("γ (Geman-McClure)")
        ax.set_ylabel(lab)
        virgula(ax)
        letra(ax, f"({lt})")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    salvar(fig, figs / "lafusion_sensibilidade_gm")


def fig_fps(plt, out, figs):
    ps = base.read_csv(out / "parte1" / "custo_por_seq.csv")
    bins = base.read_csv(out / "parte1" / "custo_por_n_alvos.csv")
    reg = base.read_csv(out / "parte1" / "custo_regressao.csv")
    if ps is None or ps.empty:
        return
    labels = base.load_cfg()["mot"]["variantes"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARGURA, 2.7))
    for v, lab in labels.items():
        s = ps[ps["variante"] == v]
        if s.empty:
            continue
        a1.scatter(s["densidade_media"], s["fps"], s=14, color=CORES_VAR[v], marker=MARC_VAR[v], label=lab,
                   edgecolor="white", linewidth=0.3)
        if bins is not None:
            b = bins[(bins["variante"] == v) & (bins["quadros"] >= 30)]
            a2.scatter(b["n_alvos"], b["tempo_medio_ms"], s=9, color=CORES_VAR[v], marker=MARC_VAR[v], alpha=0.8,
                       label=lab)
            if reg is not None and len(b):
                rr = reg[reg["variante"] == v]
                if len(rr):
                    rr = rr.iloc[0]
                    xs = np.linspace(b["n_alvos"].min(), b["n_alvos"].max(), 50)
                    a2.plot(xs, rr["intercepto_ms"] + rr["inclinacao_ms_por_alvo"] * xs, color=CORES_VAR[v], lw=1.1)
    a1.set_yscale("log")
    a2.set_yscale("log")
    a1.set_xlabel("Densidade média da sequência (alvos/quadro)")
    a1.set_ylabel("Quadros por segundo")
    a2.set_xlabel("Alvos no quadro")
    a2.set_ylabel("Tempo por quadro (ms)")
    for ax, lt in ((a1, "(a)"), (a2, "(b)")):
        virgula(ax)
        from matplotlib.ticker import NullFormatter

        ax.yaxis.set_minor_formatter(NullFormatter())
        letra(ax, lt)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    salvar(fig, figs / "mot_fps_densidade")


def fig_h30_idf1(plt, out, figs):
    pred = base.read_csv(out / "parte1" / "mot_pred_por_seq.csv")
    met = base.read_csv(out / "parte1" / "mot_metricas_por_seq.csv")
    est = base.read_csv(out / "estatistica_testes.csv")
    if pred is None or met is None or pred.empty:
        return
    labels = base.load_cfg()["mot"]["variantes"]
    fig, ax = plt.subplots(figsize=(LARGURA * 0.62, 2.9))
    for v in ("baseline_ref", "hit_ref"):
        e = pred[(pred["variante"] == v) & (pred["H"] == 30) & (pred["damping"] == 1.0)].set_index("seq")["mean_norm"]
        i = met[met["variante"] == v].set_index("seq")["idf1"] * 100
        j = e.to_frame("e").join(i.to_frame("i"), how="inner").dropna()
        rho = NAN
        if est is not None:
            sel = est[est["teste"].astype(str).str.startswith("Spearman") & est["teste"].astype(str).str.endswith(labels[v])]
            if len(sel):
                rho = float(sel.iloc[0]["estimativa"])
        ax.scatter(j["i"], j["e"], s=16, color=CORES_VAR[v], marker=MARC_VAR[v], edgecolor="white", linewidth=0.3,
                   label=f"{labels[v]} (ρ = {br(rho, 2)})")
    ax.set_xlabel("IDF1 da sequência (%)")
    ax.set_ylabel("Erro em H30 (diagonais do alvo)")
    ax.legend(loc="upper right")
    virgula(ax)
    fig.tight_layout()
    salvar(fig, figs / "mot_h30_idf1")


def main() -> int:
    smoke = base.smoke_flag()
    out = base.out_dir(smoke)
    figs = base.sub(smoke, "figuras")
    plt = _setup()
    falhas = 0
    for fn in (fig_sige_amortecimento, fig_correlacoes, fig_gm, fig_fps, fig_h30_idf1):
        try:
            fn(plt, out, figs)
        except Exception as exc:  # noqa: BLE001
            falhas += 1
            base.log(f"  ERRO na figura {fn.__name__}: {exc!r}")
    base.log("Parte 4 concluida")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
