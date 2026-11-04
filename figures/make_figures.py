"""Figures 4 to 8 of the TCC, drawn from results/ only.

Style of the TCC: Portuguese text, decimal comma, serif font, vector PDF sized for a 16 cm text width,
colour-blind-safe Okabe-Ito palette, no title inside the figure. A PNG about 1600 px wide is saved next to
each PDF for the web (README).

    python figures/make_figures.py                  # writes figures/out/*.pdf and *.png
    python figures/make_figures.py --only fig7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stats"))
import inference  # noqa: E402

NAN = float("nan")
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
WIDTH_IN = 16 / 2.54
PNG_PX = 1600
SEED, N_BOOT = 2026, 10000

MOT_LABELS = {"baseline": "Linha de base (YOLO26n + BoT-SORT)", "hit_tracker": "HIT-Tracker",
              "hybrid_csrt_n15": "Híbrido CSRT (N=15)"}
MOT_COLORS = {"baseline": OKABE[0], "hit_tracker": OKABE[1], "hybrid_csrt_n15": OKABE[2]}
MOT_MARKERS = {"baseline": "o", "hit_tracker": "s", "hybrid_csrt_n15": "^"}
SOT_ORDER = ["BoT-SORT", "ByteTrack", "DeepSORT", "OSTrack", "ODTrack", "CSRT", "KCF", "MOSSE", "MedianFlow"]
ROOT_FACTORS = [("mean_density", "Densidade (alvos/quadro)"),
                ("mean_motion", "Movimento de câmera (px/quadro)"),
                ("mean_edge_density", "Densidade de bordas"),
                ("mean_local_contrast", "Contraste local")]


def setup():
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


def pt(v, nd=2):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{nd}f}".replace(".", ",").replace("-", "−")


def _tick(v, _pos):
    s = f"{v:.6g}"
    if "e" in s:
        s = f"{v:.2g}"
    return s.replace(".", ",").replace("-", "−")


def decimal_comma(ax, x=True, y=True):
    from matplotlib.ticker import FuncFormatter

    if x:
        ax.xaxis.set_major_formatter(FuncFormatter(_tick))
    if y:
        ax.yaxis.set_major_formatter(FuncFormatter(_tick))


def panel_letter(ax, s):
    ax.text(0.01, 0.99, s, transform=ax.transAxes, ha="left", va="top", fontsize=9)


def save(fig, path: Path):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    dpi = PNG_PX / (fig.get_size_inches()[0] if fig.get_size_inches()[0] else WIDTH_IN)
    fig.savefig(path.with_suffix(".png"), dpi=dpi)
    plt.close(fig)
    print(f"  {path.with_suffix('.pdf').name}, {path.with_suffix('.png').name}")


# --------------------------------------------------------------------------- Figure 4

def fig4_root_cause(plt, res: Path, out: Path):
    tab = pd.read_csv(res / "root_cause_per_seq.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 4.9))
    for ax, (col, label), letter in zip(axes.flat, ROOT_FACTORS, "abcd"):
        x, y = tab[col].to_numpy(float), tab["IDF1"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=14, color=OKABE[0], edgecolor="white", linewidth=0.4, zorder=3)
        b, a = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        ax.plot(xs, a + b * xs, color=OKABE[1], lw=1.1)
        r = inference.correlation(x, y, "pearson", n_resamples=1000)["r"]
        ax.text(0.97, 0.06, f"r = {pt(r, 2)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5)
        ax.set_xlabel(label)
        ax.set_ylabel("IDF1 da linha de base (%)")
        decimal_comma(ax)
        panel_letter(ax, f"({letter})")
    fig.tight_layout()
    save(fig, out / "fig4_correlacoes")


# --------------------------------------------------------------------------- Figure 5

def fig5_gamma(plt, res: Path, out: Path):
    runs = pd.read_csv(res / "incremental_runs_summary.csv")
    g = runs[runs["group"] == "f_gm"].sort_values("gamma")
    summary = pd.read_csv(res / "summary.csv")
    base = summary[(summary.table == "T12") & (summary.row == "baseline")].set_index("column")["value"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.4))
    for ax, col, label, letter in zip(axes, ("MOTA", "IDF1", "IDs"),
                                      ("MOTA (%)", "IDF1 (%)", "Trocas de identidade (IDs)"), "abc"):
        ax.plot(g["gamma"], g[col], color=OKABE[0], marker="o", ms=3.5, lw=1.2, label="Geman-McClure + NSA-Kalman")
        sel = g[np.isclose(g["gamma"], 0.55)]
        ax.scatter(sel["gamma"], sel[col], s=40, facecolor="none", edgecolor=OKABE[1], lw=1.2, zorder=4,
                   label="γ = 0,55 (HIT-Tracker)")
        ax.axhline(float(base[col]), color=OKABE[2], lw=0.9, ls=":", label="Linha de base")
        ax.set_xlabel("γ (Geman-McClure)")
        ax.set_ylabel(label)
        decimal_comma(ax)
        panel_letter(ax, f"({letter})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out / "fig5_gamma_geman_mcclure")


# --------------------------------------------------------------------------- Figure 6

def fig6_cost(plt, res: Path, out: Path):
    from matplotlib.ticker import NullFormatter

    per_seq = pd.read_csv(res / "cost_per_seq.csv")
    bins = pd.read_csv(res / "cost_by_num_targets.csv")
    sums = pd.read_csv(res / "cost_sufficient_stats.csv")
    cols = ["n", "sx", "sy", "sxx", "sxy", "syy"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.7))
    for v, label in MOT_LABELS.items():
        s = per_seq[per_seq["variant"] == v]
        a1.scatter(s["mean_density"], 1000.0 / s["mean_time_ms"], s=14, color=MOT_COLORS[v], marker=MOT_MARKERS[v],
                   label=label, edgecolor="white", linewidth=0.3)
        b = bins[(bins["variant"] == v) & (bins["frames"] >= 30)]
        a2.scatter(b["n_targets"], b["mean_time_ms"], s=9, color=MOT_COLORS[v], marker=MOT_MARKERS[v], alpha=0.8,
                   label=label)
        intercept, slope, _ = inference.ols_from_sums(sums[sums["variant"] == v][cols].to_numpy(float).sum(axis=0))
        xs = np.linspace(b["n_targets"].min(), b["n_targets"].max(), 50)
        a2.plot(xs, intercept + slope * xs, color=MOT_COLORS[v], lw=1.1)
    a1.set_yscale("log")
    a2.set_ylim(bottom=0)
    a1.set_xlabel("Densidade média da sequência (alvos/quadro)")
    a1.set_ylabel("Quadros por segundo")
    a2.set_xlabel("Alvos no quadro")
    a2.set_ylabel("Tempo por quadro (ms)")
    for ax, letter in ((a1, "(a)"), (a2, "(b)")):
        decimal_comma(ax)
        ax.yaxis.set_minor_formatter(NullFormatter())
        panel_letter(ax, letter)
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, out / "fig6_custo_densidade")


# --------------------------------------------------------------------------- Figure 7

def fig7_damping(plt, res: Path, out: Path):
    pred = pd.read_csv(res / "sot_prediction_per_video.csv")
    horizons = [5, 10, 15, 30]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.6), gridspec_kw={"width_ratios": [1, 1.6]})
    for gamma, label, color, marker in ((1.0, "Persistente (γ = 1)", OKABE[0], "o"),
                                        (0.95, "Amortecido (γ = 0,95)", OKABE[1], "s")):
        y, lo, hi = [], [], []
        for h in horizons:
            v = pred[(pred["method"] == "CSRT") & (pred["damping"] == gamma) & (pred["H"] == h)]["mean_px"].to_numpy()
            l, u, _ = inference.bca_mean(v, 0.95, SEED, N_BOOT)
            y.append(v.mean())
            lo.append(l)
            hi.append(u)
        y, lo, hi = map(np.array, (y, lo, hi))
        a1.errorbar(horizons, y, yerr=np.vstack([y - lo, hi - y]), color=color, marker=marker, ms=4, lw=1.2,
                    capsize=2.5, label=label)
    a1.set_xlabel("Horizonte H (quadros)")
    a1.set_ylabel("Erro de predição (px)")
    a1.set_xticks(horizons)
    a1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92))
    decimal_comma(a1)
    panel_letter(a1, "(a)")

    def h30(method, gamma):
        return pred[(pred["method"] == method) & (pred["damping"] == gamma) & (pred["H"] == 30)]["mean_px"].mean()

    x = np.arange(len(SOT_ORDER))
    width = 0.38
    persistent = np.array([h30(m, 1.0) for m in SOT_ORDER])
    damped = np.array([h30(m, 0.95) for m in SOT_ORDER])
    a2.bar(x - width / 2, persistent, width, color=OKABE[0], label="Persistente (γ = 1)")
    a2.bar(x + width / 2, damped, width, color=OKABE[1], label="Amortecido (γ = 0,95)")
    for xi, p, d in zip(x, persistent, damped):
        a2.text(xi, max(p, d) * 1.02, f"{pt(-100 * (p - d) / p, 0)}%", ha="center", va="bottom", fontsize=6.5)
    a2.set_xticks(x)
    a2.set_xticklabels(["Híbrido CSRT" if m == "CSRT" else m for m in SOT_ORDER], rotation=35, ha="right")
    a2.set_ylabel("Erro em H30 (px)")
    a2.legend(loc="upper right")
    decimal_comma(a2, x=False)
    panel_letter(a2, "(b)")
    fig.tight_layout()
    save(fig, out / "fig7_amortecimento")


# --------------------------------------------------------------------------- Figure 8

def fig8_h30_idf1(plt, res: Path, out: Path):
    pred = pd.read_csv(res / "mot_prediction_per_seq.csv")
    pred = pred[pred["mode"] == "clear"]
    tracking = pd.read_csv(res / "mot_tracking_per_seq.csv")
    tests = pd.read_csv(res / "hypothesis_tests.csv")
    fig, ax = plt.subplots(figsize=(WIDTH_IN * 0.62, 2.9))
    for v in ("baseline", "hit_tracker"):
        e = pred[(pred["variant"] == v) & (pred["H"] == 30) & (pred["damping"] == 1.0)].set_index("seq")["mean_norm"]
        i = tracking[tracking["variant"] == v].set_index("seq")["idf1"] * 100
        j = e.to_frame("e").join(i.to_frame("i"), how="inner").dropna()
        row = tests[tests["test"].str.startswith("H2") & tests["test"].str.endswith(
            "HIT-Tracker" if v == "hit_tracker" else "baseline (YOLO26n + BoT-SORT)")]
        rho = float(row.iloc[0]["estimate"]) if len(row) else NAN
        ax.scatter(j["i"], j["e"], s=16, color=MOT_COLORS[v], marker=MOT_MARKERS[v], edgecolor="white",
                   linewidth=0.3, label=f"{MOT_LABELS[v]} (ρ = {pt(rho, 2)})")
        b, a = np.polyfit(j["i"].to_numpy(float), j["e"].to_numpy(float), 1)
        xs = np.linspace(j["i"].min(), j["i"].max(), 50)
        ax.plot(xs, a + b * xs, color=MOT_COLORS[v], lw=1.1)
    ax.set_xlabel("IDF1 da sequência (%)")
    ax.set_ylabel("Erro em H30 (diagonais do alvo)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    decimal_comma(ax)
    fig.tight_layout()
    save(fig, out / "fig8_h30_idf1")


FIGURES = {"fig4": fig4_root_cause, "fig5": fig5_gamma, "fig6": fig6_cost, "fig7": fig7_damping, "fig8": fig8_h30_idf1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "out")
    parser.add_argument("--only", choices=sorted(FIGURES), nargs="*")
    args = parser.parse_args()
    plt = setup()
    failures = 0
    for name, fn in FIGURES.items():
        if args.only and name not in args.only:
            continue
        try:
            fn(plt, args.results, args.out)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR in {name}: {exc!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
