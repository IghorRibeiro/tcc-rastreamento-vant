#!/usr/bin/env python3
"""Gera results/summary/ (so arquivos pequenos) a partir das saidas das etapas 0-4.

Conteudo:
    summary.md                    numeros principais, verificacoes e execucao
    hypothesis_tests.csv|.md      testes pre-registrados (HYPOTHESES.md) e resultado MECANICO da regra
    exp_a_prediction_error.csv|.md  erro H5..H30 (px e normalizado), linha de base e HIT-Tracker
    exp_a_per_sequence.csv        por sequencia: MOTA, IDF1, IDs, densidade, H30
    exp_a_common_support.csv      linha de base x HIT-Tracker nos mesmos pares
    exp_b_overview.csv|.md        hibrido(s), linha de base e HIT com deteccao em todo quadro
    exp_b_prediction_error.csv    erro H5..H30 das rodadas do Experimento B
    exp_b_per_sequence.csv        por sequencia: FPS, speedup, densidade, MOTA/IDF1, H30
    fps_vs_density.csv + fig_fps_vs_density.png
    h30_vs_idf1.csv + fig_h30_vs_idf1.png
    h4_drift.csv|.md + h4_drift_curve.csv + fig_drift_curve.png
    validation_checks.md          fidelidade ao SIGE, reproducao do LAFusion, selecao das trajetorias
    environment.json, stage_times.csv, check_setup.txt, config_used.yaml

    python mot_prediction/make_summary.py
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from motpred import stats as st  # noqa: E402
from motpred.common import (MP_DIR, load_config, log, out_dirs, read_json, rpath, set_seeds,  # noqa: E402
                            write_csv, write_json)

# Paleta de referencia (skill de visualizacao): slots 1-3, validos para qualquer par
C_BASE, C_HYB, C_HIT = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------

def load_eval(directory: Path) -> dict:
    """Junta os arquivos por sequencia de um diretorio de avaliacao."""
    out = {"mot": pd.DataFrame(), "errors": pd.DataFrame(), "tracks": pd.DataFrame(),
           "common": pd.DataFrame(), "drift": pd.DataFrame(), "drift_curve": pd.DataFrame()}
    if not directory.is_dir():
        return out
    mot = [read_json(p) for p in sorted(directory.glob("*_mot.json"))]
    out["mot"] = pd.DataFrame(mot)
    for key, pattern in (("errors", "*_errors.csv"), ("tracks", "*_tracks.csv"), ("common", "*_common.csv"),
                         ("drift_curve", "*_drift_curve.csv")):
        frames = [pd.read_csv(p) for p in sorted(directory.glob(pattern)) if p.stat().st_size > 0]
        frames = [f for f in frames if not f.empty]
        out[key] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    drift = [read_json(p) for p in sorted(directory.glob("*_drift.json"))]
    out["drift"] = pd.DataFrame(drift)
    return out


def load_frames(directory: Path) -> pd.DataFrame:
    if not directory.is_dir():
        return pd.DataFrame()
    frames = []
    for path in sorted(directory.glob("*.csv")):
        df = pd.read_csv(path)
        df.insert(0, "seq", path.stem)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fmt(value, digits=1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    return f"{value:.{digits}f}"


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(sem dados)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Agregacao do erro de predicao
# ---------------------------------------------------------------------------

def aggregate_errors(cfg, errors: pd.DataFrame, tracks: pd.DataFrame, label_col="rotulo") -> pd.DataFrame:
    """Entre sequencias (unidade = sequencia) e entre trilhas, por (rotulo, modo, gamma, H)."""
    if errors.empty:
        return pd.DataFrame()
    rows = []
    for (label, mode, damping, horizon), grp in errors.groupby([label_col, "mode", "damping", "H"], sort=True):
        px = grp["mean_px"].to_numpy(float)
        nm = grp["mean_norm"].to_numpy(float)
        low, high = st.ci95_t(cfg, px)
        low_n, high_n = st.ci95_t(cfg, nm)
        row = {
            "rotulo": label, "correspondencia": mode, "gamma": damping, "H": int(horizon),
            "sequencias_com_pares": int(np.isfinite(px).sum()),
            "media_px": float(np.nanmean(px)) if np.isfinite(px).any() else math.nan,
            "dp_px": float(pd.Series(px).std(ddof=1)),
            "ic95_inf_px": low, "ic95_sup_px": high,
            "media_norm": float(np.nanmean(nm)) if np.isfinite(nm).any() else math.nan,
            "dp_norm": float(pd.Series(nm).std(ddof=1)),
            "ic95_inf_norm": low_n, "ic95_sup_norm": high_n,
            "pares_validos": int(grp["validos"].sum()),
            "passos_kalman": int(grp["passos"].sum()),
            "sem_assoc": int(grp["sem_assoc"].sum()),
            "descartados_alvo_ausente": int(grp["ausente"].sum()),
        }
        denom = row["pares_validos"] + row["descartados_alvo_ausente"]
        row["frac_descartados"] = row["descartados_alvo_ausente"] / denom if denom else math.nan
        if not tracks.empty:
            tsel = tracks[(tracks[label_col] == label) & (tracks["mode"] == mode) & (tracks["damping"] == damping)
                          & (tracks["H"] == horizon)]
            row["trilhas_n"] = int(len(tsel))
            row["trilhas_media_px"] = float(tsel["mean_px"].mean()) if len(tsel) else math.nan
            row["trilhas_dp_px"] = float(tsel["mean_px"].std(ddof=1)) if len(tsel) > 1 else math.nan
            row["trilhas_media_norm"] = float(tsel["mean_norm"].mean()) if len(tsel) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compact_error_table(agg: pd.DataFrame, labels: list[str], value="px") -> pd.DataFrame:
    """Linhas rotulo x gamma (pi_CLEAR), colunas H5..H30 'media ± dp'; + IC95% de H30."""
    rows = []
    if agg.empty:
        return pd.DataFrame()
    for label in labels:
        for damping in sorted(agg["gamma"].unique(), reverse=True):
            sel = agg[(agg["rotulo"] == label) & (agg["correspondencia"] == "clear") & (agg["gamma"] == damping)]
            if sel.empty:
                continue
            row = {"Rastreador": label, "γ": fmt(damping, 2)}
            for _, r in sel.sort_values("H").iterrows():
                digits = 1 if value == "px" else 3
                row[f"H{int(r['H'])}"] = f"{fmt(r[f'media_{value}'], digits)} ± {fmt(r[f'dp_{value}'], digits)}"
                if int(r["H"]) == 30:
                    row["IC95% H30"] = f"[{fmt(r[f'ic95_inf_{value}'], digits)}; {fmt(r[f'ic95_sup_{value}'], digits)}]"
                    row["pares H30"] = int(r["pares_validos"])
                    row["descartados H30"] = f"{fmt(100 * r['frac_descartados'], 1)}%"
            rows.append(row)
    return pd.DataFrame(rows)


def per_seq_value(errors: pd.DataFrame, label: str, mode="clear", damping=1.0, horizon=30,
                  col="mean_px") -> pd.Series:
    sel = errors[(errors["rotulo"] == label) & (errors["mode"] == mode) & (errors["damping"] == damping)
                 & (errors["H"] == horizon)]
    return sel.set_index("seq")[col]


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _figure(ncols=1, width=6.4, height=4.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=150, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax in axes:
        _style(ax)
    return fig, axes, plt


def fig_fps_density(table: pd.DataFrame, path: Path, series: list[tuple[str, str, str]]) -> None:
    if table.empty:
        return
    fig, (ax1, ax2), plt = _figure(ncols=2, width=10.0, height=4.0)
    for variant, name, color in series:
        sel = table[table["variante"] == variant].sort_values("centro")
        if sel.empty:
            continue
        ax1.plot(sel["centro"], sel["fps"], color=color, linewidth=2, marker="o", markersize=5, label=name)
        if variant != "baseline_ref" and "speedup" in sel:
            ax2.plot(sel["centro"], sel["speedup"], color=color, linewidth=2, marker="o", markersize=5, label=name)
    ax2.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
    ax2.text(ax2.get_xlim()[0], 1.0, " paridade", color=INK2, fontsize=8, va="bottom")
    ax1.set_xlabel("alvos anotados no quadro (faixa)")
    ax1.set_ylabel("FPS (1000 / tempo médio por quadro)")
    ax1.set_title("FPS × densidade", color=INK, fontsize=11, loc="left")
    ax2.set_xlabel("alvos anotados no quadro (faixa)")
    ax2.set_ylabel("speedup = FPS / FPS da linha de base")
    ax2.set_title("Vantagem de custo sobre a detecção em todo quadro", color=INK, fontsize=11, loc="left")
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK)
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_h30_idf1(table: pd.DataFrame, path: Path, series: list[tuple[str, str, str]]) -> None:
    if table.empty:
        return
    fig, (ax,), plt = _figure(width=6.4, height=4.4)
    for label, name, color in series:
        sel = table[table["rastreador"] == label]
        if sel.empty:
            continue
        rho = st.correlation(sel["IDF1"], sel["H30_norm"])
        ax.scatter(sel["IDF1"] * 100, sel["H30_norm"], s=36, color=color, edgecolor=SURFACE, linewidth=1.2,
                   label=f"{name} (Spearman ρ = {fmt(rho['spearman_rho'], 2)}, n = {rho['n']})")
    ax.set_xlabel("IDF1 da sequência (%)")
    ax.set_ylabel("erro médio em H30 / diagonal do alvo")
    ax.set_title("Erro de predição em H30 × IDF1 (uma marca por sequência)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_drift(curves: pd.DataFrame, path: Path, n_red: int) -> None:
    if curves.empty:
        return
    fig, (ax,), plt = _figure(width=6.4, height=4.0)
    for domain, name, color in (("track3_ir", "Track 3 (infravermelho, MOT)", C_HYB),
                                ("dut_visivel", "DUT Anti-UAV (visível, SOT)", C_BASE)):
        sel = curves[curves["dominio"] == domain].sort_values("age")
        if sel.empty:
            continue
        ax.plot(sel["age"], sel["e_norm_media"], color=color, linewidth=2, marker="o", markersize=4, label=name)
    ax.axvline(n_red - 1, color=INK2, linewidth=1, linestyle="--")
    ax.text(n_red - 1, ax.get_ylim()[1], f" a = N_red − 1 = {n_red - 1}", color=INK2, fontsize=8, va="top")
    ax.set_xlabel("idade desde a última ancoragem (quadros)")
    ax.set_ylabel("erro do centro / diagonal do alvo")
    ax.set_title("Deriva do CSRT entre redetecções", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Regras de decisao (HYPOTHESES.md, secao 3) -- resultado mecanico
# ---------------------------------------------------------------------------

def paired_decision(result: dict) -> str:
    low, high = result.get("IC95_inf"), result.get("IC95_sup")
    if low is None or high is None or not (math.isfinite(low) and math.isfinite(high)):
        return "sem dados"
    if high < 0:
        return "diferença < 0"
    if low > 0:
        return "diferença > 0"
    return "IC95% contém 0"


def corr_decision(result: dict, alpha: float) -> str:
    rho, p = result.get("spearman_rho"), result.get("spearman_p")
    if rho is None or p is None or not (math.isfinite(rho) and math.isfinite(p)):
        return "sem dados"
    if p < alpha and rho < 0:
        return "ρ < 0 significativa"
    if p < alpha and rho > 0:
        return "ρ > 0 significativa"
    return "não significativa"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seeds(int(cfg["seed"]))
    alpha = float(cfg["analysis"]["alpha"])
    dirs = out_dirs(cfg)
    out = dirs["summary"]
    out.mkdir(parents=True, exist_ok=True)
    hyb_name = cfg["hybrid"]["name"]
    base_ref = cfg["reference"]["baseline"]["name"]
    hit_ref = cfg["reference"]["hit"]["name"]
    tests = []
    notes = []

    # ------------------------------------------------------------ Experimento A
    a_base = load_eval(dirs["exp_a"] / "eval" / "baseline")
    a_hit = load_eval(dirs["exp_a"] / "eval" / "hit")
    errors_a = pd.concat([a_base["errors"], a_hit["errors"]], ignore_index=True)
    tracks_a = pd.concat([a_base["tracks"], a_hit["tracks"]], ignore_index=True)
    agg_a = aggregate_errors(cfg, errors_a, tracks_a)
    if not agg_a.empty:
        write_csv(agg_a, out / "exp_a_prediction_error.csv")
        md = ["# Experimento A: erro de predição com identidade (π^CLEAR)", "",
              "Média ± desvio-padrão entre sequências (unidade = sequência); IC95% por t de Student.", "",
              "## Pixels", "", md_table(compact_error_table(agg_a, ["baseline", "hit"], "px")),
              "## Normalizado pela diagonal da caixa do alvo", "",
              md_table(compact_error_table(agg_a, ["baseline", "hit"], "norm"))]
        sec = agg_a[(agg_a["correspondencia"] == "idf1") & (agg_a["H"] == 30)][
            ["rotulo", "gamma", "media_px", "dp_px", "media_norm", "pares_validos", "frac_descartados"]].round(3)
        md += ["## H30 com a correspondência secundária (π^ID, mapeamento do IDF1)", "", md_table(sec)]
        (out / "exp_a_prediction_error.md").write_text("\n".join(md), encoding="utf-8")

    per_seq_rows = []
    for label, data in (("baseline", a_base), ("hit", a_hit)):
        if data["mot"].empty:
            continue
        for _, rec in data["mot"].iterrows():
            seq = rec["seq"]
            row = {"seq": seq, "rastreador": label, "MOTA": rec["mot_mota"], "IDF1": rec["mot_idf1"],
                   "IDs": rec["mot_num_switches"], "FP": rec["mot_num_false_positives"],
                   "FN": rec["mot_num_misses"], "GT": rec["mot_num_objects"],
                   "densidade_media": rec["densidade_media"], "densidade_max": rec["densidade_max"],
                   "trilhas": rec.get("lacunas_n_trilhas"),
                   "frac_passos_apos_lacuna": rec.get("lacunas_fracao_passos_apos_lacuna"),
                   "idf1_reconstruido_confere": rec.get("mot_idf1_confere")}
            for damping, tag in ((1.0, "persist"), (0.95, "amort")):
                for col, name in (("mean_px", "px"), ("mean_norm", "norm")):
                    values = per_seq_value(errors_a, label, "clear", damping, 30, col)
                    row[f"H30_{name}_{tag}"] = float(values.get(seq, math.nan))
            per_seq_rows.append(row)
    per_seq_a = pd.DataFrame(per_seq_rows)
    if not per_seq_a.empty:
        write_csv(per_seq_a.round(6), out / "exp_a_per_sequence.csv")

    common = a_hit["common"]
    if not common.empty:
        rows = []
        for (damping, horizon), grp in common.groupby(["damping", "H"], sort=True):
            res = st.paired(cfg, grp["mean_px_b"], grp["mean_px_a"])
            rows.append({"gamma": damping, "H": horizon, "sequencias": len(grp), "pares_comuns": int(grp["n_comum"].sum()),
                         "base_px": float(grp["mean_px_a"].mean()), "hit_px": float(grp["mean_px_b"].mean()),
                         "base_norm": float(grp["mean_norm_a"].mean()), "hit_norm": float(grp["mean_norm_b"].mean()),
                         "dif_hit_menos_base_px": res["diferenca"], "IC95_inf": res["IC95_inf"],
                         "IC95_sup": res["IC95_sup"], "p_wilcoxon": res["p_wilcoxon"]})
        write_csv(pd.DataFrame(rows).round(4), out / "exp_a_common_support.csv")

    # ------------------------------------------------------------ H1, H2a, H2b
    def series(label, damping=1.0, col="mean_px", errors=errors_a):
        return per_seq_value(errors, label, "clear", damping, 30, col)

    h1_outcomes = []
    for label in ("baseline", "hit"):
        if errors_a.empty:
            continue
        persist, damped = series(label, 1.0), series(label, 0.95)
        common_idx = persist.index.intersection(damped.index)
        res = st.paired(cfg, damped.loc[common_idx], persist.loc[common_idx])
        decision = paired_decision(res)
        h1_outcomes.append(decision)
        tests.append({"hipotese": "H1", "primario": "sim", "teste": "Δ = H30 amortecido − persistente (px, π^CLEAR)",
                      "grupo": label, "n": res["n"], "estatistica": res["diferenca"],
                      "IC95_inf": res["IC95_inf"], "IC95_sup": res["IC95_sup"], "p_t": res["p_t_pareado"],
                      "p_wilcoxon": res["p_wilcoxon"], "reducao_pct": res["reducao_pct"], "leitura_da_regra": decision})
        for col in ("mean_norm",):
            res_n = st.paired(cfg, series(label, 0.95, col).loc[common_idx], series(label, 1.0, col).loc[common_idx])
            tests.append({"hipotese": "H1", "primario": "não", "teste": "Δ H30 normalizado", "grupo": label,
                          "n": res_n["n"], "estatistica": res_n["diferenca"], "IC95_inf": res_n["IC95_inf"],
                          "IC95_sup": res_n["IC95_sup"], "p_t": res_n["p_t_pareado"],
                          "p_wilcoxon": res_n["p_wilcoxon"], "reducao_pct": res_n["reducao_pct"],
                          "leitura_da_regra": paired_decision(res_n)})
    if len(h1_outcomes) == 2:
        if all(o == "diferença < 0" for o in h1_outcomes):
            verdict = "confirmada"
        elif any(o == "diferença > 0" for o in h1_outcomes):
            verdict = "refutada"
        elif any(o == "diferença < 0" for o in h1_outcomes):
            verdict = "inconclusiva (parcial: só um rastreador)"
        else:
            verdict = "inconclusiva"
        tests.append({"hipotese": "H1", "primario": "regra", "teste": "regra pré-registrada", "grupo": "ambos",
                      "leitura_da_regra": verdict})

    h2a = []
    if not per_seq_a.empty:
        for label in ("baseline", "hit"):
            sel = per_seq_a[per_seq_a["rastreador"] == label]
            res = st.correlation(sel["IDF1"], sel["H30_norm_persist"])
            decision = corr_decision(res, alpha)
            h2a.append(decision)
            tests.append({"hipotese": "H2a", "primario": "sim", "teste": "Spearman IDF1 × H30 normalizado (persistente)",
                          "grupo": label, "n": res["n"], "estatistica": res["spearman_rho"], "p_t": res["spearman_p"],
                          "leitura_da_regra": decision, "pearson_r": res["pearson_r"], "pearson_p": res["pearson_p"]})
            res_px = st.correlation(sel["IDF1"], sel["H30_px_persist"])
            tests.append({"hipotese": "H2a", "primario": "não", "teste": "Spearman IDF1 × H30 px", "grupo": label,
                          "n": res_px["n"], "estatistica": res_px["spearman_rho"], "p_t": res_px["spearman_p"],
                          "leitura_da_regra": corr_decision(res_px, alpha)})
            res_ids = st.correlation(sel["IDs"] / sel["GT"], sel["H30_norm_persist"])
            tests.append({"hipotese": "H2a", "primario": "não", "teste": "Spearman IDs/GT × H30 normalizado",
                          "grupo": label, "n": res_ids["n"], "estatistica": res_ids["spearman_rho"],
                          "p_t": res_ids["spearman_p"], "leitura_da_regra": corr_decision(res_ids, alpha)})
        pooled = st.correlation(per_seq_a["IDF1"], per_seq_a["H30_norm_persist"])
        tests.append({"hipotese": "H2a", "primario": "não", "teste": "Spearman combinado (n=60, sem independência)",
                      "grupo": "baseline+hit", "n": pooled["n"], "estatistica": pooled["spearman_rho"],
                      "p_t": pooled["spearman_p"], "leitura_da_regra": corr_decision(pooled, alpha)})
        if len(h2a) == 2:
            verdict = ("confirmada" if all(d == "ρ < 0 significativa" for d in h2a) else
                       "refutada" if any(d == "ρ > 0 significativa" for d in h2a) else "inconclusiva")
            tests.append({"hipotese": "H2a", "primario": "regra", "teste": "regra pré-registrada", "grupo": "ambos",
                          "leitura_da_regra": verdict})

        base = per_seq_a[per_seq_a["rastreador"] == "baseline"].set_index("seq")
        hit = per_seq_a[per_seq_a["rastreador"] == "hit"].set_index("seq")
        idx = base.index.intersection(hit.index)
        for col, primary in (("H30_px_persist", "sim"), ("H30_norm_persist", "não"), ("H30_px_amort", "não")):
            res = st.paired(cfg, hit.loc[idx, col], base.loc[idx, col])
            decision = paired_decision(res)
            if primary == "sim":
                verdict = {"diferença < 0": "confirmada", "diferença > 0": "refutada"}.get(decision, "inconclusiva")
                decision = f"{decision} → {verdict}"
            tests.append({"hipotese": "H2b", "primario": primary, "teste": f"Δ = HIT − base ({col})", "grupo": "hit-base",
                          "n": res["n"], "estatistica": res["diferenca"], "IC95_inf": res["IC95_inf"],
                          "IC95_sup": res["IC95_sup"], "p_t": res["p_t_pareado"], "p_wilcoxon": res["p_wilcoxon"],
                          "reducao_pct": res["reducao_pct"], "leitura_da_regra": decision})
        h30_idf1 = per_seq_a[["seq", "rastreador", "IDF1", "IDs", "densidade_media", "H30_px_persist",
                              "H30_norm_persist"]].rename(columns={"H30_norm_persist": "H30_norm",
                                                                   "H30_px_persist": "H30_px"})
    else:
        h30_idf1 = pd.DataFrame()

    # ------------------------------------------------------------ Experimento B
    variants = [base_ref, hit_ref, hyb_name] + [v["name"] for v in cfg["sweeps"]["variants"]]
    b_data = {v: load_eval(dirs["exp_b"] / "eval" / v) for v in variants}
    b_frames = {v: load_frames(dirs["exp_b"] / "frames" / v) for v in variants}
    b_errors = pd.concat([d["errors"] for d in b_data.values() if not d["errors"].empty], ignore_index=True) \
        if any(not d["errors"].empty for d in b_data.values()) else pd.DataFrame()
    b_tracks = pd.concat([d["tracks"] for d in b_data.values() if not d["tracks"].empty], ignore_index=True) \
        if any(not d["tracks"].empty for d in b_data.values()) else pd.DataFrame()
    agg_b = aggregate_errors(cfg, b_errors, b_tracks)
    if not agg_b.empty:
        write_csv(agg_b, out / "exp_b_prediction_error.csv")

    overview = []
    for variant in variants:
        mot, frames = b_data[variant]["mot"], b_frames[variant]
        if mot.empty or frames.empty:
            continue
        per_seq_fps = 1000.0 / frames.groupby("seq")["time_ms"].mean()
        tot = {k: float(mot[f"mot_{k}"].sum()) for k in ("num_misses", "num_false_positives", "num_switches",
                                                          "num_objects", "num_predictions", "idtp")}
        row = {
            "variante": variant, "sequencias": int(len(mot)),
            "MOTA_%": 100 * (1 - (tot["num_misses"] + tot["num_false_positives"] + tot["num_switches"]) / tot["num_objects"]),
            "IDF1_%": 100 * 2 * tot["idtp"] / (tot["num_objects"] + tot["num_predictions"]),
            "IDs": int(tot["num_switches"]), "FP": int(tot["num_false_positives"]), "FN": int(tot["num_misses"]),
            "FPS_medio_seq": float(per_seq_fps.mean()),
            "FPS_global": 1000.0 / float(frames["time_ms"].mean()),
            "frac_quadros_com_deteccao": float(frames["detect"].mean()) if "detect" in frames else math.nan,
        }
        for damping, tag in ((1.0, "persist"), (0.95, "amort")):
            sel = agg_b[(agg_b["rotulo"] == variant) & (agg_b["correspondencia"] == "clear")
                        & (agg_b["gamma"] == damping) & (agg_b["H"] == 30)] if not agg_b.empty else pd.DataFrame()
            if len(sel):
                row[f"H30_px_{tag}"] = f"{fmt(sel['media_px'].iloc[0])} ± {fmt(sel['dp_px'].iloc[0])}"
                row[f"H30_norm_{tag}"] = fmt(sel["media_norm"].iloc[0], 3)
        overview.append(row)
    overview_df = pd.DataFrame(overview)
    if not overview_df.empty:
        write_csv(overview_df.round(4), out / "exp_b_overview.csv")
        md = ["# Experimento B: arquitetura híbrida no MOT", "",
              f"* `{base_ref}`: YOLO26n (pesos `{cfg['reference']['baseline']['weights']}`) + BoT-SORT, detecção em todo quadro.",
              f"* `{hit_ref}`: HIT-Tracker (HardMining + SAHI512 + GM + NSA), detecção em todo quadro.",
              f"* `{hyb_name}`: híbrido, detector `{cfg['weights'][cfg['hybrid']['detector_weights']]}`, "
              f"N_red = {cfg['hybrid']['n_red']}, {cfg['hybrid']['local_tracker'].upper()}, custo GM γ = "
              f"{cfg['hybrid']['association']['gamma']}.", "",
              "FPS = 1000 / tempo médio por quadro (sem decodificação do vídeo), medido nesta máquina e nesta sessão.", "",
              md_table(overview_df.round(2))]
        (out / "exp_b_overview.md").write_text("\n".join(md), encoding="utf-8")

    # por sequencia + H3
    fb, fh = b_frames[base_ref], b_frames[hyb_name]
    seq_rows = []
    if not fb.empty and not fh.empty:
        fps_b = 1000.0 / fb.groupby("seq")["time_ms"].mean()
        fps_h = 1000.0 / fh.groupby("seq")["time_ms"].mean()
        mot_b = b_data[base_ref]["mot"].set_index("seq")
        mot_h = b_data[hyb_name]["mot"].set_index("seq")
        det_h = fh.groupby("seq")["detect"].mean()
        for seq in fps_b.index.intersection(fps_h.index):
            h30 = per_seq_value(b_errors, hyb_name, "clear", 1.0, 30, "mean_px") if not b_errors.empty else pd.Series()
            seq_rows.append({"seq": seq, "densidade_media": mot_h.loc[seq, "densidade_media"] if seq in mot_h.index else math.nan,
                             "FPS_base": fps_b[seq], "FPS_hibrido": fps_h[seq], "speedup": fps_h[seq] / fps_b[seq],
                             "frac_deteccao_hibrido": det_h.get(seq, math.nan),
                             "MOTA_base": mot_b.loc[seq, "mot_mota"] if seq in mot_b.index else math.nan,
                             "MOTA_hibrido": mot_h.loc[seq, "mot_mota"] if seq in mot_h.index else math.nan,
                             "IDF1_base": mot_b.loc[seq, "mot_idf1"] if seq in mot_b.index else math.nan,
                             "IDF1_hibrido": mot_h.loc[seq, "mot_idf1"] if seq in mot_h.index else math.nan,
                             "H30_px_hibrido": float(h30.get(seq, math.nan))})
    per_seq_b = pd.DataFrame(seq_rows)
    if not per_seq_b.empty:
        write_csv(per_seq_b.round(4), out / "exp_b_per_sequence.csv")

    # FPS x densidade
    bins = [float(b) for b in cfg["analysis"]["density_bins"]]
    min_frames = int(cfg["analysis"]["min_frames_per_bin"])
    density_rows = []
    all_max = max([float(f["n_gt"].max()) for f in b_frames.values() if not f.empty] or [0.0])
    if all_max >= bins[-1]:
        bins = bins + [math.inf]
    labels = [f"{int(lo)}–{int(hi) - 1}" if math.isfinite(hi) else f"≥{int(lo)}" for lo, hi in zip(bins[:-1], bins[1:])]
    centers = [(lo + (hi if math.isfinite(hi) else lo + 10)) / 2 for lo, hi in zip(bins[:-1], bins[1:])]
    base_by_bin = None
    for variant in [base_ref, hit_ref, hyb_name] + [v["name"] for v in cfg["sweeps"]["variants"]]:
        frames = b_frames[variant]
        if frames.empty:
            continue
        cut = pd.cut(frames["n_gt"], bins=bins, right=False, labels=labels)
        grouped = frames.groupby(cut, observed=False).agg(quadros=("time_ms", "size"), tempo_ms=("time_ms", "mean"),
                                                          ativos=("n_active", "mean"))
        grouped["fps"] = 1000.0 / grouped["tempo_ms"]
        fps_by_label = {str(k): float(v) for k, v in zip(grouped.index, grouped["fps"])}
        if variant == base_ref:
            base_by_bin = fps_by_label
        for label, center in zip(labels, centers):
            r = grouped.loc[label] if label in fps_by_label else None
            if r is None:
                continue
            row = {"variante": variant, "faixa_alvos": label, "centro": center, "quadros": int(r["quadros"]),
                   "tempo_ms": float(r["tempo_ms"]), "fps": float(r["fps"]), "ativos_medio": float(r["ativos"])}
            if base_by_bin is not None and variant != base_ref:
                row["speedup"] = row["fps"] / base_by_bin.get(label, math.nan)
            density_rows.append(row)
    density = pd.DataFrame(density_rows)
    if not density.empty:
        density = density[density["quadros"] >= 1]
        write_csv(density.round(4), out / "fps_vs_density.csv")
        plotted = density[density["quadros"] >= min_frames]
        fig_fps_density(plotted, out / "fig_fps_vs_density.png",
                        [(base_ref, "linha de base (detecção em todo quadro)", C_BASE),
                         (hyb_name, f"híbrido CSRT, N_red = {cfg['hybrid']['n_red']}", C_HYB),
                         (hit_ref, "HIT-Tracker (detecção em todo quadro)", C_HIT)])

    if not per_seq_b.empty and not density.empty:
        hyb_bins = density[(density["variante"] == hyb_name) & (density["quadros"] >= min_frames)]
        premise = bool((hyb_bins["speedup"] > 1).any()) if "speedup" in hyb_bins else False
        inversion = premise and bool((hyb_bins["speedup"] < 1).any())
        res = st.correlation(per_seq_b["densidade_media"], per_seq_b["speedup"])
        if not premise:
            verdict = "refutada (premissa falsa: sem vantagem de custo em nenhuma faixa)"
        else:
            decision = corr_decision(res, alpha)
            verdict = {"ρ < 0 significativa": "confirmada", "ρ > 0 significativa": "refutada"}.get(decision, "inconclusiva")
        tests.append({"hipotese": "H3", "primario": "sim", "teste": "Spearman densidade × speedup (por sequência)",
                      "grupo": hyb_name, "n": res["n"], "estatistica": res["spearman_rho"], "p_t": res["spearman_p"],
                      "leitura_da_regra": f"premissa={'sim' if premise else 'não'}; inversão observada="
                                          f"{'sim' if inversion else 'não'} → {verdict}"})
        for variant, frames in ((base_ref, fb), (hyb_name, fh)):
            fit = st.linear_fit(frames["n_gt"], frames["time_ms"])
            tests.append({"hipotese": "H3", "primario": "não", "teste": "tempo_ms = a + b·n_gt (por quadro)",
                          "grupo": variant, "n": fit["n"], "estatistica": fit["inclinacao"],
                          "leitura_da_regra": f"b = {fmt(fit['inclinacao'], 3)} ms/alvo; "
                                                                           f"a = {fmt(fit['intercepto'], 1)} ms; R² = {fmt(fit['r2'], 3)}"})

    # ------------------------------------------------------------ H4
    ir = b_data[hyb_name]["drift"]
    vis = load_eval(dirs["h4_dut"] / "eval")
    vis_drift, vis_curve = vis["drift"], vis["drift_curve"]
    drift_rows = []
    if not ir.empty:
        drift_rows.append(ir.assign(dominio="track3_ir"))
    if not vis_drift.empty:
        drift_rows.append(vis_drift.assign(dominio="dut_visivel"))
    if drift_rows:
        drift = pd.concat(drift_rows, ignore_index=True)
        write_csv(drift.round(5), out / "h4_drift.csv")
        curves = []
        for domain, curve in (("track3_ir", b_data[hyb_name]["drift_curve"]), ("dut_visivel", vis_curve)):
            if curve.empty:
                continue
            per_age = curve.groupby("age").agg(sequencias=("seq", "nunique"), pontos=("n", "sum"),
                                               e_norm_media=("mean_e_norm", "mean"), iou_media=("mean_iou", "mean"),
                                               frac_iou50=("frac_iou50", "mean")).reset_index()
            per_age.insert(0, "dominio", domain)
            curves.append(per_age)
        curves_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
        if not curves_df.empty:
            write_csv(curves_df.round(5), out / "h4_drift_curve.csv")
            fig_drift(curves_df, out / "fig_drift_curve.png", int(cfg["hybrid"]["n_red"]))
        summary_rows = []
        for domain, grp in drift.groupby("dominio"):
            summary_rows.append({"dominio": domain, "sequencias": len(grp),
                                 "taxa_deriva_mediana": grp["taxa_deriva"].median(),
                                 "taxa_deriva_media": grp["taxa_deriva"].mean(),
                                 "taxa_falha_mediana": grp["taxa_falha"].median(),
                                 "e_norm_a0_media": grp["e_norm_a0"].mean(), "e_norm_afim_media": grp["e_norm_afim"].mean(),
                                 "iou_afim_media": grp["iou_afim"].mean(),
                                 "segmentos_ancorados": int(grp["segmentos_ancorados"].sum())})
        (out / "h4_drift.md").write_text("# H4: deriva do CSRT entre redetecções\n\n" +
                                         md_table(pd.DataFrame(summary_rows).round(4)), encoding="utf-8")
    if not ir.empty and not vis_drift.empty:
        res = st.mann_whitney_greater(ir["taxa_deriva"], vis_drift["taxa_deriva"])
        if res["p_maior"] < alpha and res["mediana_x"] > res["mediana_y"]:
            verdict = "confirmada"
        elif res["p_menor"] < alpha:
            verdict = "refutada"
        else:
            verdict = "inconclusiva"
        tests.append({"hipotese": "H4", "primario": "sim", "teste": "Mann-Whitney unilateral: deriva IR > visível",
                      "grupo": "track3_ir x dut_visivel", "n": f"{res['n_x']} x {res['n_y']}", "estatistica": res["U"],
                      "p_t": res["p_maior"], "p_wilcoxon": res["p_menor"],
                      "leitura_da_regra": f"mediana IR {fmt(res['mediana_x'], 5)} x visível {fmt(res['mediana_y'], 5)} → {verdict}"})
        for metric in ("taxa_falha", "iou_afim"):
            res_m = st.mann_whitney_greater(ir[metric], vis_drift[metric])
            tests.append({"hipotese": "H4", "primario": "não", "teste": f"Mann-Whitney {metric} (IR > visível)",
                          "grupo": "track3_ir x dut_visivel", "n": f"{res_m['n_x']} x {res_m['n_y']}",
                          "estatistica": res_m["U"], "p_t": res_m["p_maior"], "p_wilcoxon": res_m["p_menor"],
                          "leitura_da_regra": f"mediana IR {fmt(res_m['mediana_x'], 4)} x visível {fmt(res_m['mediana_y'], 4)}"})
    elif not ir.empty:
        tests.append({"hipotese": "H4", "primario": "sim", "teste": "Mann-Whitney unilateral: deriva IR > visível",
                      "grupo": "—", "leitura_da_regra": "inconclusiva (sem referência visível do DUT)"})

    # H1 descritivo no hibrido
    if not b_errors.empty:
        persist, damped = per_seq_value(b_errors, hyb_name, "clear", 1.0), per_seq_value(b_errors, hyb_name, "clear", 0.95)
        idx = persist.index.intersection(damped.index)
        if len(idx) >= 3:
            res = st.paired(cfg, damped.loc[idx], persist.loc[idx])
            tests.append({"hipotese": "H1", "primario": "não", "teste": "Δ H30 amortecido − persistente (px)",
                          "grupo": hyb_name, "n": res["n"], "estatistica": res["diferenca"], "IC95_inf": res["IC95_inf"],
                          "IC95_sup": res["IC95_sup"], "p_t": res["p_t_pareado"], "p_wilcoxon": res["p_wilcoxon"],
                          "reducao_pct": res["reducao_pct"], "leitura_da_regra": paired_decision(res)})

    tests_df = pd.DataFrame(tests)
    if not tests_df.empty:
        write_csv(tests_df, out / "hypothesis_tests.csv")
        show = tests_df.copy()
        if "n" in show:
            show["n"] = show["n"].apply(lambda v: str(int(v)) if isinstance(v, (int, float, np.floating))
                                        and math.isfinite(v) else (v if isinstance(v, str) else "–"))
        for col in ("estatistica", "IC95_inf", "IC95_sup", "p_t", "p_wilcoxon", "reducao_pct"):
            if col in show:
                show[col] = show[col].apply(lambda v: fmt(v, 4) if isinstance(v, (int, float, np.floating)) else (v if isinstance(v, str) else "–"))
        cols = [c for c in ("hipotese", "primario", "teste", "grupo", "n", "estatistica", "IC95_inf", "IC95_sup",
                            "p_t", "p_wilcoxon", "reducao_pct", "leitura_da_regra") if c in show]
        (out / "hypothesis_tests.md").write_text(
            "# Testes pré-registrados (HYPOTHESES.md, seção 3)\n\n"
            "`leitura_da_regra` é a aplicação mecânica da regra; a interpretação é feita à parte.\n"
            "Colunas: `p_t` = p do t pareado / Spearman / Mann-Whitney (IR > visível); "
            "`p_wilcoxon` = Wilcoxon / Mann-Whitney oposto (IR < visível).\n\n" + md_table(show[cols].fillna("–")),
            encoding="utf-8")

    # H30 x IDF1 (inclui o hibrido)
    if not per_seq_b.empty:
        hyb_rows = per_seq_b[["seq", "IDF1_hibrido", "densidade_media"]].rename(columns={"IDF1_hibrido": "IDF1"})
        hyb_rows["rastreador"] = hyb_name
        hyb_rows["H30_norm"] = hyb_rows["seq"].map(per_seq_value(b_errors, hyb_name, "clear", 1.0, 30, "mean_norm"))
        hyb_rows["H30_px"] = hyb_rows["seq"].map(per_seq_value(b_errors, hyb_name, "clear", 1.0, 30, "mean_px"))
        h30_idf1 = pd.concat([h30_idf1, hyb_rows], ignore_index=True)
    if not h30_idf1.empty:
        write_csv(h30_idf1.round(5), out / "h30_vs_idf1.csv")
        fig_h30_idf1(h30_idf1, out / "fig_h30_vs_idf1.png",
                     [("baseline", "linha de base", C_BASE), ("hit", "HIT-Tracker", C_HIT),
                      (hyb_name, "híbrido", C_HYB)])

    # ------------------------------------------------------------ verificacoes e execucao
    checks_md = ["# Verificações", ""]
    sel_path = dirs["setup"] / "trajectory_selection.json"
    if sel_path.exists():
        sel = read_json(sel_path)
        checks_md += ["## Trajetórias usadas no Experimento A", ""]
        rows = []
        for tracker, entry in sel.get("rastreadores", {}).items():
            for cand in entry.get("candidatos", []):
                agg = cand.get("overall") or {}
                rows.append({"rastreador": tracker, "candidato": cand.get("sufixo"),
                             "arquivos": f"{cand.get('arquivos')}/{cand.get('esperados')}",
                             "MOTA_%": fmt(100 * agg["mota"], 2) if agg else "–",
                             "IDF1_%": fmt(100 * agg["idf1"], 2) if agg else "–",
                             "IDs": int(agg["num_switches"]) if agg else "–",
                             "= publicado por seq.": cand.get("confere_por_sequencia", "–"),
                             "sha256 = manifesto": cand.get("igual_ao_manifesto", "–"),
                             "selecionado": "sim" if cand.get("sufixo") == entry.get("selecionado") else ""})
        checks_md.append(md_table(pd.DataFrame(rows)))
    smoke_checks = dirs["smoke"] / "checks.json"
    if smoke_checks.exists():
        checks_md += ["## Teste rápido", "", md_table(pd.DataFrame(read_json(smoke_checks)["verificacoes"]))]
    status_path = dirs["setup"] / "setup_status.json"
    status = read_json(status_path) if status_path.exists() else {}
    if status:
        checks_md += ["## check_setup", "", f"* sem subamostragem: {status.get('sem_subamostragem')}",
                      f"* bloqueantes: {len(status.get('bloqueantes', []))}; avisos: {len(status.get('avisos', []))}", ""]
        for item in status.get("avisos", []):
            checks_md.append(f"  * aviso: {item}")
    sige_json = sorted((dirs["h4_dut"] / "eval").glob("*_sige.json")) if (dirs["h4_dut"] / "eval").is_dir() else []
    if sige_json:
        dut = pd.DataFrame([read_json(p) for p in sige_json])
        checks_md += ["", "## Híbrido do SIGE no DUT (reprodução neste ambiente)", "",
                      f"* sequências: {len(dut)}; SR@0.5 médio {fmt(dut['sige_SR@0.5'].mean(), 3)} (artigo: 0,811); "
                      f"SA {fmt(dut['sige_SA'].mean(), 3)} (artigo: 0,789)",
                      f"* H30 persistente {fmt(dut['g1.00_H30_media'].mean())} ± {fmt(dut['g1.00_H30_media'].std())} px "
                      f"(artigo: 139,9 ± 97,4); amortecido {fmt(dut['g0.95_H30_media'].mean())} px (artigo: 109,1)",
                      f"* FPS médio {fmt(dut['fps'].mean())} (artigo: 66,4 no M2)"]
        write_csv(dut.round(5), out / "h4_dut_sige_per_video.csv")
    (out / "validation_checks.md").write_text("\n".join(checks_md) + "\n", encoding="utf-8")

    env = status.get("ambiente", {})
    write_json(out / "environment.json", {"ambiente": env, "config_sha256": cfg["_config_sha256"],
                                          "taxas": status.get("taxas"), "botsort": status.get("botsort"),
                                          "kalman_sige": status.get("kalman_sige")})
    for source, name in ((MP_DIR / "check_setup.txt", "check_setup.txt"), (Path(cfg["_config_path"]), "config_used.yaml"),
                         (rpath(cfg["paths"]["logs"]) / "stage_times.csv", "stage_times.csv"),
                         (dirs["smoke"] / "estimate.json", "smoke_estimate.json"),
                         (dirs["smoke"] / "smoke_report.md", "smoke_report.md")):
        if source.exists():
            shutil.copyfile(source, out / name)

    # ------------------------------------------------------------ summary.md
    lines = ["# Resumo: predição de trajetória no cenário MOT", "",
             f"Gerado por `make_summary.py`. Configuração sha256 `{cfg['_config_sha256'][:16]}`.", ""]
    if env:
        pk = env.get("pacotes", {})
        lines += [f"* Máquina: {env.get('cpu', env.get('plataforma'))}; Python {env.get('python')}; "
                  f"ultralytics {pk.get('ultralytics')}; torch {pk.get('torch')}; OpenCV {env.get('cv2_version')}", ""]
    if not agg_a.empty:
        lines += ["## Experimento A (trajetórias salvas; π^CLEAR; média ± dp entre sequências)", "",
                  md_table(compact_error_table(agg_a, ["baseline", "hit"], "px")),
                  md_table(compact_error_table(agg_a, ["baseline", "hit"], "norm"))]
    if not overview_df.empty:
        lines += ["## Experimento B", "", md_table(overview_df.round(2))]
    if not tests_df.empty:
        rules = tests_df[tests_df["primario"].isin(["sim", "regra"])][["hipotese", "primario", "teste", "grupo",
                                                                        "leitura_da_regra"]]
        lines += ["## Testes primários (leitura mecânica da regra)", "", md_table(rules.fillna("–"))]
    if notes:
        lines += ["## Notas", ""] + [f"* {n}" for n in notes]
    lines += ["", "Arquivos: ver a lista no topo de `make_summary.py`."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    total_kb = sum(p.stat().st_size for p in out.glob("*") if p.is_file()) / 1024
    log(f"results/summary pronto: {len(list(out.glob('*')))} arquivos, {total_kb:.0f} KB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
