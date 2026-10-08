"""Testes estatisticos pre-registrados (HYPOTHESES.md, secao 3).

bootstrap_ci e o de publicar_github/.../src/05_avaliar.py (mesma semente 2026);
ci95_t e o de uav_common.py. Nada aqui decide a interpretacao: as funcoes
devolvem numeros e o resultado MECANICO da regra pre-registrada.
"""

from __future__ import annotations

import math

import numpy as np

from .common import sige_avaliar, uav_common


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def bootstrap_ci(cfg: dict, values, n_boot: int | None = None, seed: int | None = None):
    n_boot = int(n_boot or cfg["prediction"]["bootstrap_n"])
    seed = int(seed if seed is not None else cfg["prediction"]["bootstrap_seed"])
    return sige_avaliar(cfg).bootstrap_ci(np.asarray(values, dtype=float), n_boot=n_boot, seed=seed)


def ci95_t(cfg: dict, values):
    return uav_common(cfg).ci95_t(list(values))


def paired(cfg: dict, a, b) -> dict:
    """Diferenca pareada a - b por sequencia: media, IC95% bootstrap, t pareado, Wilcoxon."""
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    out = {"n": int(a.size), "media_a": math.nan, "media_b": math.nan, "diferenca": math.nan,
           "reducao_pct": math.nan, "IC95_inf": math.nan, "IC95_sup": math.nan,
           "p_t_pareado": math.nan, "p_wilcoxon": math.nan}
    if a.size < 3:
        return out
    diff = a - b
    low, high = bootstrap_ci(cfg, diff)
    out.update({
        "media_a": float(a.mean()), "media_b": float(b.mean()), "diferenca": float(diff.mean()),
        "reducao_pct": float((1 - a.mean() / b.mean()) * 100) if b.mean() else math.nan,
        "IC95_inf": low, "IC95_sup": high,
        "p_t_pareado": float(stats.ttest_rel(a, b).pvalue),
    })
    try:
        out["p_wilcoxon"] = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        pass
    return out


def correlation(x, y) -> dict:
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    out = {"n": int(x.size), "spearman_rho": math.nan, "spearman_p": math.nan,
           "pearson_r": math.nan, "pearson_p": math.nan}
    if x.size < 4 or np.all(x == x[0]) or np.all(y == y[0]):
        return out
    rho = stats.spearmanr(x, y)
    r = stats.pearsonr(x, y)
    out.update({"spearman_rho": float(rho.statistic if hasattr(rho, "statistic") else rho[0]),
                "spearman_p": float(rho.pvalue),
                "pearson_r": float(r.statistic if hasattr(r, "statistic") else r[0]),
                "pearson_p": float(r.pvalue)})
    return out


def mann_whitney_greater(x, y) -> dict:
    """Unilateral: x > y, e o oposto (x < y), ambos reportados."""
    from scipy import stats

    x = _finite(x)
    y = _finite(y)
    out = {"n_x": int(x.size), "n_y": int(y.size), "mediana_x": math.nan, "mediana_y": math.nan,
           "U": math.nan, "p_maior": math.nan, "p_menor": math.nan}
    if x.size < 3 or y.size < 3:
        return out
    greater = stats.mannwhitneyu(x, y, alternative="greater")
    less = stats.mannwhitneyu(x, y, alternative="less")
    out.update({"mediana_x": float(np.median(x)), "mediana_y": float(np.median(y)),
                "U": float(greater.statistic), "p_maior": float(greater.pvalue),
                "p_menor": float(less.pvalue)})
    return out


def linear_fit(x, y) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or np.all(x == x[0]):
        return {"n": int(x.size), "intercepto": math.nan, "inclinacao": math.nan, "r2": math.nan}
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"n": int(x.size), "intercepto": float(intercept), "inclinacao": float(slope),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else math.nan}
