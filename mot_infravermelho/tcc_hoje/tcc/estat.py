"""Estatistica unica do tcc_hoje.

* IC por bootstrap BCa (scipy.stats.bootstrap), 10.000 reamostragens, semente fixa
  (a MESMA semente em cada teste, entao o resultado nao depende da ordem).
* Wilcoxon pareado (bicaudal) + correlacao rank-biserial.
* Holm por familia; IC de Bonferroni com nivel 1 - alpha/m.
* Bootstrap por sequencia (cluster) com BCa para inclinacoes de regressao.
Nenhum IC baseado em t.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import stats
from scipy.special import ndtr, ndtri

NAN = float("nan")


def _bootstrap(data, statistic, conf, seed, n_resamples, paired=False):
    kwargs = dict(n_resamples=n_resamples, confidence_level=conf, method="BCa",
                  vectorized=True, paired=paired, axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = stats.bootstrap(data, statistic, rng=np.random.default_rng(seed), **kwargs)
        except TypeError:   # SciPy antigo: random_state
            res = stats.bootstrap(data, statistic, random_state=np.random.default_rng(seed), **kwargs)
    lo, hi = res.confidence_interval
    return float(lo), float(hi)


def _mean(x, axis=-1):
    return np.mean(x, axis=axis)


def bca_mean(values, conf=0.95, seed=2026, n_resamples=10000):
    """IC BCa da media. Devolve (inf, sup, observacao)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return NAN, NAN, f"n={x.size} < 3"
    if np.allclose(x, x[0]):
        return float(x[0]), float(x[0]), "valores constantes"
    try:
        lo, hi = _bootstrap((x,), _mean, conf, seed, n_resamples)
    except Exception as exc:  # noqa: BLE001
        return NAN, NAN, f"BCa falhou: {exc}"
    return lo, hi, ""


def _pearson_v(x, y, axis=-1):
    xm = x - np.mean(x, axis=axis, keepdims=True)
    ym = y - np.mean(y, axis=axis, keepdims=True)
    num = np.sum(xm * ym, axis=axis)
    den = np.sqrt(np.sum(xm * xm, axis=axis) * np.sum(ym * ym, axis=axis))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def _spearman_v(x, y, axis=-1):
    return _pearson_v(stats.rankdata(x, axis=axis), stats.rankdata(y, axis=axis), axis=axis)


def _finite_pairs(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def correlacao(x, y, kind="pearson", conf=0.95, seed=2026, n_resamples=10000):
    """r (ou rho), p e IC BCa pareado. Devolve dict."""
    x, y = _finite_pairs(x, y)
    out = {"n": int(x.size), "r": NAN, "p": NAN, "ic_inf": NAN, "ic_sup": NAN, "obs": ""}
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        out["obs"] = "n < 3 ou variavel constante"
        return out
    if kind == "pearson":
        res = stats.pearsonr(x, y)
        fn = _pearson_v
    else:
        res = stats.spearmanr(x, y)
        fn = _spearman_v
    out["r"] = float(res.statistic if hasattr(res, "statistic") else res[0])
    out["p"] = float(res.pvalue if hasattr(res, "pvalue") else res[1])
    if x.size >= 4:
        try:
            out["ic_inf"], out["ic_sup"] = _bootstrap((x, y), fn, conf, seed, n_resamples, paired=True)
        except Exception as exc:  # noqa: BLE001
            out["obs"] = f"BCa falhou: {exc}"
    else:
        out["obs"] = "n < 4: sem IC"
    return out


def wilcoxon_rb(diffs):
    """Wilcoxon pareado bicaudal (zero_method='wilcox') + rank-biserial."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    out = {"n": int(d.size), "n_nao_nulos": int(nz.size), "W": NAN, "p": NAN, "rank_biserial": NAN, "obs": ""}
    if nz.size < 1:
        out["obs"] = "todas as diferencas sao nulas"
        return out
    ranks = stats.rankdata(np.abs(nz))
    r_pos = float(ranks[nz > 0].sum())
    r_neg = float(ranks[nz < 0].sum())
    out["rank_biserial"] = (r_pos - r_neg) / (r_pos + r_neg)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        out["W"] = float(res.statistic)
        out["p"] = float(res.pvalue)
    except Exception as exc:  # noqa: BLE001
        out["obs"] = f"Wilcoxon falhou: {exc}"
    return out


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    adj = np.full(p.shape, NAN)
    finite = np.where(np.isfinite(p))[0]
    m = finite.size
    order = finite[np.argsort(p[finite], kind="mergesort")]
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adj[idx] = running
    return adj


# ---------------------------------------------------------------------------
# Regressao tempo ~ alvos com bootstrap por sequencia (cluster), BCa
# ---------------------------------------------------------------------------

def suficientes(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.array([x.size, x.sum(), y.sum(), (x * x).sum(), (x * y).sum(), (y * y).sum()], dtype=float)


def _slope(S):
    """S: (..., 6) somas [n, sx, sy, sxx, sxy, syy] -> inclinacao OLS."""
    n, sx, sy, sxx, sxy = S[..., 0], S[..., 1], S[..., 2], S[..., 3], S[..., 4]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sxy - sx * sy / n) / (sxx - sx * sx / n)


def ols_from_S(S):
    n, sx, sy, sxx, sxy, syy = S
    b = _slope(np.asarray(S))
    a = (sy - b * sx) / n
    ss_tot = syy - sy * sy / n
    ss_res = syy - a * sy - b * sxy
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else NAN
    return float(a), float(b), float(r2)


def _bca_from_draws(theta_hat, draws, jack, conf):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size < 10 or not np.isfinite(theta_hat):
        return NAN, NAN
    prop = (np.sum(draws < theta_hat) + 0.5 * np.sum(draws == theta_hat)) / draws.size
    prop = min(max(prop, 1.0 / (draws.size + 1)), 1.0 - 1.0 / (draws.size + 1))
    z0 = ndtri(prop)
    jack = np.asarray(jack, dtype=float)
    jack = jack[np.isfinite(jack)]
    if jack.size >= 2:
        dev = jack.mean() - jack
        den = 6.0 * (np.sum(dev ** 2) ** 1.5)
        a = float(np.sum(dev ** 3) / den) if den > 0 else 0.0
    else:
        a = 0.0
    alpha = 1.0 - conf
    out = []
    for zq in (ndtri(alpha / 2.0), ndtri(1.0 - alpha / 2.0)):
        level = ndtr(z0 + (z0 + zq) / (1.0 - a * (z0 + zq)))
        out.append(float(np.quantile(draws, min(max(level, 0.0), 1.0))))
    return out[0], out[1]


def cluster_slopes(S_by_method: dict, conf_levels=(0.95,), seed=2026, n_resamples=10000,
                   diffs=()):
    """S_by_method: {metodo: array (K, 6)} com as MESMAS K sequencias, na mesma ordem.

    Reamostra sequencias com reposicao (mesmo sorteio para todos os metodos, entao as
    diferencas sao pareadas). Devolve {nome: {estimativa, ic: {nivel: (inf, sup)}, p}}.
    """
    methods = list(S_by_method)
    K = next(iter(S_by_method.values())).shape[0]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(K, np.full(K, 1.0 / K), size=n_resamples).astype(float)
    full = np.ones(K)
    results = {}
    draws_by = {}
    jack_by = {}
    hat_by = {}
    for m in methods:
        S = np.asarray(S_by_method[m], dtype=float)
        hat_by[m] = float(_slope(full @ S))
        draws_by[m] = _slope(counts @ S)
        jack_by[m] = np.array([_slope((full - np.eye(K)[i]) @ S) for i in range(K)]) if K >= 3 else np.array([])
    targets = [(m, hat_by[m], draws_by[m], jack_by[m]) for m in methods]
    for a, b in diffs:
        targets.append((f"{a} - {b}", hat_by[a] - hat_by[b], draws_by[a] - draws_by[b],
                        (jack_by[a] - jack_by[b]) if K >= 3 else np.array([])))
    for name, hat, draws, jack in targets:
        finite = draws[np.isfinite(draws)]
        if finite.size:
            le = (np.sum(finite <= 0) + 1) / (finite.size + 1)
            ge = (np.sum(finite >= 0) + 1) / (finite.size + 1)
            p = float(min(1.0, 2.0 * min(le, ge)))
        else:
            p = NAN
        ic = {}
        for level in conf_levels:
            ic[level] = _bca_from_draws(hat, draws, jack, level) if K >= 3 else (NAN, NAN)
        results[name] = {"estimativa": hat, "ic": ic, "p": p, "K": K}
    return results


def fmt(value, nd=2):
    """Numero em pt-BR (virgula decimal) para tabelas Markdown."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    if not math.isfinite(v):
        return "—"
    return f"{v:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
