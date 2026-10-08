"""Erro de predicao de trajetoria com identidade (Experimento A).

O filtro e o do SIGE, sem alteracao: uav_common.init_kalman e uav_common.forward
sao importados; o laco de kalman_states() e copia linha a linha do laco de
uav_common.prediction_errors (publicar_github/.../src/uav_common.py). A UNICA
mudanca para o MOT esta fora do filtro: o alvo de referencia em t+H vem da
correspondencia trilha->alvo (pi_CLEAR ou pi_ID) em vez de ser o alvo unico.

fidelity_check() prova, no teste rapido, que kalman_states() + a busca do alvo
unico reproduz uav_common.prediction_errors bit a bit.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Mesma matriz local de uav_common.prediction_errors
TRANSITION = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

MODES = ("clear", "idf1")


def kalman_states(uc, centers: dict, damping: float):
    """Copia do laco de uav_common.prediction_errors; devolve (quadro, estado corrigido)."""
    kalman = None
    for frame in sorted(centers):
        cx, cy = centers[frame]
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        if kalman is None:
            kalman = uc.init_kalman(cx, cy)
            continue

        kalman.predict()
        kalman.correct(measurement)
        if damping != 1.0:
            damped = kalman.statePost.copy()
            damped[2, 0] *= damping
            damped[3, 0] *= damping
            kalman.statePost = damped

        yield frame, kalman.statePost.copy()


def sige_equivalent_errors(uc, centers: dict, gt_centers: dict, damping: float, horizons) -> dict:
    """kalman_states + alvo unico: deve ser identico a uav_common.prediction_errors."""
    errors = {h: [] for h in horizons}
    for frame, state in kalman_states(uc, centers, damping):
        for horizon in horizons:
            target = gt_centers.get(frame + horizon)
            if target is None:
                continue
            predicted = uc.forward(state, horizon, TRANSITION, damping)
            errors[horizon].append(float(np.hypot(predicted[0, 0] - target[0], predicted[1, 0] - target[1])))
    return errors


def fidelity_check(uc, centers: dict, gt_centers: dict, horizons, dampings) -> dict:
    """Compara a reimplementacao com uav_common.prediction_errors (deve ser igual)."""
    result = {"igual": True, "max_dif_abs": 0.0, "n_valores": 0}
    for damping in dampings:
        reference = uc.prediction_errors(centers, gt_centers, damping, horizons)
        mine = sige_equivalent_errors(uc, centers, gt_centers, damping, horizons)
        for horizon in horizons:
            a = np.asarray(reference.get(horizon, []), dtype=float)
            b = np.asarray(mine.get(horizon, []), dtype=float)
            if a.shape != b.shape:
                result["igual"] = False
                result["max_dif_abs"] = float("inf")
                continue
            result["n_valores"] += int(a.size)
            if a.size:
                diff = float(np.max(np.abs(a - b)))
                result["max_dif_abs"] = max(result["max_dif_abs"], diff)
                if not np.array_equal(a, b):
                    result["igual"] = False
    return result


def track_centers(pred_df: pd.DataFrame) -> dict:
    """{id: {quadro: (cx, cy)}}; como uav_common.read_predictions, o 1o registro do quadro vale."""
    out: dict[int, dict[int, tuple[float, float]]] = {}
    for hid, frame, x, y, w, h in zip(pred_df["id"], pred_df["frame"], pred_df["x"], pred_df["y"],
                                      pred_df["w"], pred_df["h"]):
        centers = out.setdefault(int(hid), {})
        centers.setdefault(int(frame), (float(x) + float(w) / 2.0, float(y) + float(h) / 2.0))
    return out


def gap_stats(tracks: dict) -> dict:
    """Quantos passos do filtro vem depois de uma lacuna (> 1 quadro sem caixa)."""
    steps = 0
    after_gap = 0
    gap_lengths = []
    lengths = []
    for centers in tracks.values():
        frames = sorted(centers)
        lengths.append(len(frames))
        for prev, cur in zip(frames[:-1], frames[1:]):
            steps += 1
            if cur - prev > 1:
                after_gap += 1
                gap_lengths.append(cur - prev - 1)
    return {
        "n_trilhas": len(tracks),
        "passos_kalman": steps,
        "passos_apos_lacuna": after_gap,
        "fracao_passos_apos_lacuna": after_gap / steps if steps else float("nan"),
        "lacuna_mediana_quadros": float(np.median(gap_lengths)) if gap_lengths else 0.0,
        "trilha_mediana_quadros": float(np.median(lengths)) if lengths else 0.0,
    }


def pair_errors(uc, pred_df: pd.DataFrame, gt_index, corr: dict, horizons, dampings):
    """Erro com identidade para todos os pares (trilha, t, H, gamma, modo).

    Devolve (pares: DataFrame, contagens: DataFrame). Contagens por (modo, gamma, H):
    passos, sem_assoc, ausente, validos.
    """
    tracks = track_centers(pred_df)
    clear = corr["clear"]
    idmap = corr["idf1"]
    by_id = gt_index.by_id

    rec_hid, rec_frame, rec_h, rec_damp, rec_mode, rec_g, rec_e, rec_en = ([] for _ in range(8))
    counts = {}
    for mode in MODES:
        for damping in dampings:
            for horizon in horizons:
                counts[(mode, damping, horizon)] = {"passos": 0, "sem_assoc": 0, "ausente": 0, "validos": 0}

    for hid in sorted(tracks):
        centers = tracks[hid]
        g_id = idmap.get(hid)
        for damping in dampings:
            for frame, state in kalman_states(uc, centers, damping):
                g_clear = clear.get((frame, hid))
                for horizon in horizons:
                    targets = []
                    for mode, g in (("clear", g_clear), ("idf1", g_id)):
                        cell = counts[(mode, damping, horizon)]
                        cell["passos"] += 1
                        if g is None:
                            cell["sem_assoc"] += 1
                            continue
                        target = by_id.get(g, {}).get(frame + horizon)
                        if target is None:
                            cell["ausente"] += 1
                            continue
                        cell["validos"] += 1
                        targets.append((mode, g, target))
                    if not targets:
                        continue
                    predicted = uc.forward(state, horizon, TRANSITION, damping)
                    for mode, g, (tx, ty, diag) in targets:
                        error = float(np.hypot(predicted[0, 0] - tx, predicted[1, 0] - ty))
                        rec_hid.append(hid)
                        rec_frame.append(frame)
                        rec_h.append(horizon)
                        rec_damp.append(damping)
                        rec_mode.append(mode)
                        rec_g.append(g)
                        rec_e.append(error)
                        rec_en.append(error / diag if diag > 0 else math.nan)

    pairs = pd.DataFrame({
        "hid": np.asarray(rec_hid, dtype=np.int64),
        "frame": np.asarray(rec_frame, dtype=np.int64),
        "H": np.asarray(rec_h, dtype=np.int64),
        "damping": np.asarray(rec_damp, dtype=np.float64),
        "mode": pd.Categorical(rec_mode, categories=list(MODES)),
        "g": np.asarray(rec_g, dtype=np.int64),
        "e_px": np.asarray(rec_e, dtype=np.float64),
        "e_norm": np.asarray(rec_en, dtype=np.float64),
    })
    count_rows = [{"mode": m, "damping": d, "H": h, **v} for (m, d, h), v in counts.items()]
    return pairs, pd.DataFrame(count_rows), gap_stats(tracks)


def _std(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if len(series) > 1 else float("nan")


def aggregate_sequence(pairs: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (modo, gamma, H): media/dp/mediana/p95 em px e normalizado + contagens."""
    rows = []
    for (mode, damping, horizon), cell in counts.groupby(["mode", "damping", "H"], sort=True):
        sel = pairs[(pairs["mode"] == mode) & (pairs["damping"] == damping) & (pairs["H"] == horizon)]
        e = sel["e_px"]
        en = sel["e_norm"].dropna()
        rows.append({
            "mode": mode, "damping": float(damping), "H": int(horizon),
            "n": int(len(sel)),
            "mean_px": float(e.mean()) if len(e) else math.nan,
            "std_px": _std(e),
            "median_px": float(e.median()) if len(e) else math.nan,
            "p95_px": float(np.percentile(e, 95)) if len(e) else math.nan,
            "mean_norm": float(en.mean()) if len(en) else math.nan,
            "std_norm": _std(en),
            "median_norm": float(en.median()) if len(en) else math.nan,
            "n_trilhas": int(sel["hid"].nunique()),
            "passos": int(cell["passos"].sum()),
            "sem_assoc": int(cell["sem_assoc"].sum()),
            "ausente": int(cell["ausente"].sum()),
            "validos": int(cell["validos"].sum()),
        })
    return pd.DataFrame(rows)


def aggregate_tracks(pairs: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (modo, gamma, H, trilha)."""
    if pairs.empty:
        return pd.DataFrame(columns=["mode", "damping", "H", "hid", "n", "mean_px", "std_px",
                                     "mean_norm", "std_norm"])
    grouped = pairs.groupby(["mode", "damping", "H", "hid"], observed=True, sort=True)
    table = grouped.agg(n=("e_px", "size"), mean_px=("e_px", "mean"), std_px=("e_px", "std"),
                        mean_norm=("e_norm", "mean"), std_norm=("e_norm", "std")).reset_index()
    table["mode"] = table["mode"].astype(str)
    return table


def common_support(pairs_a: pd.DataFrame, pairs_b: pd.DataFrame, mode: str = "clear") -> pd.DataFrame:
    """Medias de A e B sobre os mesmos (g, t, H, gamma)."""
    keys = ["damping", "H", "g", "frame"]
    cols = keys + ["e_px", "e_norm"]
    a = pairs_a[pairs_a["mode"] == mode][cols]
    b = pairs_b[pairs_b["mode"] == mode][cols]
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
    rows = []
    for (damping, horizon), grp in merged.groupby(["damping", "H"], sort=True):
        rows.append({
            "damping": float(damping), "H": int(horizon), "n_comum": int(len(grp)),
            "mean_px_a": float(grp["e_px_a"].mean()), "mean_px_b": float(grp["e_px_b"].mean()),
            "mean_norm_a": float(grp["e_norm_a"].mean()), "mean_norm_b": float(grp["e_norm_b"].mean()),
        })
    return pd.DataFrame(rows)
