"""
Filtro pós-tracking baseado em recorrência ESPACIAL (não por ID
individual): divide o frame em um grid fixo e conta, ao longo de toda
a sequência, em quantos frames distintos cada célula recebeu ao menos
uma detecção -- independente de qual track ID gerou aquela detecção.

Células com frequência de ocupação muito alta são tratadas como
prováveis estruturas de fundo (janelas, portas), mesmo que nenhum
track individual seja longo o suficiente para ser pego por um filtro
temporal-por-ID (ver static_track_filter.py -- descartado após
diagnóstico mostrar tracks fragmentados/curtos nas sequências
problemáticas).
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def _clean_detections(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas com valores inválidos (NaN/inf) nas colunas
    essenciais, resultado de eventuais linhas malformadas nos arquivos
    de tracking. Loga quantas linhas foram descartadas, se houver."""
    required_cols = ["frame", "id", "x", "y", "w", "h"]
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols).copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"  [AVISO] {dropped} linha(s) inválida(s) descartada(s) de {before} totais.")
    return df


def build_occupancy_map(df: pd.DataFrame, frame_w: int, frame_h: int, cell_size: int) -> np.ndarray:
    n_cols = frame_w // cell_size + 1
    n_rows = frame_h // cell_size + 1

    df = _clean_detections(df)
    df["cx"] = df["x"] + df["w"] / 2
    df["cy"] = df["y"] + df["h"] / 2
    df["col"] = (df["cx"] // cell_size).clip(0, n_cols - 1).astype(int)
    df["row"] = (df["cy"] // cell_size).clip(0, n_rows - 1).astype(int)

    # Para cada célula, conta em quantos FRAMES DISTINTOS ela recebeu
    # ao menos uma detecção (evita contar múltiplas detecções no mesmo
    # frame/célula como "mais recorrência")
    occupancy = np.zeros((n_rows, n_cols), dtype=int)
    for (row, col), group in df.groupby(["row", "col"]):
        occupancy[row, col] = group["frame"].nunique()

    return occupancy


def filter_by_spatial_recurrence(
    df: pd.DataFrame,
    frame_w: int,
    frame_h: int,
    cell_size: int = 20,
    occupancy_ratio_threshold: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    occupancy_ratio_threshold: fração dos frames da sequência em que
        uma célula precisa ter tido detecção para ser considerada
        "hot" (provável clutter estático).
    """
    df = _clean_detections(df)
    df["cx"] = df["x"] + df["w"] / 2
    df["cy"] = df["y"] + df["h"] / 2

    total_frames = df["frame"].nunique()
    occupancy = build_occupancy_map(df, frame_w, frame_h, cell_size)
    occupancy_ratio = occupancy / max(total_frames, 1)

    hot_mask = occupancy_ratio >= occupancy_ratio_threshold

    n_cols = frame_w // cell_size + 1
    n_rows = frame_h // cell_size + 1
    df["col"] = (df["cx"] // cell_size).clip(0, n_cols - 1).astype(int)
    df["row"] = (df["cy"] // cell_size).clip(0, n_rows - 1).astype(int)

    is_hot = df.apply(lambda r: hot_mask[int(r["row"]), int(r["col"])], axis=1)

    kept_df = df[~is_hot].drop(columns=["cx", "cy", "row", "col"])
    suppressed_df: Series | DataFrame = df[is_hot].drop(columns=["cx", "cy", "row", "col"])
    return kept_df, suppressed_df