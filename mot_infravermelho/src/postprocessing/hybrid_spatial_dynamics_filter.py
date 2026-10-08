"""
Filtro híbrido: combina recorrência espacial (frequência de ocupação
de uma célula do grid ao longo da sequência) com dinâmica local
(velocidade mediana das detecções que caíram naquela célula).

Uma célula só é suprimida se for MUITO recorrente E as detecções que
a povoam tiverem velocidade tipicamente baixa -- distinguindo clutter
estático persistente (janelas, portas) de regiões de passagem legítima
de drones (que geram recorrência espacial, mas com velocidade alta).
"""

import numpy as np
import pandas as pd


def _clean_detections(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["frame", "id", "x", "y", "w", "h"]
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols).copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"  [AVISO] {dropped} linha(s) inválida(s) descartada(s) de {before} totais.")
    return df


def _compute_local_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Velocidade instantânea de cada detecção, baseada na detecção
    anterior do MESMO track id (não exige continuidade sustentada por
    uma janela longa -- só compara com o passo imediatamente anterior
    daquele id, tolerando gaps de frames)."""
    df = df.sort_values(["id", "frame"]).copy()
    df["prev_frame"] = df.groupby("id")["frame"].shift(1)
    df["prev_cx"] = df.groupby("id")["cx"].shift(1)
    df["prev_cy"] = df.groupby("id")["cy"].shift(1)

    dt = df["frame"] - df["prev_frame"]
    dx = df["cx"] - df["prev_cx"]
    dy = df["cy"] - df["prev_cy"]

    with np.errstate(invalid="ignore", divide="ignore"):
        df["velocity"] = np.sqrt(dx**2 + dy**2) / dt
    df.loc[dt <= 0, "velocity"] = np.nan  # primeira aparição do id: sem velocidade definida

    return df


def filter_by_spatial_dynamics(
    df: pd.DataFrame,
    frame_w: int,
    frame_h: int,
    cell_size: int = 20,
    occupancy_ratio_threshold: float = 0.3,
    velocity_threshold: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    occupancy_ratio_threshold: fração dos frames em que a célula
        precisa ter recebido detecção para ser candidata a clutter.
    velocity_threshold: velocidade mediana (px/frame) das detecções
        naquela célula, abaixo da qual reforça a hipótese de clutter
        estático (em vez de passagem legítima de drone).
    """
    df = _clean_detections(df)
    df["cx"] = df["x"] + df["w"] / 2
    df["cy"] = df["y"] + df["h"] / 2

    df = _compute_local_velocity(df)

    n_cols = frame_w // cell_size + 1
    n_rows = frame_h // cell_size + 1
    df["col"] = (df["cx"] // cell_size).clip(0, n_cols - 1).astype(int)
    df["row"] = (df["cy"] // cell_size).clip(0, n_rows - 1).astype(int)

    total_frames = df["frame"].nunique()

    # Estatísticas por célula: ocupação (independente de velocidade) +
    # velocidade mediana (ignorando NaN da primeira aparição de cada id)
    occupancy = df.groupby(["row", "col"])["frame"].nunique()
    median_velocity = df.groupby(["row", "col"])["velocity"].median()

    occupancy_ratio = occupancy / max(total_frames, 1)

    hot_cells = set(
        occupancy_ratio[
            (occupancy_ratio >= occupancy_ratio_threshold)
            & (median_velocity.reindex(occupancy_ratio.index) < velocity_threshold)
        ].index
    )

    is_hot = df.apply(lambda r: (int(r["row"]), int(r["col"])) in hot_cells, axis=1)

    drop_cols = ["cx", "cy", "row", "col", "prev_frame", "prev_cx", "prev_cy", "velocity"]
    kept_df = df[~is_hot].drop(columns=drop_cols)
    suppressed_df = df[is_hot].drop(columns=drop_cols)
    return kept_df, suppressed_df