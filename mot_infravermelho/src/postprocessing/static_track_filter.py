"""
Filtro pós-tracking: suprime detecções de tracks cuja posição do
centro permanece praticamente parada (baixa variância) por uma janela
de frames sustentada -- indicativo de clutter estático (janelas,
portas, estruturas de fundo), não de um drone real.

Opera diretamente sobre arquivos de saída já gerados pelo tracker
(formato MOT), sem re-rodar detecção ou tracking.
"""

import numpy as np
import pandas as pd


def filter_static_tracks(
    df: pd.DataFrame,
    window: int = 30,
    min_periods_ratio: float = 0.6,
    std_threshold: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    df: colunas esperadas frame, id, x, y, w, h (formato MOT já carregado).
    window: span de frames considerado para o cálculo de variância.
    min_periods_ratio: fração mínima de `window` com detecção válida
        para o cálculo ser considerado confiável (evita marcar tracks
        curtos/esparsos como estáticos por falta de dados).
    std_threshold: limiar de desvio-padrão combinado (pixels) abaixo
        do qual o track é considerado estático naquele ponto.

    Retorna (kept_df, suppressed_df).
    """
    df = df.copy()
    df["cx"] = df["x"] + df["w"] / 2
    df["cy"] = df["y"] + df["h"] / 2

    suppressed_flags = pd.Series(False, index=df.index)
    min_periods = max(2, int(window * min_periods_ratio))

    for tid, group in df.groupby("id"):
        group = group.sort_values("frame")
        frame_min, frame_max = int(group["frame"].min()), int(group["frame"].max())

        full_index = pd.RangeIndex(frame_min, frame_max + 1)
        cx_series = pd.Series(index=full_index, dtype=float)
        cy_series = pd.Series(index=full_index, dtype=float)
        cx_series.loc[group["frame"].values] = group["cx"].values
        cy_series.loc[group["frame"].values] = group["cy"].values

        std_cx = cx_series.rolling(window, min_periods=min_periods).std()
        std_cy = cy_series.rolling(window, min_periods=min_periods).std()
        combined_std = np.sqrt(std_cx.fillna(np.inf) ** 2 + std_cy.fillna(np.inf) ** 2)

        is_static_by_frame = combined_std < std_threshold

        for idx, frame in zip(group.index, group["frame"]):
            if is_static_by_frame.get(frame, False):
                suppressed_flags.loc[idx] = True

    kept_df = df[~suppressed_flags].drop(columns=["cx", "cy"])
    suppressed_df = df[suppressed_flags].drop(columns=["cx", "cy"])
    return kept_df, suppressed_df


def df_to_mot_lines(df: pd.DataFrame) -> list[str]:
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"{int(row['frame'])},{int(row['id'])},{row['x']:.2f},{row['y']:.2f},"
            f"{row['w']:.2f},{row['h']:.2f},1,-1,-1,-1"
        )
    return lines