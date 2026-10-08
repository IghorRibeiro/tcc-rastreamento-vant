"""
Calcula métricas de qualidade/complexidade de imagem por sequência
(entropia, densidade de bordas, nitidez, contraste local, % de pixels
quentes) e correlaciona com o desempenho de tracking (MOTA, IDF1,
FP rate, FN rate), testando a hipótese de que fundo estruturado
("clutter") -- não densidade nem movimento de câmera -- explica a
maior parte da variância de desempenho entre sequências.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from common import MOT_EVAL_DIR, VIDEO_ROOT, get_val_sequences, seq_to_video_filename  # noqa: E402

OUT_DIR = MOT_EVAL_DIR
FRAME_SAMPLE_STRIDE = 25  # ~30 frames amostrados por sequência de 750
BLOCK_SIZE = 16  # tamanho do bloco para contraste local


def compute_frame_metrics(gray: np.ndarray) -> dict:
    # Entropia (informação/complexidade da distribuição de intensidades)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()
    hist_norm = hist_norm[hist_norm > 0]
    entropy = float(-np.sum(hist_norm * np.log2(hist_norm)))

    # Densidade de bordas (Canny) -- proxy de "quanta estrutura" tem na cena
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    # Variância do Laplaciano (nitidez geral da imagem)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Contraste local (média do desvio-padrão em blocos NxN)
    h, w = gray.shape
    contrasts = []
    for y in range(0, h - BLOCK_SIZE, BLOCK_SIZE):
        for x in range(0, w - BLOCK_SIZE, BLOCK_SIZE):
            block = gray[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            contrasts.append(block.std())
    local_contrast = float(np.mean(contrasts)) if contrasts else 0.0

    # % de pixels "quentes" (threshold adaptativo: média + 2 desvios-padrão)
    thresh = gray.mean() + 2 * gray.std()
    hot_pixel_pct = float(np.mean(gray > thresh) * 100)

    return {
        "entropy": entropy,
        "edge_density": edge_density,
        "laplacian_var": laplacian_var,
        "local_contrast": local_contrast,
        "hot_pixel_pct": hot_pixel_pct,
    }


def compute_sequence_metrics(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    samples = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_SAMPLE_STRIDE != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        samples.append(compute_frame_metrics(gray))

    cap.release()
    if not samples:
        return None

    df = pd.DataFrame(samples)
    return {f"mean_{col}": df[col].mean() for col in df.columns}


def load_raw_results(tracker_name: str) -> pd.DataFrame:
    """Carrega o CSV bruto (não o '_readable'), que ainda tem FP/FN."""
    path = MOT_EVAL_DIR / f"mot_summary_{tracker_name}.csv"
    if not path.exists():
        print(f"[AVISO] {path} não encontrado, pulando '{tracker_name}'")
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df = df.drop(index="OVERALL", errors="ignore")
    df.index.name = "sequence"
    df = df.reset_index()

    # Normaliza FP/FN pelo número de objetos GT (sequências têm volumes bem diferentes)
    if "num_objects" in df.columns:
        df["fp_rate"] = df["num_false_positives"] / df["num_objects"]
        df["fn_rate"] = df["num_misses"] / df["num_objects"]
    df["mota_pct"] = df["mota"] * 100
    df["idf1_pct"] = df["idf1"] * 100
    return df


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def main():
    val_seqs = get_val_sequences()

    print(f"Calculando métricas de imagem (stride={FRAME_SAMPLE_STRIDE}) para {len(val_seqs)} sequências...\n")
    rows = []
    for seq_name in val_seqs:
        video_path = VIDEO_ROOT / seq_to_video_filename(seq_name)
        if not video_path.exists():
            print(f"[AVISO] vídeo não encontrado para {seq_name}, pulando")
            continue
        metrics = compute_sequence_metrics(video_path)
        if metrics is None:
            continue
        metrics["sequence"] = seq_name
        rows.append(metrics)
        print(f"{seq_name}: entropy={metrics['mean_entropy']:.2f}  "
              f"edges={metrics['mean_edge_density']:.4f}  "
              f"laplacian={metrics['mean_laplacian_var']:.1f}  "
              f"contrast={metrics['mean_local_contrast']:.2f}  "
              f"hot%={metrics['mean_hot_pixel_pct']:.2f}")

    metrics_df = pd.DataFrame(rows)
    metrics_csv = OUT_DIR / "frame_quality_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nMétricas salvas em: {metrics_csv}\n")

    image_metrics = ["mean_entropy", "mean_edge_density", "mean_laplacian_var",
                      "mean_local_contrast", "mean_hot_pixel_pct"]
    perf_metrics = ["mota_pct", "idf1_pct", "fp_rate", "fn_rate"]

    all_combined = metrics_df.copy()

    for tracker_name in ["botsort", "bytetrack"]:
        results_df = load_raw_results(tracker_name)
        if results_df.empty:
            continue

        merged = metrics_df.merge(results_df, on="sequence", how="inner")
        all_combined = all_combined.merge(
            results_df.rename(columns={c: f"{c}_{tracker_name}" for c in results_df.columns if c != "sequence"}),
            on="sequence", how="left",
        )

        print("=" * 70)
        print(f"Correlação (métricas de imagem x desempenho) -- tracker: {tracker_name}")
        print("=" * 70)

        corr_table = []
        for img_m in image_metrics:
            row = {"image_metric": img_m}
            for perf_m in perf_metrics:
                row[perf_m] = round(pearson_r(merged[img_m].values, merged[perf_m].values), 3)
            corr_table.append(row)

        corr_df = pd.DataFrame(corr_table).set_index("image_metric")
        print(corr_df.to_string())
        print()

        corr_csv = OUT_DIR / f"image_quality_correlation_{tracker_name}.csv"
        corr_df.to_csv(corr_csv)
        print(f"Salvo em: {corr_csv}\n")

    out_all = OUT_DIR / "frame_quality_vs_performance.csv"
    all_combined.to_csv(out_all, index=False)
    print(f"CSV consolidado salvo em: {out_all}")

    # Destaque: ranking de fundo mais "estruturado" (maior densidade de bordas)
    ranked = metrics_df.sort_values("mean_edge_density", ascending=False)
    print("\nSequências com fundo mais ESTRUTURADO (maior densidade de bordas):")
    print(ranked[["sequence", "mean_edge_density"]].head(6).to_string(index=False))


if __name__ == "__main__":
    main()