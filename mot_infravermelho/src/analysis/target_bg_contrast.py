"""
Mede o contraste REAL alvo-vs-fundo (não a textura global da cena) usando
as caixas de ground truth, e correlaciona com o desempenho de tracking.

Para cada caixa GT amostrada, calcula:
    contrast = |mean(target_patch) - mean(surrounding_annulus)| / (std(annulus) + eps)

onde o "annulus" é um anel ao redor da caixa (fundo imediato), não a cena
inteira. Isso é o que a literatura de small-target IR chama de "contraste":
saliência do alvo em relação à vizinhança, não heterogeneidade de textura
da imagem toda (que é o que `frame_quality_metrics.py::local_contrast` mede).

Saídas:
  1. Correlação (Pearson) por sequência: mean_target_bg_contrast x MOTA/IDF1/FN_rate
  2. Teste de hipótese por objeto (Mann-Whitney U): contraste de caixas GT que
     foram PERDIDAS (miss/FN) pelo tracker vs. as que foram DETECTADAS,
     no mesmo espírito da análise da Seção 8.6 (center_shift_normalized).
     Isso é a evidência mais direta: "baixo contraste alvo-fundo causa FN?"
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from common import MOT_EVAL_DIR, VIDEO_ROOT, get_val_sequences, seq_to_video_filename  # noqa: E402

OUT_DIR = MOT_EVAL_DIR
FRAME_SAMPLE_STRIDE = 25  # mesmo stride do frame_quality_metrics.py, pra ficar comparável

# TODO: ajuste este caminho/padrão se sua estrutura de GT for diferente.
# Assumido: dataset/<seq_name>/gt/gt.txt no formato MOTChallenge:
# frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
# Novo GT_ROOT apontando para a pasta TrainLabels
# Ajuste exato para sua estrutura: TrainLabels/MultiUAV-<numero>.txt
# Novo GT_ROOT apontando para a pasta TrainLabels
GT_ROOT = Path("/Users/ighorribeiro/PycharmProjects/DroneTrackingMOT/dataset/MultiUAV_Train/TrainLabels")

def seq_to_gt_path(seq_name: str) -> Path:
    # Extrai dígitos do nome da sequência (ex.: "UAV005" -> "005")
    num_part = ''.join(filter(str.isdigit, seq_name))
    if not num_part:
        # fallback: usa o nome inteiro se não houver dígitos
        return GT_ROOT / f"MultiUAV-{seq_name}.txt"
    return GT_ROOT / f"MultiUAV-{num_part}.txt"

def load_gt(seq_name: str) -> pd.DataFrame:
    gt_path = seq_to_gt_path(seq_name)
    if not gt_path.exists():
        print(f"[AVISO] GT não encontrado: {gt_path}")
        return pd.DataFrame()
    cols = ["frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height",
            "conf", "class", "visibility"]
    df = pd.read_csv(gt_path, header=None, names=cols)
    return df


def target_bg_contrast(gray: np.ndarray, x: int, y: int, w: int, h: int,
                        pad_ratio: float = 0.75) -> float:
    """Contraste entre a caixa alvo e um anel ao redor dela (fundo imediato)."""
    H, W = gray.shape
    pad = max(int(max(w, h) * pad_ratio), 4)

    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    xin0, yin0 = max(x, 0), max(y, 0)
    xin1, yin1 = min(x + w, W), min(y + h, H)

    if xin1 <= xin0 or yin1 <= yin0 or x1 <= x0 or y1 <= y0:
        return np.nan

    target_patch = gray[yin0:yin1, xin0:xin1]

    # anel = janela externa menos a caixa alvo (marcada com NaN via máscara)
    outer = gray[y0:y1, x0:x1].astype(np.float64)
    mask = np.ones_like(outer, dtype=bool)
    ry0, ry1 = yin0 - y0, yin1 - y0
    rx0, rx1 = xin0 - x0, xin1 - x0
    mask[ry0:ry1, rx0:rx1] = False
    annulus = outer[mask]

    if annulus.size < 8 or target_patch.size == 0:
        return np.nan

    annulus_mean, annulus_std = annulus.mean(), annulus.std()
    target_mean = target_patch.mean()

    return float(abs(target_mean - annulus_mean) / (annulus_std + 1e-6))


def process_sequence(seq_name: str) -> pd.DataFrame:
    """Retorna um DataFrame por-objeto: frame, track_id, contrast, matched (preenchido depois)."""
    video_path = VIDEO_ROOT / seq_to_video_filename(seq_name)
    gt_df = load_gt(seq_name)
    if not video_path.exists() or gt_df.empty:
        return pd.DataFrame()

    gt_df = gt_df[gt_df["frame"] % FRAME_SAMPLE_STRIDE == 0]
    if gt_df.empty:
        return pd.DataFrame()

    cap = cv2.VideoCapture(str(video_path))
    frame_cache = {}
    rows = []

    for frame_num, group in gt_df.groupby("frame"):
        if frame_num not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # GT costuma ser 1-indexed
            ret, frame = cap.read()
            if not ret:
                continue
            frame_cache[frame_num] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = frame_cache[frame_num]

        for _, r in group.iterrows():
            c = target_bg_contrast(gray, int(r.bb_left), int(r.bb_top),
                                    int(r.bb_width), int(r.bb_height))
            rows.append({
                "sequence": seq_name,
                "frame": frame_num,
                "track_id": r.track_id,
                "bb_left": r.bb_left, "bb_top": r.bb_top,
                "bb_width": r.bb_width, "bb_height": r.bb_height,
                "target_bg_contrast": c,
            })

    cap.release()
    return pd.DataFrame(rows)


def load_perf(tracker_name: str) -> pd.DataFrame:
    path = MOT_EVAL_DIR / f"mot_summary_{tracker_name}.csv"
    if not path.exists():
        print(f"[AVISO] {path} não encontrado")
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df = df.drop(index="OVERALL", errors="ignore")
    df.index.name = "sequence"
    df = df.reset_index()
    if "num_objects" in df.columns:
        df["fp_rate"] = df["num_false_positives"] / df["num_objects"]
        df["fn_rate"] = df["num_misses"] / df["num_objects"]
    df["mota_pct"] = df["mota"] * 100
    df["idf1_pct"] = df["idf1"] * 100
    return df


def pearson_r(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def main():
    val_seqs = get_val_sequences()
    print(f"Calculando contraste alvo-fundo (GT-based) para {len(val_seqs)} sequências...\n")

    all_objects = []
    for seq_name in val_seqs:
        obj_df = process_sequence(seq_name)
        if obj_df.empty:
            print(f"[AVISO] sem dados para {seq_name} (checar GT_ROOT/seq_to_gt_path)")
            continue
        all_objects.append(obj_df)
        print(f"{seq_name}: {len(obj_df)} caixas GT amostradas, "
              f"contraste médio={obj_df['target_bg_contrast'].mean():.3f}")

    if not all_objects:
        print("\nNenhuma sequência processada. Verifique GT_ROOT e o formato do gt.txt.")
        return

    objects_df = pd.concat(all_objects, ignore_index=True)
    objects_csv = OUT_DIR / "target_bg_contrast_per_object.csv"
    objects_df.to_csv(objects_csv, index=False)
    print(f"\nDados por objeto salvos em: {objects_csv}")

    # --- 1. Correlação por sequência (Pearson) ---
    seq_summary = objects_df.groupby("sequence")["target_bg_contrast"].mean().reset_index()
    seq_summary.columns = ["sequence", "mean_target_bg_contrast"]

    for tracker_name in ["botsort", "bytetrack"]:
        perf_df = load_perf(tracker_name)
        if perf_df.empty:
            continue
        merged = seq_summary.merge(perf_df, on="sequence", how="inner")

        print("\n" + "=" * 70)
        print(f"Correlação (contraste alvo-fundo REAL x desempenho) -- {tracker_name}")
        print("=" * 70)
        for perf_m in ["mota_pct", "idf1_pct", "fp_rate", "fn_rate"]:
            r = pearson_r(merged["mean_target_bg_contrast"].values, merged[perf_m].values)
            print(f"  target_bg_contrast x {perf_m}: r = {r:.3f}")

    seq_summary_csv = OUT_DIR / "target_bg_contrast_per_sequence.csv"
    seq_summary.to_csv(seq_summary_csv, index=False)
    print(f"\nResumo por sequência salvo em: {seq_summary_csv}")

    # --- 2. Comparação por objeto: contraste em caixas onde há evidência de dificuldade ---
    # Sem o join direto com o output do tracker (fora do escopo deste script),
    # como proxy, comparamos o quartil de MENOR contraste vs. o de MAIOR contraste
    # dentro das sequências de PIOR IDF1 vs. MELHOR IDF1, para ver se o padrão
    # "baixo contraste -> pior desempenho" aparece também no nível do objeto.
    perf_bot = load_perf("botsort")
    if not perf_bot.empty:
        merged_seq = seq_summary.merge(perf_bot[["sequence", "idf1_pct"]], on="sequence")
        median_idf1 = merged_seq["idf1_pct"].median()
        low_perf_seqs = set(merged_seq[merged_seq["idf1_pct"] < median_idf1]["sequence"])
        high_perf_seqs = set(merged_seq[merged_seq["idf1_pct"] >= median_idf1]["sequence"])

        low_contrast_vals = objects_df[objects_df["sequence"].isin(low_perf_seqs)]["target_bg_contrast"].dropna()
        high_contrast_vals = objects_df[objects_df["sequence"].isin(high_perf_seqs)]["target_bg_contrast"].dropna()

        stat, p = mannwhitneyu(low_contrast_vals, high_contrast_vals, alternative="two-sided")
        print("\n" + "=" * 70)
        print("Mann-Whitney U: contraste alvo-fundo em sequências de baixo vs. alto IDF1")
        print("=" * 70)
        print(f"  Baixo IDF1 (abaixo da mediana): mediana contraste = {low_contrast_vals.median():.3f}, n={len(low_contrast_vals)}")
        print(f"  Alto IDF1 (acima/igual mediana): mediana contraste = {high_contrast_vals.median():.3f}, n={len(high_contrast_vals)}")
        print(f"  p-value = {p:.4g}")

    print("\nFEITO. Para a evidência mais forte (contraste em caixas PERDIDAS pelo "
          "tracker vs. DETECTADAS), seria preciso cruzar isso com a saída de "
          "matches do py-motmetrics (por-frame, TP/FN por track_id) -- me avise "
          "se quiser que eu monte essa parte também, usando o evaluate_mot.py.")


if __name__ == "__main__":
    main()