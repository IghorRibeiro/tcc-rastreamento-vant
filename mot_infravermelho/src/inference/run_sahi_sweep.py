"""
Executa automaticamente 3 configurações de SAHI na inferência e avalia
os resultados com o avaliador já existente (evaluate_mot.py).

Saídas:
- runs/mot_eval/*_sahi512.txt, *_sahi384.txt, *_sahi320.txt
- runs/mot_eval/timing_sahi512.csv, timing_sahi384.csv, timing_sahi320.csv
- runs/mot_eval/mot_summary_sahi*.csv (gerados pelo evaluate_mot)
- runs/mot_eval/sahi_summary.csv  (tabela consolidada)
- runs/mot_eval/sahi_summary.md   (ordenado por IDF1 desc)

Uso básico:
  python -m src.inference.run_sahi_sweep

Parâmetros opcionais (replicados ao runner de inferência):
  --device mps|cpu|cuda:0
  --conf 0.25
  --tracker-cfg botsort.yaml|ultra-botsort
  --only-seq UAV005 UAV007
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(SRC_DIR))
from common import MOT_EVAL_DIR


def run_tracker(
    config,
    model_name,
    weights,
    device,
    conf,
    tracker_cfg,
    only_seq
):
    cmd = [

        sys.executable,

        "-m",

        "src.inference.run_sahi_tracker",

        "--config", config,

        "--weights", str(weights),

        "--exp-name", model_name,

        "--device", device,

        "--conf", str(conf),

        "--tracker-cfg", tracker_cfg,

    ]
    if only_seq:
        cmd.extend(["--only-seq", *only_seq])
    print("[INFO] Executando:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def evaluate_suffix(suffix: str) -> Path:

    cmd = [
        sys.executable,
        "-m",
        "src.inference.evaluate_mot",
        "--tracker",
        suffix,
    ]

    print("[INFO] Avaliando:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    return MOT_EVAL_DIR / f"mot_summary_{suffix}.csv"


def main():
    parser = argparse.ArgumentParser(description="Sweep de SAHI (448/576) + avaliação MOT")
    parser.add_argument("--device", default="mps", help="mps|cpu|cuda:0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tracker-cfg", default="botsort.yaml")
    parser.add_argument("--only-seq", nargs="*", default=None)
    args = parser.parse_args()

    MOT_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    MODELS = [

        (
            "baseline",
            PROJECT_ROOT / "weights" / "best.pt"
        ),

        (
            "hardmining",
            PROJECT_ROOT / "weights" / "best_hardmining.pt"
        ),

    ]

    CONFIGS = [
        "320",
        "384",
        "448",
        "512",
        "576",
    ]

    rows: list[dict] = []
    for model_name, weights in MODELS:

        print("\n" + "=" * 80)
        print(f"MODELO: {model_name}")
        print("=" * 80)

        for cfg in CONFIGS:

            suffix = f"{model_name}_sahi{cfg}"

            print(f"\n[SWEEP] Rodando {suffix}")

            run_tracker(
                config=cfg,
                model_name=model_name,
                weights=weights,
                device=args.device,
                conf=args.conf,
                tracker_cfg=args.tracker_cfg,
                only_seq=args.only_seq,
            )

            csv_path = evaluate_suffix(suffix)

            if not csv_path.exists():
                print(f"[WARN] {csv_path} não encontrado.")
                continue

            df = pd.read_csv(csv_path, index_col=0)

            if "OVERALL" not in df.index:
                continue

            o = df.loc["OVERALL"]

            mota = float(o["mota"]) * 100
            motp = float(o["motp"])
            idf1 = float(o["idf1"]) * 100
            ids = int(o["num_switches"])
            fps = float(o.get("fps", float("nan")))

            num_obj = float(o.get("num_objects", 0))
            fp = float(o.get("num_false_positives", 0))
            fn = float(o.get("num_misses", 0))

            fp_rate = fp / num_obj * 100 if num_obj else float("nan")
            fn_rate = fn / num_obj * 100 if num_obj else float("nan")

            rows.append({
                "Model": model_name,
                "SAHI": cfg,
                "Config": suffix,
                "MOTA": round(mota, 1),
                "MOTP": round(motp, 3),
                "IDF1": round(idf1, 1),
                "IDs": ids,
                "FP_rate": round(fp_rate, 2),
                "FN_rate": round(fn_rate, 2),
                "FPS": round(fps, 2),
            })

    if not rows:
        print("[ERRO] Nenhum resultado para consolidar.")
        return

    summary_df = pd.DataFrame(rows)
    # Ordena por IDF1 desc
    summary_sorted = summary_df.sort_values(
        by=["Model", "SAHI"]
    )

    out_csv = MOT_EVAL_DIR / "sahi_summary_v2.csv"
    summary_sorted.to_csv(out_csv, index=False)

    # Markdown
    md_lines = [
        "| Model | SAHI | MOTA | MOTP | IDF1 | IDs | FP_rate | FN_rate | FPS |",
        "|:------|----:|-----:|-----:|-----:|----:|--------:|--------:|----:|",
    ]
    for _, row in summary_sorted.iterrows():
        md_lines.append(
            f"| {row['Model']} | {row['SAHI']} | "
            f"{row['MOTA']:.1f} | "
            f"{row['MOTP']:.3f} | "
            f"{row['IDF1']:.1f} | "
            f"{row['IDs']} | "
            f"{row['FP_rate']:.2f} | "
            f"{row['FN_rate']:.2f} | "
            f"{row['FPS']:.2f} |"
        )
    out_md = MOT_EVAL_DIR / "sahi_summary_v2.md"
    out_md.write_text("\n".join(md_lines))

    print("\n" + "=" * 80)
    print(f"Resumo consolidado salvo em:\n  - {out_csv}\n  - {out_md}")
    print("=" * 80)


if __name__ == "__main__":
    main()
