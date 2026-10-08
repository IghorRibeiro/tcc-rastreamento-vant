#!/usr/bin/env python3
"""Audit the provenance of manuscript numbers versus saved project artifacts."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ADV = ROOT / "scripts" / "avaliacao_avancada" / "avaliacao_avancada"

ARTICLE_TABLE_II = pd.DataFrame([
    {"method": "BoT-SORT YOLO26n", "SA": 0.814, "SR@0.5": 0.816, "AP@0.5": 0.793, "AP@0.5:0.95": 0.424, "FPS_T4": 23.74, "FPS_MPS": 43.3},
    {"method": "BoT-SORT YOLO26m", "SA": 0.859, "SR@0.5": 0.864, "AP@0.5": 0.838, "AP@0.5:0.95": 0.456, "FPS_T4": 22.38, "FPS_MPS": 23.3},
    {"method": "Híbrido", "SA": 0.831, "SR@0.5": 0.851, "AP@0.5": 0.789, "AP@0.5:0.95": 0.389, "FPS_T4": 11.88, "FPS_MPS": 66.5},
    {"method": "Híbrido Damped", "SA": 0.831, "SR@0.5": 0.851, "AP@0.5": 0.789, "AP@0.5:0.95": 0.389, "FPS_T4": 12.00, "FPS_MPS": 66.4},
    {"method": "Híbrido Damped + MOG2", "SA": 0.843, "SR@0.5": 0.863, "AP@0.5": 0.804, "AP@0.5:0.95": 0.394, "FPS_T4": 11.96, "FPS_MPS": 29.8},
])


FILES = {
    "Híbrido CV": ADV / "metricas_hibrido_COMPLETO.csv",
    "Híbrido MOG2": ADV / "hibrido_mog2" / "metricas_hibrido_mog2.csv",
    "BoT-SORT unificado": ADV / "metricas_botsort.csv",
    "Híbrido CV damped": ADV / "hibrido_cv_damped" / "metricas_hibrido_cv_damped.csv",
}


def main() -> None:
    output_dir = ROOT / "scripts" / "reviewer_validation" / "lineage"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, path in FILES.items():
        if not path.exists():
            rows.append({"artifact": label, "path": str(path), "exists": False})
            continue
        df = pd.read_csv(path)
        row = {"artifact": label, "path": str(path), "exists": True, "n_rows": len(df)}
        for metric in ["SA", "SR@0.5", "AP@0.5", "AP@0.5:0.95", "FPS", "Tempo_medio_ms", "erro_medio_h5", "erro_medio_h10", "erro_medio_h15", "erro_medio_h30"]:
            if metric in df:
                row[f"{metric}_mean"] = float(df[metric].mean())
                row[f"{metric}_std_between_videos"] = float(df[metric].std(ddof=1))
        rows.append(row)
    historical = pd.DataFrame(rows)
    historical.to_csv(output_dir / "historical_artifact_means.csv", index=False, float_format="%.8f")
    ARTICLE_TABLE_II.to_csv(output_dir / "article_table_II_expected.csv", index=False, float_format="%.8f")

    lines = [
        "# Auditoria de linhagem dos números do manuscrito",
        "",
        "Este arquivo é gerado por `analyze_lineage.py` e não altera nenhum resultado histórico.",
        "",
        "## Arquivos históricos encontrados",
        "",
        "```text\n" + historical.to_string(index=False) + "\n```",
        "",
        "## Tabela II transcrita do manuscrito",
        "",
        "```text\n" + ARTICLE_TABLE_II.to_string(index=False) + "\n```",
        "",
        "## Alertas de rastreabilidade",
        "",
        "- Os vetores da Figura 3 estão codificados manualmente em `imagemartigo.py`; não há leitura direta dos CSVs.",
        "- Os arquivos `hitratepos/*/estados_kalman/*_states.csv` existem, mas nenhum script presente no diretório grava esses arquivos.",
        "- Os scripts históricos apontam para caminhos absolutos externos (`/Volumes/...` e `/Users/...`), portanto não são reproduzíveis em outra máquina sem parametrização.",
        "- O avaliador legado usa o nome `SA`, mas implementa SA como proporção de IoU >= 0,5; o texto do artigo descreve SA como média da IoU. Essas definições precisam ser separadas na versão final.",
        "- `csrt_otimizado_bg_sub.py` usa YOLO26m e a saída `csrt_otimizado_m`, embora a narrativa do artigo apresente o detector YOLO26n no pipeline proposto.",
    ]
    (output_dir / "lineage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote lineage report to {output_dir}")
    print(historical.to_string(index=False))


if __name__ == "__main__":
    main()
