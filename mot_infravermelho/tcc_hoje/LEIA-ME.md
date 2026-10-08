# tcc_hoje — reexecução enxuta do TCC (07/10)

```bash
cd ~/PycharmProjects/DroneTrackingMOT
bash tcc_hoje/run_hoje.sh --smoke     # teste rápido (~10 min) -> tcc_hoje/resultados_smoke/
bash tcc_hoje/run_hoje.sh             # completo (~2 a 2,5 h)  -> tcc_hoje/resultados/
```

- Usa `DroneTrackingMOT/.venv` (nada é instalado). Lê TesteYolo26, `runs/mot_eval`, `mot_prediction/results` e `src/` só para leitura.
- Retomável: se cair, rode o mesmo comando. Etapas prontas ficam em `resultados*/_marcadores/`; na Parte 2 cada (configuração, sequência) pronta é pulada.
- Aborta se houver menos de 3 GB livres. Roda sob `caffeinate`.
- Parte 2 usa 2 processos (RAM ≤ 8 GB) ou 3 (RAM > 8 GB).

| Etapa | Módulo | O que faz |
|---|---|---|
| 1a | `tcc/p1a_sot.py` | DUT: 9 métodos a partir das trajetórias salvas; avaliador das tabelas + Kalman do SIGE; diagonal pelo GT oficial |
| 1b | `tcc/p1b_mot.py` | MOT: predição com identidade (CLEAR) nas trajetórias de 28/09 |
| 1c | `tcc/p1c_causa.py` | Correlações fatores × desempenho da linha de base (descritivo) |
| 1d | `tcc/p1d_custo.py` | Tempo por quadro × alvos (28/09), bootstrap por sequência |
| 2 | `tcc/p2_rodadas.py` | Determinismo, HEM, HEM+SAHI 512, GM+NSA (γ), leaderboard; SAHI 320–576 se o 512 mudar |
| 3 | `tcc/p3_estatistica.py` | Famílias F1–F4 → `estatistica_testes.csv` |
| 4 | `tcc/p4_figuras.py` | 5 figuras PDF (16 cm, pt-BR) |
| — | `tcc/resumo.py` | `RESUMO_TCC.md` e `resumo_tcc.zip` |

Para refazer uma etapa: apague `resultados/_marcadores/<etapa>.done` (e, na Parte 1b, `resultados/parte1/_mot_partes/`).
