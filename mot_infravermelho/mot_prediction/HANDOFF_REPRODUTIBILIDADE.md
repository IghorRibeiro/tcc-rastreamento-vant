# Do experimento ao comando (predição de trajetória no MOT)

No formato de `publicar_github/.../docs/MAPA_ARTIGO_PIPELINE.md` e `docs/VALIDACAO.md`. Para cada resultado, o comando que o reproduz e o arquivo onde ele aparece. **Os valores são preenchidos depois da execução**, a partir de `results/summary/`; até lá, este arquivo só fixa o caminho de cada número.

Os comandos assumem a raiz do projeto LAFusion e o ambiente `mot_prediction/.venv` (README_RUN.md, passo 1). O `run_all.sh` executa exatamente esta sequência.

---

## Pré-requisito: verificação

```bash
mot_prediction/.venv/bin/python mot_prediction/check_setup.py
```

* **Saída:** `mot_prediction/check_setup.txt`, `results/setup/trajectory_selection.json`, `results/setup/setup_status.json`.
* **Conferência esperada:** 30 sequências idênticas às do manifesto; avaliação sem subamostragem (passo 1, 750 quadros por sequência, 25 fps); CSRT disponível; BoT-SORT sem ReID; trajetórias da linha de base e do HIT-Tracker que reproduzem o artigo sequência por sequência.
* **Valores:** *(preencher)*

## Teste rápido

```bash
mot_prediction/.venv/bin/python mot_prediction/steps/s1_smoke.py
```

* **Saída:** `results/smoke/smoke_report.md`, `checks.json`, `estimate.json`.
* **Conferência esperada:** Kalman reimplementado idêntico a `uav_common.prediction_errors` (diferença 0); avaliador reproduz o publicado em `UAV005`.

---

## Experimento A: predição sobre as trajetórias salvas

```bash
mot_prediction/.venv/bin/python mot_prediction/steps/s2_exp_a.py
mot_prediction/.venv/bin/python mot_prediction/make_summary.py
```

* **Saída:** `results/summary/exp_a_prediction_error.csv|.md` (H5 a H30, px e normalizado, persistente e amortecido, π^CLEAR e π^ID), `exp_a_per_sequence.csv`, `exp_a_common_support.csv`, `h30_vs_idf1.csv`, `fig_h30_vs_idf1.png`.
* **Protocolo:** Kalman do SIGE sem alteração, um filtro por trilha, alvo de referência pela correspondência do CLEAR-MOT; unidade amostral = sequência (n = 30); IC95% por t de Student.
* **Valores:** *(preencher)*

## Experimento B: arquitetura híbrida no MOT

```bash
mot_prediction/.venv/bin/python mot_prediction/steps/s3_exp_b.py
mot_prediction/.venv/bin/python mot_prediction/steps/s4_sweeps.py          # opcional
mot_prediction/.venv/bin/python mot_prediction/make_summary.py
```

* **Saída:** `results/summary/exp_b_overview.csv|.md`, `exp_b_prediction_error.csv`, `exp_b_per_sequence.csv`, `fps_vs_density.csv`, `fig_fps_vs_density.png`.
* **Configuração principal:** YOLO26n `weights/best.pt`, N_red = 15, CSRT, custo GM (γ = 0,55, σ = 0,15) com húngaro, confiança 0,5 para nascer/reativar e 0,7 para reancorar, trilha perdida por até 30 quadros.
* **Valores:** *(preencher)*

## H4: deriva do CSRT, infravermelho × visível

* **Saída:** `results/summary/h4_drift.csv|.md`, `h4_drift_curve.csv`, `fig_drift_curve.png`, `h4_dut_sige_per_video.csv`.
* **Valores:** *(preencher)*

## Testes das hipóteses

* **Saída:** `results/summary/hypothesis_tests.csv|.md` (regras de `HYPOTHESES.md`, seção 3).
* **Valores:** *(preencher)*

---

## Afirmações do texto

| Afirmação | Onde conferir |
|---|---|
| 30 sequências, 22.500 quadros, 25 fps, sem subamostragem | `check_setup.txt`, seções 3, 5 e 6 |
| Trajetórias da linha de base = as do artigo (75,9% / 71,2% / 2.327) | `check_setup.txt`, seção 4; `validation_checks.md` |
| Kalman idêntico ao do SIGE | `smoke_report.md` (fidelidade bit a bit) |
| Erro em H30 da linha de base e do HIT-Tracker | `exp_a_prediction_error.md` |
| Efeito do amortecimento no MOT (H1) | `hypothesis_tests.md`, linhas H1 |
| IDF1 × erro de predição (H2a) e HIT × linha de base (H2b) | `hypothesis_tests.md`; `fig_h30_vs_idf1.png` |
| MOTA, IDF1, IDs, FPS do híbrido | `exp_b_overview.md` |
| Custo × densidade (H3) | `fps_vs_density.csv`; `fig_fps_vs_density.png`; `hypothesis_tests.md` |
| Deriva do CSRT (H4) | `h4_drift.md`; `fig_drift_curve.png` |
| Ambiente (versões, máquina) | `environment.json` |

## Validação

*(preencher com os números de `validation_checks.md`: reprodução do LAFusion pelas trajetórias selecionadas, fidelidade do Kalman, reprodução do SIGE no DUT neste ambiente)*

## Limitações registradas antes da execução

Execução única; partição interna de validação (não o conjunto de teste oficial); uma máquina (Apple M2), com FPS comparável só dentro da mesma sessão; OpenCV contrib 4.11 no lugar do 4.13 do SIGE; H4 compara domínios que diferem em mais do que a modalidade (`HYPOTHESES.md`, H4).
