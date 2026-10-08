# Rastreamento de VANTs com Visão Computacional e Redes Neurais Profundas — código do TCC

Código que gerou os resultados do Trabalho de Conclusão de Curso *Rastreamento de Drones usando
Visão Computacional e Redes Neurais Profundas* (CApA-Of, CIAA, 2026), de Ighor de Souza Ribeiro.
O TCC reúne dois trabalhos e os liga por uma análise comum de predição de trajetória:

| Etapa | Cenário | Dados | Método principal | Repositório do artigo |
|---|---|---|---|---|
| 1 | Um VANT no espectro visível (SOT) | DUT Anti-UAV | detector YOLO26n que reancora um rastreador local CSRT a cada 15 quadros (arquitetura híbrida) | [hybrid-uav-tracking-prediction](https://github.com/IghorRibeiro/hybrid-uav-tracking-prediction) |
| 2 | Enxames no infravermelho (MOT) | 4th Anti-UAV Challenge, Track 3 | HIT-Tracker: YOLO26n ajustado por *hard-example mining*, inferência por recortes de 512 px (SAHI), associação húngara com custo de Geman-McClure e NSA-Kalman | [DroneTrackingMOT](https://github.com/IghorRibeiro/DroneTrackingMOT) |
| 3 | Predição de trajetória nos dois cenários | as duas bases | filtro de Kalman de velocidade constante externo ao rastreador, persistente (γ = 1) e amortecido (γ = 0,95), horizontes de 5 a 30 quadros | este repositório |

Este repositório tem **só código**: não traz dados, pesos nem resultados brutos (trajetórias,
predições, tabelas geradas). Os números do texto saem dos scripts abaixo rodando sobre os dados
e as saídas salvas dos dois projetos originais.

## Estrutura

```text
tcc-rastreamento-vant/
├── README.md, LICENSE, requirements.txt, requirements-sige.txt, ligar_dados.sh
├── figuras/
│   └── pipeline_mot.py            figura do fluxo do HIT-Tracker (Figuras/pipeline_mot.pdf)
├── sot_visivel/                   Etapa 1 -- DUT Anti-UAV
│   ├── hybrid-uav-tracking-prediction/   pacote publicado do artigo do SIGE (src/01..06, uav_common.py)
│   └── projeto_sige/              scripts originais que geraram as trajetórias e as tabelas do SIGE
│       ├── metricasdeteccao.py, validar_yolo26s.py, figuras_drawio.py, botsort_drone.yaml
│       └── scripts/{conversao, fine_tuning, tracking, kalman, reviewer_validation, avaliacao_avancada}
└── mot_infravermelho/             Etapas 2 e 3 -- mesma organização do DroneTrackingMOT
    ├── train.py, configs/data.yaml
    ├── src/{data, inference, postprocessing, preprocessing, analysis}   pipeline do LAFusion
    ├── paper_figures/             métricas e figuras do artigo do LAFusion
    ├── mot_prediction/            experimento de 28/09: híbrido no MOT, linha de base e HIT com tempo por quadro
    └── tcc_hoje/                  reexecução final (07/10): Partes 1 a 4, estatística, figuras e resumo
```

`mot_infravermelho/` mantém a organização do projeto original porque o `mot_prediction` e o
`tcc_hoje` localizam `src/`, `dataset/`, `weights/` e `runs/` a partir da pasta-mãe deles.

## Instalação

São dois ambientes (Python 3.13.5; os resultados foram gerados em um Apple M2 com MPS):

```bash
git clone https://github.com/IghorRibeiro/tcc-rastreamento-vant.git
cd tcc-rastreamento-vant

# MOT no infravermelho, mot_prediction e reexecução final
python3.13 -m venv mot_infravermelho/.venv
mot_infravermelho/.venv/bin/pip install -r requirements.txt

# VANT único no visível (só para gerar de novo as trajetórias do DUT)
python3.13 -m venv .venv-sige
.venv-sige/bin/pip install -r requirements-sige.txt
```

Use apenas `opencv-contrib-python` (os rastreadores CSRT, KCF, MOSSE e MedianFlow estão nele).

## Dados, pesos e saídas salvas

| O quê | Onde obter |
|---|---|
| DUT Anti-UAV | origens oficiais em <https://github.com/wangdongdut/DUT-Anti-UAV>; `sot_visivel/hybrid-uav-tracking-prediction/src/01_baixar_dataset.py` baixa e `02_converter_dataset.py` converte |
| Track 3 do 4th Anti-UAV Challenge | <https://doi.org/10.5281/zenodo.15103888>; `mot_infravermelho/src/data/convert_mot_to_yolo.py` e `split_train_val.py` preparam o conjunto do detector |
| Pesos | `pesos_treinados/yolo26n_dut_best.pt` no repositório do SIGE; `weights/best.pt` e `weights/best_hardmining.pt` no repositório do DroneTrackingMOT |

Quem já tem os dois projetos originais pode ligá-los a este repositório, sem copiar nada:

```bash
bash ligar_dados.sh ~/PycharmProjects/DroneTrackingMOT ~/PycharmProjects/TesteYolo26
```

O script cria links simbólicos (ignorados pelo git) para `dataset/`, `weights/`, `runs/`,
`mot_prediction/results/` e o `.venv` do projeto do MOT, e para os dados e pesos do SIGE. A
reexecução final lê o projeto do SIGE pelo caminho `paths.teste_yolo26` em
`mot_infravermelho/tcc_hoje/config.yaml` (padrão `~/PycharmProjects/TesteYolo26`).

## Como reproduzir cada tabela e figura

Ordem natural: (1) trajetórias do DUT, (2) `mot_prediction` (rodadas de 28/09 com tempo por quadro),
(3) `tcc_hoje` (Partes 1 a 4). Os passos 1 e 2 só precisam ser refeitos para gerar de novo as
saídas salvas; a reexecução final parte delas.

```bash
cd mot_infravermelho
bash mot_prediction/run_all.sh --skip-sweeps     # passo 2 (~3 h no M2)
bash tcc_hoje/run_hoje.sh --smoke                # teste rápido (~10 min)
bash tcc_hoje/run_hoje.sh                        # passo 3 (~1h30 a 2h15) -> tcc_hoje/resultados/
```

A reexecução final é retomável (marcadores em `tcc_hoje/resultados/_marcadores/`), aborta com
menos de 3 GB livres e grava `RESUMO_TCC.md`, `estatistica_testes.csv`, as tabelas em CSV e as
figuras.

| Item do TCC | Código | Saída |
|---|---|---|
| Detectores no DUT (precisão, revocação, mAP), avaliados nas 24.804 imagens de teste do rastreamento | `sot_visivel/projeto_sige/metricasdeteccao.py`; treino em `scripts/fine_tuning/1.fine_tuning.py` (ou `src/03_treinar_detector.py` do pacote) | terminal / `detectores.csv` (Parte 1a, valores publicados) |
| Trajetórias do DUT: BoT-SORT | `projeto_sige/scripts/tracking/rastreador_botsort.py` | `avaliacao_tracking/predicoes/exp01_yolo26n_botsort_padrao/` |
| Trajetórias do DUT: ByteTrack e DeepSORT (`max_cosine_distance = 0,7`) | `projeto_sige/scripts/reviewer_validation/run_alt_trackers.py` | `predicoes/reviewer_validation/{bytetrack,deepsort}/` |
| Trajetórias do DUT: OSTrack e ODTrack | `hybrid-uav-tracking-prediction/src/extras/rodar_ostrack_odtrack.py` (repositórios oficiais clonados à parte) | `predicoes/reviewer_validation/{ostrack,odtrack}/` |
| Trajetórias do DUT: híbrido com CSRT, KCF, MOSSE e MedianFlow | `projeto_sige/scripts/tracking/run_opencv_trackers.py`, `kcf_hibrido.py`, `scripts/conversao/conversao.py`; Kalman em `scripts/kalman/kalman_hibrido_n_CV.py` | `predicoes/{hibrido_csrt,kcf_hibrido,mosse_hibrido,medianflow_hibrido}/` |
| Rastreamento no DUT e rastreadores locais (SA, SR@0,5, IoU, AUC, cobertura) | `tcc_hoje/tcc/p1a_sot.py`, que importa sem alterar `projeto_sige/scripts/reviewer_validation/evaluate_tracking.py` | `parte1/sot_resumo_metodos.csv`, `sot_reproducao_publicado.csv` |
| Predição no DUT (H5 a H30, persistente e amortecida, em px e normalizada) | `p1a_sot.py` com o Kalman do SIGE (`uav_common.py`) | `parte1/sot_pred_por_video.csv`, `sot_resumo_metodos.csv` |
| Diagonal do alvo (DUT pelo GT oficial, sem os quadros −100; Track 3 pelo GT) | `p1a_sot.py`, `p1b_mot.py` | `parte1/dut_diagonal_resumo.json`, `mot_alvos_resumo.json` |
| Detector do Track 3 e *hard-example mining* | `train.py`; `src/inference/run_tracker_on_train.py` → `select_hard_examples.py` → `build_hard_dataset.py`; métricas em `src/analysis/evaluate_detection_models.py` | pesos em `weights/` |
| Linha de base, HIT-Tracker e híbrido no MOT, com tempo por quadro | `mot_prediction/` (`run_all.sh`, etapa 3) | `mot_prediction/results/exp_b/` |
| Leaderboard (ByteTrack, OC-SORT, Deep OC-SORT, FastTrack, TrackTrack), SAHI, sensibilidade ao γ e determinismo | `tcc_hoje/tcc/p2_rodadas.py`; laços originais em `src/inference/run_multitracker_leaderboard.py`, `run_sahi_tracker.py`, `run_robust_flit_sweep.py`, `run_sahi_flit_nsa.py`; patches em `src/postprocessing/` | `parte2/rodadas_resumo.csv`, `determinismo.csv`, `sahi_decisao.json` |
| Evolução da linha de base até o HIT-Tracker | Parte 2 + `mot_prediction` | `RESUMO_TCC.md`, seção 7 |
| Causa raiz (densidade, movimento de câmera, densidade de bordas, contraste) | fatores por `src/analysis/frame_quality_metrics.py`, `camera_motion_estimation.py`, `target_bg_contrast.py`, `density_correlation.py` (em `runs/mot_eval/`); correlações em `tcc_hoje/tcc/p1c_causa.py` | `parte1/causa_raiz_correlacoes.csv` |
| Pré-processamento (MOG2, EMA, fluxo óptico) | `src/inference/run_preprocessing_ablation.py`, `evaluate_preprocessing_ablation.py`, `src/preprocessing/background_masks.py`; no SIGE, `projeto_sige/scripts/avaliacao_avancada/hibrido_mog2.py` | resultados anteriores, não reexecutados |
| Custo × densidade | `tcc_hoje/tcc/p1d_custo.py` (tempos de 28/09) | `parte1/custo_regressao.csv` |
| Predição no MOT com identidade (CLEAR MOT) | `tcc_hoje/tcc/p1b_mot.py` | `parte1/mot_pred_resumo.csv`, `mot_suporte_comum_resumo.csv` |
| Testes estatísticos (famílias F1 a F4) | `tcc_hoje/tcc/p3_estatistica.py`, `estat.py` | `estatistica_testes.csv` |
| Figuras de resultados: `sige_amortecimento`, `lafusion_correlacoes`, `lafusion_sensibilidade_gm`, `mot_fps_densidade`, `mot_h30_idf1` | `tcc_hoje/tcc/p4_figuras.py` | `tcc_hoje/resultados/figuras/*.pdf` |
| `pipeline_sot.pdf` | draw.io (o diagrama vai embutido no próprio PDF: abra-o no draw.io para editar); recortes por `projeto_sige/figuras_drawio.py` | — |
| `pipeline_mot.pdf` | `figuras/pipeline_mot.py` | ver abaixo |

Para refazer só a estatística (segundos, sem rodar rastreador), a partir dos CSVs da Parte 1:

```bash
cd mot_infravermelho/tcc_hoje && PYTHONPATH=. ../.venv/bin/python -m tcc.p3_estatistica
```

Figura do fluxo do HIT-Tracker (lê um quadro real do Track 3, roda o detector só nesse quadro para
desenhar as caixas e usa as trilhas salvas do HIT-Tracker; nenhum resultado é recalculado):

```bash
mot_infravermelho/.venv/bin/python figuras/pipeline_mot.py \
    --raiz-mot mot_infravermelho --saida /caminho/do/TCC/Figuras/pipeline_mot.pdf
```

Sequência, quadro, trilha, cores, fontes e textos ficam no bloco `CONFIG` no topo do script; a
primeira execução grava um cache em `figuras/cache_pipeline_mot/`, e as seguintes só redesenham.

## Notas de reprodutibilidade

- **Estado do rastreador.** Com `persist=True`, o Ultralytics reaproveita o rastreador entre
  chamadas do mesmo modelo. Nas rodadas do TCC cada sequência (e cada γ) usa um modelo novo, então
  nenhuma trilha, contador de identidade ou compensação de movimento passa de uma sequência para
  outra; por isso alguns números diferem dos artigos (seção 14 do `RESUMO_TCC.md`).
- **Filtro de Kalman.** O erro de predição usa o Kalman do SIGE publicado (`uav_common.py`:
  covariância inicial identidade e amortecimento a cada passo da previsão), igual no DUT e no MOT.
  A Parte 1a também recalcula, com o avaliador das tabelas do artigo do SIGE, os valores publicados,
  para conferência.
- **Família F3.** Compara o CSRT com KCF, MOSSE e MedianFlow em H30 usando os mesmos erros por vídeo
  da tabela de rastreadores locais do TCC (Kalman do SIGE). Na rodada de 07/10 o F3 usava os erros
  por vídeo salvos da execução do artigo; os demais testes não mudam.
- **MOSSE e MedianFlow.** As trajetórias originais da tabela de rastreadores locais do artigo do
  SIGE não estão salvas; o TCC usa a reexecução de 06/09, sinalizada nas saídas.
- **FPS.** Não houve nova medição de velocidade; o custo × densidade usa os tempos por quadro de 28/09.
- **Caminhos.** Os scripts originais de `sot_visivel/projeto_sige/` foram mantidos como estavam,
  com caminhos absolutos da máquina do autor no bloco de configuração do topo de cada um.
- **Versões.** Versões diferentes das fixadas mudam os números nas últimas casas
  (ver `sot_visivel/hybrid-uav-tracking-prediction/docs/DIFERENCAS_ENTRE_AMBIENTES.md`).

## Licença

Código sob licença [MIT](LICENSE). O pipeline depende do [Ultralytics](https://github.com/ultralytics/ultralytics)
(AGPL-3.0), e os patches em `mot_infravermelho/src/postprocessing/` adaptam métodos dos rastreadores
do Ultralytics; quem redistribuir esse trecho deve observar os termos da AGPL-3.0. OSTrack e ODTrack
não estão incluídos e seguem as licenças dos seus repositórios. As bases de dados seguem as licenças
dos seus autores (DUT Anti-UAV: Apache-2.0; Track 3: MIT).

## Como citar

Ribeiro, I. S. *Rastreamento de Drones usando Visão Computacional e Redes Neurais Profundas*.
Trabalho de Conclusão de Curso (CApA-Of), Centro de Instrução Almirante Alexandrino, 2026.
