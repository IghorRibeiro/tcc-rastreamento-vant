# Validação dos revisores

Este diretório contém a execução independente das comparações solicitadas nos pareceres. O procedimento usa as imagens extraídas em `datasets/anti_uav_tracking_test_final/images/test` e as anotações YOLO em `labels/test`; não depende dos MP4s externos referenciados pelos scripts históricos.

## Dependências verificadas no ambiente conectado

| Pacote | Versão observada |
|---|---:|
| Python | 3.13.5 |
| Ultralytics | 8.4.53 |
| OpenCV | 4.13.0 |
| NumPy | 2.4.5 |
| SciPy | 1.17.1 |
| deep-sort-realtime | 1.3.2 |

Instalação mínima no ambiente virtual:

```bash
.venv/bin/pip install ultralytics opencv-python numpy pandas scipy pycocotools deep-sort-realtime
```

## Execução

ByteTrack, usando YOLO26n e o `bytetrack.yaml` nativo do Ultralytics:

```bash
.venv/bin/python scripts/reviewer_validation/run_alt_trackers.py \
  --tracker bytetrack --device mps
```

DeepSORT, usando YOLO26n em todos os quadros e o embedder MobileNet do pacote `deep-sort-realtime`:

```bash
.venv/bin/python scripts/reviewer_validation/run_alt_trackers.py \
  --tracker deepsort --device mps
```

Uma única sequência pode ser testada com `--video video01`. Os resultados são gravados em `avaliacao_tracking/predicoes/reviewer_validation/{bytetrack,deepsort}/videoXX.txt` no formato:

```text
frame,x,y,w,h,confidence,time_ms
```

A avaliação com variabilidade entre as 20 sequências é executada com:

```bash
.venv/bin/python scripts/reviewer_validation/evaluate_tracking.py
```

Os resultados consolidados da execução completa são:

- `results_with_deepsort/all_methods_per_video.csv`: métricas por método e sequência;
- `results_with_deepsort/summary_mean_std_ci95.csv`: média, desvio-padrão entre sequências, mediana, mínimo, máximo e IC95%;
- `results_with_deepsort/paired_significance_tests.csv`: testes t pareado e Wilcoxon entre os métodos;
- `results_with_deepsort/figures/`: gráficos comparativos para o manuscrito e a apresentação.

A execução final contém **20 arquivos ByteTrack e 20 arquivos DeepSORT**, todos verificados. A pasta `deepsort_partial_previous/` é um arquivo histórico incompleto e não participa da análise.

## Interpretação

`SA_legacy_IoU50` preserva a definição operacional usada no avaliador histórico: proporção de quadros com IoU maior ou igual a 0,5, incluindo o tratamento de quadros sem alvo. `mean_IoU_present` é reportada separadamente para não confundir essa taxa com a média de IoU descrita no texto do artigo.

O erro H5/H10/H15/H30 é calculado a partir dos centros das caixas produzidas pelo tracker, alimentando um filtro de Kalman de velocidade constante sem amortecimento. Assim, a comparação entre rastreadores não incorpora o ganho específico do Kalman Damped; a ablação do amortecimento permanece em `scripts/avaliacao_avancada/hibrido_cv_damped.py`.

Os tempos são medidos no mesmo computador e incluem inferência do detector e processamento do tracker. Eles são adequados para comparação relativa entre os métodos executados nesse protocolo, mas não devem ser misturados diretamente com os FPS históricos obtidos na Tesla T4 ou no MacBook M2.
