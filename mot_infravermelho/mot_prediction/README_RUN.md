# Como rodar o experimento mot_prediction

Tudo roda no Mac (Apple M2), a partir da raiz do projeto LAFusion (`~/PycharmProjects/DroneTrackingMOT`). Nada fora de `mot_prediction/` é escrito.

## 1. Ambiente (automático, ~1 min na primeira vez)

O experimento usa o **`.venv` normal do LAFusion**, na raiz do projeto. O `run_all.sh` sempre chama esse Python pelo caminho, então não importa qual ambiente esteja ativo no terminal (`(base)`, `(.venv)`).

A única coisa que falta nele é o CSRT, que fica no pacote *contrib* do OpenCV. Na primeira execução, o `setup_env.sh` (etapa "ambiente") troca o `opencv-python 4.11.0.86` pelo `opencv-contrib-python 4.11.0.86`. É a mesma versão, com os módulos extras, e o código do LAFusion continua igual. Nas execuções seguintes, ele só confere, em poucos segundos. Para desfazer a troca:

```bash
.venv/bin/python -m pip uninstall -y opencv-contrib-python
.venv/bin/python -m pip install opencv-python==4.11.0.86
```

Se existir uma pasta `mot_prediction/.venv` de uma tentativa anterior, ela não é usada e pode ser apagada.

## 2. Referência visível para H4 (opcional, mas sem ela H4 fica inconclusiva)

H4 compara a deriva do CSRT no Track 3 com a do híbrido do SIGE no DUT Anti-UAV. Os dados do DUT não estão nesta pasta. Se você tiver o DUT convertido pelo pipeline do SIGE, preencha no `mot_prediction/config.yaml`:

```yaml
dut:
  images_dir: "/caminho/para/datasets/anti_uav_tracking_test_final/images/test"   # videoNN_00001.jpg
  gt_dir: "/caminho/para/datasets/_bruto"                                          # videoNN_gt.txt oficiais
```

Os pesos do SIGE já estão em `publicar_github/.../pesos_treinados/yolo26n_dut_best.pt`. Se souber a taxa de quadros do DUT, preencha `dut.fps_declared`.

## 3. Teste rápido (~15 min)

```bash
cd ~/PycharmProjects/DroneTrackingMOT
bash mot_prediction/run_all.sh --smoke
```

Roda a etapa 0 (`check_setup.py`, que grava `mot_prediction/check_setup.txt`) e a etapa 1 (teste numa sequência, `UAV005`). No fim, imprime a estimativa de tempo das demais etapas e o horário previsto de término.

Confira antes da noite:

* `mot_prediction/check_setup.txt`, seção RESUMO: zero problemas bloqueantes; seção 4: qual arquivo de trajetória da linha de base foi selecionado e se reproduz o artigo;
* `mot_prediction/results/smoke/smoke_report.md`: todas as verificações bloqueantes "ok" (fidelidade do Kalman ao SIGE e reprodução das métricas publicadas).

Se algo falhar, o terminal e o `smoke_report.md` dizem o quê. Corrija e rode o mesmo comando de novo.

## 4. Execução noturna

```bash
cd ~/PycharmProjects/DroneTrackingMOT
caffeinate -dimsu bash mot_prediction/run_all.sh
```

* O script já se reexecuta sob `caffeinate` se você esquecer; o `caffeinate` explícito só deixa isso visível.
* Mantenha o Mac na tomada e feche outros programas pesados: o FPS medido é parte dos resultados (H3).
* Sem as varreduras (N_red 5/10/30, KCF, MOSSE e HIT-Tracker com tempo por quadro): `bash mot_prediction/run_all.sh --skip-sweeps`.
* Se a execução cair (queda de energia, erro), rode o **mesmo comando** de novo: as etapas e as sequências concluídas são puladas e nada é apagado. O log de cada execução fica em `mot_prediction/logs/run_<data>.log` e o tempo de cada etapa em `mot_prediction/logs/stage_times.csv`.

Etapas, na ordem: 0 verificação → 1 teste rápido → 2 Experimento A → 3 Experimento B principal (+ DUT, se configurado) → 4 varreduras → resumo.

## 5. O que me trazer de volta

Só a pasta **`mot_prediction/results/summary/`** (ou o arquivo `mot_prediction/results/summary_<data>.zip`, que o script cria no fim). Ela tem apenas tabelas CSV e Markdown, três figuras PNG, o `summary.md`, o `check_setup.txt` e os tempos das etapas. Não é preciso mandar trajetórias nem saídas brutas.

Se a noite parar no meio, rode `mot_prediction/.venv/bin/python mot_prediction/make_summary.py` e traga o `summary/` parcial junto com as últimas 50 linhas do log.

## 6. Etapas avulsas (se precisar)

```bash
PY=mot_prediction/.venv/bin/python
$PY mot_prediction/check_setup.py [--rapido]
$PY mot_prediction/steps/s1_smoke.py
$PY mot_prediction/steps/s2_exp_a.py [--only-seq UAV005]
$PY mot_prediction/steps/s3_exp_b.py [--only-seq UAV005] [--sem-dut]
$PY mot_prediction/steps/s4_sweeps.py [--variant hybrid_kcf_n15] [--sem-hit]
$PY mot_prediction/make_summary.py
```

Para incluir o DUT depois que a etapa 3 já rodou: preencha `dut.*` e rode `s3_exp_b.py` de novo (o Track 3 é pulado).

## 7. Onde fica cada coisa

```
mot_prediction/
├── config.yaml                  caminhos e parametros (unico)
├── HYPOTHESES.md                hipoteses e metricas (pre-registro)
├── ASSUMPTIONS.md               adaptacoes ao MOT, com a origem no SIGE/LAFusion
├── HANDOFF_REPRODUTIBILIDADE.md do experimento ao comando (preenchido apos os resultados)
├── run_all.sh  setup_env.sh  check_setup.py  make_summary.py  requirements.txt
├── steps/                       s1_smoke.py, s2_exp_a.py, s3_exp_b.py, s4_sweeps.py
├── motpred/                     codigo (importa o SIGE e o LAFusion sem altera-los)
├── logs/                        run_<data>.log, stage_times.csv
└── results/
    ├── setup/                   selecao das trajetorias, estado do ambiente
    ├── smoke/                   teste rapido e estimativa
    ├── exp_a/                   avaliacao por sequencia (linha de base e HIT-Tracker)
    ├── exp_b/                   traj/, frames/, anchors/, eval/ por variante
    ├── h4_dut/                  hibrido do SIGE no DUT
    └── summary/                 <- o que volta para analise
```
