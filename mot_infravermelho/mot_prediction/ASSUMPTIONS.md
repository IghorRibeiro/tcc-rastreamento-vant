# Suposições e adaptações

Cada item diz o que foi decidido, de onde vem no código original e por quê. Os caminhos do SIGE são relativos a `publicar_github/hybrid-uav-tracking-prediction/`; os do LAFusion, à raiz do projeto. Nada em `publicar_github/`, `src/`, `runs/`, `weights/` ou `audit/` é alterado: o código original é importado (`motpred/common.py`) e os resultados publicados são apenas lidos.

---

## A. Dados e protocolo

**A-01. Avaliação sem subamostragem.** O rastreamento do LAFusion roda nos vídeos completos (750 quadros por sequência, 25 fps), não no conjunto com passo 4. O passo 4 (`src/data/split_train_val.py`, `frame_num % 4 == 1`) só gera o conjunto de detecção (`dataset/dataset_yolo/images/val`, 5.724 imagens); isso consta em `README.md` §2 e em `audit/provenance_current.md` §1 ("os executores MOT usam os vídeos completos e não o stride"). Consequência: H5, H10, H15 e H30 são quadros originais do Track 3, isto é, 0,2 s, 0,4 s, 0,6 s e 1,2 s. O `check_setup.py` confere isso no `gt.txt`, nos vídeos e nas trajetórias (seção 5 do `check_setup.txt`). Se a conferência mostrar o contrário, os horizontes em segundos serão corrigidos em adendo ao `HYPOTHESES.md`.

**A-02. Qual arquivo é a "linha de base" do artigo.** Os arquivos `runs/mot_eval/UAV*_botsort.txt` foram regravados em 03/08/2026 (mesma data do `timing_botsort.csv`, posterior ao `mot_summary_botsort.csv` de 26/07) e hoje dão 75,44% MOTA, 70,46% IDF1 e 2.443 IDs (`audit/validated_runs/validation_report.md`), e não os 75,9% / 71,2% / 2.327 do artigo. Os arquivos `UAV*_GMC_noReID.txt` (ablação 2×2, configuração padrão do `botsort.yaml`, 26/07) são os candidatos a ter gerado o número do artigo (`runs/mot_eval/ablation_gmc_reid_summary.csv`). Regra adotada: o `check_setup.py` recalcula as métricas de cada candidato e escolhe o primeiro que reproduz, sequência por sequência, o `runs/mot_eval/mot_summary_botsort.csv` (tolerância 1e-6 em MOTA e IDF1, IDs exatos). Se nenhum reproduzir, usa o primeiro completo e o resumo informa as métricas efetivas. Para o HIT-Tracker, `UAV*_hardmining_sahi512_flit_nsa.txt` reproduz 79,9% / 82,7% / 494 segundo o manifesto da auditoria; o mesmo teste é aplicado.

**A-03. Sequências.** As 30 sequências de `src/inference/common.py::get_val_sequences()` (semente 42, fração 0,15), conferidas contra `audit/validated_runs/run_manifest.json`.

**A-04. Avaliador MOT.** `motpred/mot_eval.py::build_accumulator` repete o laço de `src/inference/evaluate_mot.py::main` (conjunto de quadros, ordem das linhas, `iou_matrix(max_iou=0.5)`, `MOTAccumulator(auto_id=True)`). A única mudança é agrupar os quadros com `groupby` em vez de filtrar um a um, o que só altera o tempo. O agregado geral usa as fórmulas do py-motmetrics a partir das somas por sequência, em vez de guardar os 30 acumuladores na memória. O teste rápido confere a sequência de teste contra o resumo publicado; o `check_setup` confere as 30.

**A-05. Definição de FPS.** O LAFusion mediu FPS como quadros / tempo total do `model.track(stream=True)`, incluindo a decodificação do vídeo. Aqui se usa a definição do SIGE (`uav_common.tracking_metrics`): 1000 / tempo médio de processamento por quadro, sem decodificação. Os FPS deste experimento não são comparáveis em valor absoluto aos 46,46 FPS do artigo; a comparação válida é entre métodos da mesma sessão (linha de base e híbrido são intercalados por sequência na etapa 3).

**A-06. Estado do rastreador na rodada de referência.** `run_tracker_on_val.py` usou um único `YOLO` com `persist=True` para as 30 sequências, então o BoT-SORT carregava trilhas e contador de IDs de uma sequência para a outra. A rodada de referência cria um modelo novo por sequência e zera o rastreador depois do aquecimento. O efeito se restringe aos primeiros quadros de cada sequência. Essa rodada serve para medir tempo por quadro e dar um ponto de comparação na mesma sessão; o Experimento A usa as trajetórias salvas do artigo.

**A-07. Aquecimento.** Duas chamadas do detector não cronometradas por sequência, em todos os métodos (o SIGE não aquecia; sem isso a primeira sequência de cada método pagaria a compilação do MPS).

---

## B. Arquitetura híbrida no MOT

Origem: `src/04_rastreadores.py::run_hibrido` (SIGE). Implementação: `motpred/hybrid.py`.

| ID | No SIGE (SOT) | No MOT (este experimento) | Motivo |
|---|---|---|---|
| B-01 | `detect_best()`: só a detecção de maior confiança | todas as detecções do quadro, com o mesmo arredondamento de `detect_best()` | vários alvos |
| B-02 | detecção quando `tracker is None` e em `index % period == 0` se o CSRT teve sucesso | detecção no quadro 1, em `index % N_red == 0` e em todo quadro enquanto não houver trilha ativa | a redetecção agendada é global; a regra "detectar no quadro seguinte a uma falha" do SOT faria a taxa de detecção depender do número de alvos. Ela existe como variante (`redetect_on_failure`, desabilitada) |
| B-03 | não há associação | húngaro (`ultralytics.trackers.utils.matching.linear_assignment`, o lapjv do BoT-SORT) com o custo de `src/postprocessing/robust_flit_association.py::make_flit_get_dists` (γ = 0,55; σ = 0,15 fixo no código; distância normalizada pela diagonal 640×512), `fuse_score` e limiar `match_thresh = 0,8` do `botsort.yaml` | é a mesma função que o HIT-Tracker instala no BoT-SORT. A caixa da trilha é a do rastreador local no quadro (ativa) ou a última caixa (perdida); no HIT-Tracker é a predita pelo Kalman interno, que o híbrido não tem |
| B-04 | — | um único estágio de associação; confiança da chamada ao detector = 0,5 | o SIGE ignora detecções abaixo de 0,5; o BoT-SORT tem dois estágios (alta e baixa confiança), que não existem no híbrido |
| B-05 | reancora se a detecção tem confiança ≥ 0,7 | trilha ativa associada a detecção ≥ 0,7: CSRT reiniciado na caixa detectada; < 0,7: o CSRT segue, mas a associação confirma a trilha | mantém `switch_conf` |
| B-06 | inicializa com confiança ≥ 0,5 | detecção não associada com confiança ≥ 0,5 cria trilha; sem número mínimo de acertos | mantém `init_conf`; o SIGE reporta desde a inicialização |
| B-07 | falha do CSRT → `tracker = None`, nada é reportado | trilha passa a "perdida" (não reportada) e pode ser reativada, com o mesmo ID, por detecção ≥ 0,5 associada em até 30 quadros | 30 = `track_buffer` do `botsort.yaml`; no SOT a identidade é implícita |
| B-08 | não existe | morte: perdida há mais de 30 quadros, ou ativa sem nenhuma detecção associada há ≥ 30 quadros (avaliado nos quadros com detecção) | sem essa regra, um CSRT que deriva para o fundo (principal fonte de erro no Track 3, segundo o LAFusion) viveria indefinidamente |
| B-09 | exceções do OpenCV não são tratadas | exceção do rastreador local em caixa muito pequena conta como falha e é registrada (`exceptions`) | alvos de 10 a 25 px no infravermelho |
| B-10 | pesos do DUT | `weights/best.pt` (YOLO26n base do LAFusion), no quadro inteiro | pareado com a linha de base, como no SIGE (Tabela 3: híbrido e BoT-SORT com o mesmo YOLO26n). Pesos HardMining e SAHI existem como variantes desabilitadas |
| B-11 | falha num quadro de redetecção → detecção só no quadro seguinte | a trilha que falha num quadro de redetecção pode ser reativada pela detecção do mesmo quadro | consequência de B-02 |
| B-12 | lê JPEGs | lê o `.mp4` com `cv2.VideoCapture`, como os executores do LAFusion; decodificação fora do cronômetro | é a fonte usada no LAFusion |

O filtro de Kalman não roda dentro do laço, como no SIGE (docstring de `04_rastreadores.py`: "o filtro de Kalman consome apenas o centro da caixa de cada quadro, entao ele e aplicado no PASSO 5 sobre a trajetoria gravada").

---

## C. Predição de trajetória

Origem: `src/uav_common.py::init_kalman`, `::forward` e `::prediction_errors` (SIGE). Implementação: `motpred/prediction.py`.

**C-01. Filtro sem alteração.** `init_kalman` e `forward` são importados. O laço de `kalman_states()` é cópia linha a linha do laço de `prediction_errors` (inicializa no primeiro quadro; depois `predict` → `correct` → amortecimento no estado corrigido; `forward` com γ a cada passo). O teste rápido compara a reimplementação com `prediction_errors` bit a bit, numa trajetória do GT e numa trilha real com lacunas; divergência interrompe a execução.

**C-02. Um filtro por trilha; lacunas puladas.** Como no SIGE (em que a saída do híbrido também tem lacunas depois de uma falha do CSRT), o filtro dá um passo por observação, sem tratar a lacuna. A fração de passos precedidos por lacuna é relatada por sequência (`frac_passos_apos_lacuna`), porque no BoT-SORT uma trilha perdida e recuperada com o mesmo ID é comum.

**C-03. Alvo de referência.** Única mudança exigida pelo MOT: o alvo em t + H é o associado à trilha em t (π^CLEAR, eventos MATCH e SWITCH do py-motmetrics) e não "o" alvo da sequência. A correspondência secundária é o mapeamento global do IDF1. Como a MOTA e o IDF1 usam correspondências diferentes, as duas são calculadas e a primária está fixada no `HYPOTHESES.md`.

**C-04. Descartes contabilizados.** No SIGE, `if target is None: continue` descarta sem contar. Aqui o descarte por alvo ausente em t + H e os passos sem associação em t são contados e reportados.

**C-05. Normalização.** Pela diagonal da caixa do alvo em t + H.

**C-06. Agregação.** Por sequência, média sobre todos os pares (o análogo da média por sequência do SIGE); entre sequências, média ± desvio-padrão e IC95% por t de Student (`uav_common.ci95_t`). Bootstrap e testes pareados usam `05_avaliar.py::bootstrap_ci` (semente 2026).

**C-07. Kalman externo também no HIT-Tracker.** O HIT-Tracker tem o NSA-Kalman dentro do BoT-SORT. O filtro padronizado é aplicado por fora, sobre a trajetória gravada, para os três rastreadores, como no SIGE (Tabela 3: todas as trajetórias pelo mesmo Kalman persistente).

---

## D. H4 (deriva do CSRT)

**D-01. Referência visível.** O híbrido do SIGE roda no DUT Anti-UAV por `motpred/dut_sige.py::run_hibrido_instrumented`, cópia de `04_rastreadores.py::run_hibrido` que só acrescenta o registro das ancoragens; `detect_best` e `make_local_tracker` são os originais. Roda no mesmo ambiente deste experimento (OpenCV contrib 4.11), e não no do artigo do SIGE (contrib 4.13, Ultralytics 8.4.53); a conferência com os números do artigo aparece em `validation_checks.md`.

**D-02. Dados do DUT.** Não estão nesta pasta. Sem `dut.images_dir` e `dut.gt_dir` no `config.yaml`, H4 é declarada inconclusiva por falta de referência, como previsto no pré-registro.

**D-03. Taxa de quadros do DUT.** O DUT é distribuído como quadros JPEG; a taxa não está nos dados. A deriva é comparada por quadro; em segundos, só se `dut.fps_declared` for informado.

**D-04. Âncora.** O segmento só entra se a detecção de ancoragem tiver IoU ≥ 0,5 com algum alvo do GT; assim a deriva medida é a do rastreador local, não a de uma detecção errada.

---

## E. Ambiente e execução

**E-01. Ambiente.** O experimento roda no `.venv` do LAFusion (Python 3.13.5, ultralytics 8.4.113, torch 2.13.0; as versões efetivas ficam no `check_setup.txt`). A única alteração, feita pelo `setup_env.sh` e só se faltar o CSRT, é trocar o `opencv-python 4.11.0.86` pelo `opencv-contrib-python 4.11.0.86`: o CSRT, o KCF e o MOSSE só existem no contrib, que contém todo o `opencv-python`, então o comportamento do código do LAFusion não muda. Um ambiente próprio com as versões fixadas foi tentado e abandonado, porque o `numpy 1.26.4` não tem pacote pronto para Python 3.13 e o resolvedor do pip não fechou as dependências. Decisão do autor: basta manter a mesma ordem de grandeza, sem exigir as versões exatas. O `run_all.sh` nunca usa outro Python sem `--python` explícito.

**E-02. paths.txt.** Não havia `paths.txt` na raiz em 28/09/2026. Os caminhos estão no `config.yaml`; se um `paths.txt` for criado, o `check_setup.py` confere cada caminho listado.

**E-03. Sementes.** 42 para `random`, `numpy` e `torch`; 2026 no bootstrap (igual ao SIGE). O CSRT é determinístico; a inferência no MPS pode variar nas últimas casas.

**E-04. Referência do HIT-Tracker.** `motpred/runners.py::HitReference` copia o laço de `src/inference/run_sahi_flit_nsa.py::run_tracker_on_sequence` (SAHI 512, sobreposição 0,2, confiança 0,25, `BOTSORT` novo por sequência, patches GM e NSA ativos antes da criação do rastreador e removidos no fim) e acrescenta o tempo por quadro. Serve só como referência de custo e de métricas na mesma sessão.

**E-06. Varreduras ajustadas pelo tempo de execução (decisão de 28/09, depois do teste rápido e antes da execução principal).** Em `UAV005` (45 alvos por quadro), o híbrido com CSRT levou 1,22 s por quadro (0,8 FPS), contra 13,7 ms da linha de base: cerca de 20 ms por trilha na atualização do CSRT e outro tanto na reinicialização de cada trilha reancorada. O KCF levou 15 ms por quadro e o MOSSE, 1,7 ms. Com isso, cada variante com CSRT leva de 3 a 4 h nas 30 sequências. A configuração principal (CSRT, N_red = 15), que é a de todas as hipóteses, fica como estava; a varredura opcional de N_red ∈ {5, 10, 30} passa a usar o KCF, e as variantes CSRT com N_red 5/10/30 ficam no `config.yaml` desabilitadas. Isso só afeta resultados descritivos: nenhum teste primário do `HYPOTHESES.md` usa as varreduras. O custo medido do CSRT é um resultado para H3, e não um defeito da implementação: é o custo do `TrackerCSRT` do OpenCV multiplicado pelo número de alvos.

**E-07. Ground truth do DUT.** O SIGE avalia pelo GT oficial (`videoNN_gt.txt`) e aceita, como alternativa, os rótulos YOLO gerados no PASSO 2 (`05_avaliar.py --labels-dir`, via `uav_common.read_yolo_labels`). Se `dut.gt_dir` não tiver o GT oficial, usam-se os rótulos YOLO, convertidos para pixels com a resolução medida no primeiro quadro de cada sequência. A diferença entre as duas fontes é só o arredondamento da conversão; o resumo informa qual foi usada.

**E-05. Execução única.** Cada configuração roda uma vez, numa máquina (Apple M2), na partição interna de validação.
