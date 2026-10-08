# Hipóteses pré-registradas: predição de trajetória no cenário MOT

**Registro:** 28/09/2026, antes de qualquer execução deste experimento.
**Regra:** este arquivo não é alterado depois que a primeira execução gravar algo em `results/`. Qualquer correção posterior entra como adendo datado no fim do arquivo, sem apagar o texto original. As decisões abaixo são mecânicas; a interpretação é feita separadamente, com fato e interpretação identificados.

Resultados negativos ou nulos têm o mesmo valor que os positivos e serão relatados com o mesmo detalhe.

---

## 1. Hipóteses

| ID | Enunciado |
|---|---|
| **H1** | O amortecimento de velocidade (γ = 0,95) reduz o erro de predição em H30 também no cenário MOT. |
| **H2** | Trajetórias com melhor IDF1 produzem menor erro de predição (H2a); o HIT-Tracker prediz melhor que a linha de base (H2b). |
| **H3** | A vantagem de custo computacional da arquitetura híbrida diminui com a densidade do enxame e pode se inverter. |
| **H4** | O CSRT no infravermelho (Track 3) apresenta mais deriva que no visível (DUT Anti-UAV). |

---

## 2. Definições

### 2.1 Notação

* Sequência *s*: uma das 30 sequências de validação do Track 3 (lista em `audit/validated_runs/run_manifest.json`, reconstruída por `src/inference/common.py::get_val_sequences()`, semente 42, fração 0,15).
* Quadro *t*: numeração 1-based do vídeo original (25 fps), a mesma do `gt.txt` e dos arquivos de trajetória.
* Alvo de referência *g*: identificador do `gt.txt`. Trilha *k*: identificador na saída do rastreador.
* Centro da caixa: *c* = (x + w/2, y + h/2). Diagonal da caixa do alvo: *d_g(t)* = √(w² + h²).

### 2.2 Correspondência trilha–alvo

* **Primária, π^CLEAR:** a correspondência por quadro do CLEAR-MOT, isto é, os eventos `MATCH` e `SWITCH` do py-motmetrics 1.4.0 na mesma sequência de chamadas de `src/inference/evaluate_mot.py` (distância 1 − IoU com `max_iou = 0,5`, ou seja, IoU ≥ 0,5). É a correspondência que gera a MOTA e a contagem de IDs.
* **Secundária, π^ID:** o mapeamento global um-para-um trilha→alvo do IDF1 (Ristani *et al.*, 2016), que maximiza o IDTP, com IDTP(k, g) = número de quadros em que IoU(k, g) ≥ 0,5. Validação obrigatória: 2·ΣIDTP / (N_gt + N_pred) deve reproduzir o IDF1 do py-motmetrics; se não reproduzir, os resultados com π^ID são marcados como não confiáveis.

### 2.3 Preditor (idêntico ao SIGE)

Filtro de Kalman de velocidade constante de `publicar_github/hybrid-uav-tracking-prediction/src/uav_common.py`, importado sem alteração: `init_kalman` (estado [x, y, ẋ, ẏ], Q = 0,03·I, R = 0,1·I, P₀ = I, velocidade inicial nula) e `forward`. Um filtro por trilha *k*, percorrendo em ordem os quadros em que *k* tem caixa, exatamente como o laço de `prediction_errors`:

1. no primeiro quadro da trilha o filtro só é inicializado (sem predição);
2. nos demais: `predict()` → `correct(centro)` → se γ < 1, velocidade × γ no estado corrigido;
3. a predição para *t + H* é `forward(estado, H, F, γ)`, com γ aplicado a cada passo.

Persistente: γ = 1,0. Amortecido: γ = 0,95. Quadros em que a trilha não tem caixa são pulados (um passo por observação, como no SIGE); a fração de passos precedidos por lacuna é relatada.

### 2.4 Erro de predição com identidade

Para cada trilha *k*, cada quadro *t* com estado corrigido e cada horizonte H ∈ {5, 10, 15, 30} quadros:

* *g* = π_t(k). Se *k* não está associada a nenhum alvo em *t* (falso positivo em *t*), não há par; o caso é contado em **N_sem_assoc**.
* Se *g* não tem caixa em *t + H*, o par é descartado e contado em **N_ausente(H)**.
* Caso contrário:
  * **e_H(k, t)** = ‖ p̂_k(t + H | t) − c_g(t + H) ‖₂, em pixels;
  * **ẽ_H(k, t)** = e_H(k, t) / d_g(t + H), adimensional (normalizado pela diagonal da caixa do alvo).

Com π^ID, *g* = π^ID(k) é fixo ao longo da trilha.

Horizontes em tempo (Track 3 sem subamostragem, 25 fps): H5 = 0,2 s; H10 = 0,4 s; H15 = 0,6 s; H30 = 1,2 s. Se o `check_setup.txt` mostrar que a avaliação é subamostrada, os horizontes em segundos são corrigidos por esse fator e isso é registrado em adendo.

### 2.5 Agregação

* **Por trilha:** média e desvio-padrão (ddof = 1) de e_H e ẽ_H sobre os pares válidos da trilha.
* **Por sequência:** média e desvio-padrão sobre todos os pares válidos da sequência (análogo ao SIGE, em que a média por sequência é tomada sobre todos os quadros).
* **Entre sequências (resultado principal):** média ± desvio-padrão das médias por sequência, com a sequência como unidade amostral (n = 30), como no SIGE. IC95% da média em H30 por t de Student (`uav_common.ci95_t`).
* **Entre trilhas (descritivo):** média ± desvio-padrão das médias por trilha, para trilhas com ao menos um par válido.

### 2.6 Suporte comum

Para comparar dois rastreadores A e B na mesma sequência: somente os pares (g, t, H, γ) presentes em ambos com π^CLEAR. Isso remove o efeito de um rastreador cobrir mais alvos ou mais quadros que o outro.

### 2.7 Métricas MOT

MOTA, IDF1, IDs, FP e FN pelo py-motmetrics, exatamente como em `src/inference/evaluate_mot.py` (IoU ≥ 0,5). Agregado geral: MOTA = 1 − (ΣFN + ΣFP + ΣIDs) / ΣGT; IDF1 = 2·ΣIDTP / (ΣGT + ΣPRED), que são as fórmulas do agregado do py-motmetrics.

### 2.8 Densidade do enxame

* Por sequência: média, sobre os quadros anotados, do número de alvos no `gt.txt` (mesma definição de `paper_figures/metrics/density.py`).
* Por quadro: n_gt(t), número de alvos anotados no quadro.

### 2.9 Custo computacional

* **Tempo por quadro:** detecção + rastreamento local + associação, medido com `time.perf_counter()`, sem a decodificação do vídeo e sem escrita em disco, após duas chamadas de aquecimento do detector não cronometradas por sequência.
* **FPS da sequência:** 1000 / média do tempo por quadro em ms (a definição de `uav_common.tracking_metrics`, no SIGE).
* **FPS por faixa de densidade:** 1000 / média do tempo dos quadros cuja n_gt está na faixa.
* **Speedup:** FPS do híbrido / FPS da linha de base com detecção em todos os quadros, ambos medidos na mesma sessão e intercalados por sequência.

### 2.10 Deriva do rastreador local (H4)

* **Evento de ancoragem:** quadro t₀ em que a caixa da trilha passa a ser a caixa de uma detecção (nascimento, reativação ou reancoragem com confiança ≥ 0,7 no MOT; inicialização ou reancoragem no SOT do SIGE).
* **Alvo-âncora:** o alvo do ground truth com maior IoU com a caixa da detecção em t₀. O segmento só entra na análise se esse IoU for ≥ 0,5.
* **Idade:** a = t − t₀, até o próximo evento de ancoragem da mesma trilha, a perda da trilha ou a = 30.
* **Erro normalizado:** ẽ(a) = ‖ c_k(t₀ + a) − c_g(t₀ + a) ‖ / d_g(t₀ + a); também se registra IoU(a).
* **Taxa de deriva da sequência:** inclinação de mínimos quadrados de ẽ em função de *a*, para 0 ≤ a ≤ N_red − 1, usando todos os segmentos da sequência.
* **Taxa de falha:** fração dos segmentos elegíveis (t₀ + N_red − 1 ≤ último quadro da sequência) em que a trilha perde a caixa antes de a = N_red − 1 sem nova ancoragem.
* **Referência visível:** híbrido do SIGE (cópia instrumentada de `src/04_rastreadores.py::run_hibrido`) nas 20 sequências do DUT Anti-UAV, no mesmo ambiente de software.

---

## 3. Testes e regras de decisão

* α = 0,05; testes bicaudais, salvo quando indicado.
* Bootstrap: 10.000 reamostragens, semente 2026 (a mesma de `05_avaliar.py::bootstrap_ci` do SIGE).
* Testes **primários** (7 no total): H1 (2), H2a (2), H2b (1), H3 (1), H4 (1). Não há correção para comparações múltiplas; todos os demais números são descritivos.

### H1: amortecimento no MOT

* **Teste primário:** para cada rastreador do Experimento A (linha de base e HIT-Tracker), Δ = H30_amortecido − H30_persistente (px, π^CLEAR), pareado por sequência (n = 30). IC95% bootstrap da média de Δ, t pareado e Wilcoxon.
* **Confirmada:** IC95% de Δ inteiramente abaixo de 0 nos dois rastreadores.
* **Refutada:** IC95% de Δ inteiramente acima de 0 em pelo menos um rastreador.
* **Inconclusiva:** os demais casos. Confirmação em só um dos dois é relatada como "parcial".
* **Descritivos:** erro normalizado; π^ID; H5 a H15; híbrido do Experimento B; redução percentual comparada aos 22% do SIGE.

### H2a: IDF1 e erro de predição

* **Teste primário:** correlação de Spearman entre IDF1_s e ẽ_H30,s (persistente, π^CLEAR), n = 30, separadamente para linha de base e HIT-Tracker.
* **Confirmada:** ρ < 0 e p < 0,05 nos dois rastreadores.
* **Refutada:** ρ > 0 e p < 0,05 em pelo menos um.
* **Inconclusiva:** os demais casos.
* **Descritivos:** Pearson; erro em pixels; conjunto combinado (n = 60, sem independência); IDs × erro.

### H2b: HIT-Tracker × linha de base

* **Teste primário:** Δ = H30_HIT − H30_base (px, persistente, π^CLEAR), pareado por sequência (n = 30), IC95% bootstrap.
* **Confirmada:** IC95% inteiramente abaixo de 0.
* **Refutada:** IC95% inteiramente acima de 0.
* **Inconclusiva:** IC95% contém 0.
* **Descritivos:** erro normalizado; suporte comum; amortecido; π^ID.

### H3: custo × densidade

* **Premissa:** existe vantagem de custo em alguma faixa de densidade, isto é, speedup > 1 em ao menos uma faixa com ≥ 200 quadros. Se a premissa for falsa, H3 é **refutada** ("não há vantagem de custo a diminuir") e a tendência é relatada só como descrição.
* **Teste primário:** Spearman entre a densidade média da sequência e o speedup da sequência (híbrido CSRT, N_red = 15, contra a linha de base com detecção em todos os quadros), n = 30.
* **Confirmada:** premissa válida, ρ < 0 e p < 0,05.
* **Refutada:** ρ > 0 e p < 0,05, ou premissa falsa.
* **Inconclusiva:** os demais casos.
* **Inversão ("pode se inverter"):** relatada como observada se o speedup for < 1 em alguma faixa de densidade com ≥ 200 quadros e > 1 em outra. Descritivo.
* **Descritivo:** regressão por quadro tempo_ms = a + b·n_gt para cada método e razão entre as inclinações.

### H4: deriva do CSRT, infravermelho × visível

* **Teste primário:** taxa de deriva por sequência no infravermelho (Track 3, híbrido CSRT, N_red = 15, n = 30) contra o visível (DUT Anti-UAV, n = 20); Mann-Whitney U unilateral (IR > visível).
* **Confirmada:** p < 0,05 e mediana IR > mediana visível.
* **Refutada:** o teste unilateral oposto (IR < visível) dá p < 0,05.
* **Inconclusiva:** os demais casos, ou ausência dos dados do DUT.
* **Descritivos:** taxa de falha; IoU em a = N_red − 1; curva ẽ(a).
* **Alcance da conclusão, fixado antes dos resultados:** os dois conjuntos diferem em modalidade, resolução (640×512 contra 1920×1080 e 1280×720), tamanho aparente do alvo, cena, câmera, número de alvos e detector. O teste compara "CSRT no domínio Track 3" com "CSRT no domínio DUT"; a normalização pela diagonal reduz o efeito do tamanho, mas o resultado não isola a modalidade.

---

## Adendos

*(nenhum até o momento)*
