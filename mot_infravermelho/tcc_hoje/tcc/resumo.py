"""RESUMO_TCC.md + resumo_tcc.zip, lendo so os CSV/JSON de resultados/ (e os CSVs antigos citados)."""

from __future__ import annotations

import zipfile
from datetime import datetime

import numpy as np

from . import base
from .estat import fmt

NAN = float("nan")


def tabela(headers, rows) -> str:
    if not rows:
        return "_(sem dados)_\n"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def g(row, key, default=NAN):
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if v is None else v


def pct(v, nd=2):
    return fmt(v, nd)


def main() -> int:
    smoke = base.smoke_flag()
    cfg = base.load_cfg()
    out = base.out_dir(smoke)
    p1, p2 = out / "parte1", out / "parte2"
    L = []
    A = L.append
    A(f"# Resumo do TCC — reexecução enxuta ({datetime.now():%d/%m/%Y %H:%M})\n")
    if smoke:
        A("> **TESTE RÁPIDO (--smoke)**: 2 sequências curtas do MOT, 2 vídeos do DUT e "
          f"{cfg['smoke']['max_frames']} quadros por sequência na Parte 2. Os números NÃO valem para o texto.\n")
    A("Protocolo: sem novo ambiente (`DroneTrackingMOT/.venv`), sem nova medição de FPS, sem OSTrack/ODTrack/DeepSORT "
      "novos; Parte 2 só acurácia, com estado do rastreador zerado por sequência. Horizontes em quadros "
      "(H5/H10/H15/H30); comparação entre cenários pelo erro normalizado pela diagonal do GT em t+H.\n")

    # 1. Detectores ---------------------------------------------------------------
    A("## 1. Detectores (valores publicados; não reexecutados)\n")
    det = base.read_csv(p1 / "detectores.csv")
    rows = []
    if det is not None:
        for _, r in det.iterrows():
            rows.append([r["modelo"], fmt(r["precisao"], 3), fmt(r["revocacao"], 3), fmt(r["mAP50"], 3),
                         fmt(r["mAP50_95"], 3), fmt(r["imagens_avaliadas"], 0), r["fonte"]])
    A(tabela(["Modelo", "Precisão", "Revocação", "mAP@0,5", "mAP@0,5:0,95", "Imagens", "Fonte"], rows))
    A("Avaliados nas **24.804 imagens de teste do rastreamento** (`anti_uav_tracking_test_final`, split test), "
      "as mesmas usadas nas Tabelas 2 e 3.\n")

    # 2. DUT rastreamento ----------------------------------------------------------
    A("## 2. DUT Anti-UAV — rastreamento e rastreadores locais\n")
    res = base.read_csv(p1 / "sot_resumo_metodos.csv")
    rows = []
    if res is not None:
        for _, r in res.iterrows():
            rows.append([r["rotulo"], "original" if str(r["trajetoria_original"]) == "True" else "**reexecução**",
                         fmt(r["SA"], 3), fmt(r["SR05"], 3), fmt(r["IoU"], 3), fmt(r["AUC"], 3),
                         fmt(100 * r["cobertura_global"], 1) + "%"])
    A(tabela(["Método", "Trajetória", "SA", "SR@0,5", "IoU médio", "AUC", "Cobertura"], rows))
    A("Métricas pelo avaliador das tabelas publicadas (`evaluate_tracking.py`, importado sem alteração). "
      "\"Original\" = a trajetória salva reproduz o CSV por vídeo da execução publicada; \"reexecução\" = a "
      "original não foi encontrada e foi usada a reexecução de 06/09 (os erros por vídeo publicados entram no F3).\n")
    org = base.read_csv(p1 / "sot_origem_trajetorias.csv")
    if org is not None and "escolhido" in org.columns:
        rows = []
        for _, r in org[org["escolhido"] == True].iterrows():  # noqa: E712
            rows.append([r["metodo"], str(r["candidato"]).split("TesteYolo26/")[-1], str(r.get("reproduz")),
                         fmt(r.get("max_dif_H30_px"), 4), str(r.get("npred_iguais"))])
        A("Origem das trajetórias:\n")
        A(tabela(["Método", "Pasta", "Reproduz", "máx |ΔH30| (px)", "n_pred iguais"], rows))
    rep = base.read_csv(p1 / "sot_reproducao_publicado.csv")
    if rep is not None:
        rows = [[r["tabela"], r["metodo"], r["metrica"], fmt(r["publicado"], 4), fmt(r["recalc_avaliador"], 4),
                 fmt(r["recalc_kalman_sige"], 4), fmt(r["dif_avaliador"], 4), fmt(r["dif_kalman_sige"], 4)]
                for _, r in rep.iterrows()]
        A("Reprodução dos valores publicados:\n")
        A(tabela(["Tabela", "Método", "Métrica", "Publicado", "Avaliador", "Kalman SIGE", "Δ avaliador",
                  "Δ Kalman SIGE"], rows))
        A("\"Avaliador\" = Kalman do `evaluate_tracking.py` (P0 = 0; amortecimento só no estado corrigido). "
          "\"Kalman SIGE\" = `uav_common.py` publicado (P0 = I; amortecimento a cada passo da previsão), "
          "o mesmo usado no MOT.\n")
    t23 = p1 / "sot_tabela2_vs_tabela3.json"
    if t23.exists():
        j = base.read_json(t23)
        if j:
            A("**Tabela 2 (CSRT) × Tabela 3 (Híbrido):** H30 igual ao arredondamento: "
              f"{j.get('H30_igual_ao_arredondamento')}; SR@0,5 igual: {j.get('SR_igual_ao_arredondamento')}; "
              f"SA igual: {j.get('SA_igual_ao_arredondamento')} (Tabela 2 SA = {fmt(j['tabela2']['SA'], 4)}, "
              f"Tabela 3 SA = {fmt(j['tabela3']['SA'], 3)}). Recalculado da trajetória escolhida: "
              f"SA = {fmt(g(j.get('recalculado_da_trajetoria_escolhida', {}), 'SA'), 4)}, "
              f"SR = {fmt(g(j.get('recalculado_da_trajetoria_escolhida', {}), 'SR@0.5'), 4)}, "
              f"H30 = {fmt(g(j.get('recalculado_da_trajetoria_escolhida', {}), 'H30_mean_px'), 2)} px.\n")
    ds = cfg.get("deepsort_config", {})
    A(f"DeepSORT (sem rodada nova): `max_cosine_distance = {ds.get('max_cosine_distance')}`, "
      f"`max_age = {ds.get('max_age')}`, `n_init = {ds.get('n_init')}`, embedder {ds.get('embedder')} "
      "(`scripts/reviewer_validation/run_alt_trackers.py`).\n")

    # 3. DUT predicao ----------------------------------------------------------------
    A("## 3. DUT Anti-UAV — predição de trajetória (Kalman do SIGE)\n")
    rows = []
    if res is not None:
        for _, r in res.iterrows():
            rows.append([r["rotulo"] + ("" if str(r["trajetoria_original"]) == "True" else " ¹"),
                         fmt(g(r, "salvo_H30_px"), 1) + " ± " + fmt(g(r, "salvo_H30_dp"), 1),
                         fmt(r["H5_px_pers"], 1), fmt(r["H10_px_pers"], 1), fmt(r["H15_px_pers"], 1),
                         fmt(r["H30_px_pers"], 1) + " ± " + fmt(r["H30_dp_pers"], 1),
                         fmt(r["H30_px_amort"], 1) + " ± " + fmt(r["H30_dp_amort"], 1),
                         fmt(-r["reducao_H30_pct"], 1) + "%", fmt(r["H30_norm_pers"], 2), fmt(r["H30_norm_amort"], 2),
                         fmt(100 * r["H30_cob_pers"], 1) + "%"])
    A(tabela(["Método", "H30 salvo (px)", "H5", "H10", "H15", "H30 persist. (px)", "H30 amort. (px)", "Δ H30",
              "H30 norm. pers.", "H30 norm. amort.", "Cobertura H30"], rows))
    A("\"H30 salvo\" = erros por vídeo salvos da execução publicada (avaliador das tabelas, persistente). Demais "
      "colunas: Kalman do SIGE publicado, média das médias por vídeo ± dp entre vídeos. Normalizado = erro / "
      "diagonal do GT em t+H. Cobertura H30 = pares avaliados / quadros com alvo em t+H. "
      "¹ trajetória original não encontrada: H5–H30 e normalizados vêm da reexecução de 06/09; o valor publicado "
      "é o \"H30 salvo\".\n")

    # 4. Leaderboard -------------------------------------------------------------------
    A("## 4. MOT — leaderboard de rastreadores (sem FPS)\n")
    rr = base.read_csv(p2 / "rodadas_resumo.csv")
    geral = base.read_csv(p1 / "mot_metricas_geral.csv")
    rot_lb = cfg["parte2"]["rotulos_leaderboard"]
    rows = []
    if geral is not None and len(geral[geral["variante"] == "baseline_ref"]):
        b = geral[geral["variante"] == "baseline_ref"].iloc[0]
        rows.append(["BoT-SORT (linha de base de 28/09)", fmt(b["MOTA"], 2), fmt(b["IDF1"], 2), fmt(b["IDs"], 0),
                     fmt(b["FP"], 0), fmt(b["FN"], 0), "75,9", "71,2", "2.327"])
    if rr is not None:
        for _, r in rr[rr["grupo"] == "e_leaderboard"].sort_values("IDF1", ascending=False).iterrows():
            rows.append([rot_lb.get(r["config"][3:], r["config"]), fmt(r["MOTA"], 2), fmt(r["IDF1"], 2),
                         fmt(r["IDs"], 0), fmt(r["FP"], 0), fmt(r["FN"], 0), fmt(r["MOTA_antigo"], 1),
                         fmt(r["IDF1_antigo"], 1), fmt(r["IDs_antigo"], 0)])
    A(tabela(["Rastreador", "MOTA", "IDF1", "IDs", "FP", "FN", "MOTA antigo", "IDF1 antigo", "IDs antigo"], rows))
    bc = p2 / "botsort_config_check.json"
    if bc.exists():
        j = base.read_json(bc)
        A(f"BoT-SORT = linha de base de 28/09; mesma configuração do leaderboard (best.pt, botsort.yaml, conf 0,25): "
          f"**{j.get('mesma_configuracao')}** (Ultralytics {j.get('ultralytics')}).\n")

    # 5. Causa raiz ----------------------------------------------------------------------
    A("## 5. Causa raiz (descritiva; linha de base de 28/09, 30 sequências)\n")
    cor = base.read_csv(p1 / "causa_raiz_correlacoes.csv")
    rows = []
    if cor is not None:
        for _, r in cor[cor["desfecho"] == "IDF1"].iterrows():
            rows.append([r["rotulo"], fmt(r["pearson_r"], 2),
                         f"[{fmt(r['pearson_ic95_inf'], 2)}; {fmt(r['pearson_ic95_sup'], 2)}]", fmt(r["pearson_p"], 4),
                         fmt(r["spearman_rho"], 2), fmt(r["spearman_p"], 4), fmt(g(r, "pearson_r_antigo"), 2)])
    A(tabela(["Fator × IDF1", "r de Pearson", "IC95% BCa", "p", "ρ de Spearman", "p (ρ)", "r antigo"], rows))
    fe = base.read_csv(p1 / "causa_raiz_fatores_entre_si.csv")
    if fe is not None and len(fe):
        s = fe[((fe["fator_a"] == "mean_edge_density") & (fe["fator_b"] == "mean_local_contrast"))
               | ((fe["fator_b"] == "mean_edge_density") & (fe["fator_a"] == "mean_local_contrast"))]
        if len(s):
            s = s.iloc[0]
            A(f"Densidade de bordas × contraste local: r = {fmt(s['pearson_r'], 2)} "
              f"[{fmt(s['pearson_ic95_inf'], 2)}; {fmt(s['pearson_ic95_sup'], 2)}], ρ = {fmt(s['spearman_rho'], 2)}. "
              "Sem correção múltipla (descritivo). \"r antigo\" = com a rodada publicada do BoT-SORT.\n")

    # 6. SAHI ---------------------------------------------------------------------------------
    A("## 6. SAHI com o detector HEM (sem FPS)\n")
    dec_p = p2 / "sahi_decisao.json"
    dec = base.read_json(dec_p) if dec_p.exists() else {}
    rows = []
    if rr is not None:
        for name in ["hem"] + [f"hem_sahi{s}" for s in (320, 384, 448, 512, 576)]:
            sel = rr[rr["config"] == name]
            if len(sel):
                r = sel.iloc[0]
                rows.append([r["descricao"], "nova", fmt(r["MOTA"], 2), fmt(r["IDF1"], 2), fmt(r["IDs"], 0),
                             fmt(r["MOTA_antigo"], 1), fmt(r["IDF1_antigo"], 1), fmt(r["IDs_antigo"], 0)])
    if dec.get("identico_por_sequencia"):
        sv2 = base.read_csv(base.P(cfg["paths"]["mot_eval_dir"]) / "sahi_summary_v2.csv")
        if sv2 is not None:
            for _, r in sv2[(sv2["Model"] == "hardmining") & (sv2["SAHI"] != 512)].iterrows():
                rows.append([f"HEM + SAHI {int(r['SAHI'])} + BoT-SORT padrão", "antiga (mantida: 512 idêntico)",
                             fmt(r["MOTA"], 1), fmt(r["IDF1"], 1), fmt(r["IDs"], 0), "", "", ""])
    A(tabela(["Configuração", "Rodada", "MOTA", "IDF1", "IDs", "MOTA antigo", "IDF1 antigo", "IDs antigo"], rows))
    if dec:
        A(f"SAHI 512 novo × antigo idêntico por sequência: **{dec.get('identico_por_sequencia')}** → "
          f"{dec.get('acao')}.\n")

    # 7. Evolucao ---------------------------------------------------------------------------
    A("## 7. Evolução até o HIT-Tracker\n")
    rows = []
    prev = None

    def evo(label, mota, idf1, ids):
        nonlocal prev
        d = "" if prev is None else f"{fmt(mota - prev[0], 2)} / {fmt(idf1 - prev[1], 2)}"
        rows.append([label, fmt(mota, 2), fmt(idf1, 2), fmt(ids, 0), d])
        prev = (mota, idf1)

    if geral is not None and len(geral[geral["variante"] == "baseline_ref"]):
        b = geral[geral["variante"] == "baseline_ref"].iloc[0]
        evo("Linha de base: best.pt + BoT-SORT (28/09)", b["MOTA"], b["IDF1"], b["IDs"])
    for name, label in (("hem", "+ HEM (best_hardmining.pt)"), ("hem_sahi512", "+ SAHI 512")):
        if rr is not None and len(rr[rr["config"] == name]):
            r = rr[rr["config"] == name].iloc[0]
            evo(label, r["MOTA"], r["IDF1"], r["IDs"])
    if geral is not None and len(geral[geral["variante"] == "hit_ref"]):
        h = geral[geral["variante"] == "hit_ref"].iloc[0]
        evo("+ GM (γ = 0,55) + NSA-Kalman = HIT-Tracker (28/09)", h["MOTA"], h["IDF1"], h["IDs"])
    A(tabela(["Etapa", "MOTA", "IDF1", "IDs", "Δ MOTA / Δ IDF1"], rows))
    if rr is not None and len(rr[rr["config"] == "gm_nsa_g055"]):
        r = rr[rr["config"] == "gm_nsa_g055"].iloc[0]
        A(f"Ablação: best.pt + GM (γ = 0,55) + NSA, sem HEM e sem SAHI: MOTA {fmt(r['MOTA'], 2)}, "
          f"IDF1 {fmt(r['IDF1'], 2)}, IDs {fmt(r['IDs'], 0)}.\n")
    gm = rr[rr["grupo"] == "f_gm"].sort_values("gamma") if rr is not None else None
    if gm is not None and len(gm):
        rows = [[fmt(r["gamma"], 2), fmt(r["MOTA"], 2), fmt(r["IDF1"], 2), fmt(r["IDs"], 0), fmt(r["MOTA_antigo"], 1),
                 fmt(r["IDF1_antigo"], 1), fmt(r["IDs_antigo"], 0)] for _, r in gm.iterrows()]
        A("Sensibilidade ao γ (best.pt + GM + NSA, estado zerado por sequência e por γ):\n")
        A(tabela(["γ", "MOTA", "IDF1", "IDs", "MOTA antigo", "IDF1 antigo", "IDs antigo"], rows))

    # 8. Hibrido no MOT e custo -----------------------------------------------------------------
    A("## 8. Arquitetura híbrida no MOT e custo × densidade (tempos de 28/09)\n")
    rows = []
    reg = base.read_csv(p1 / "custo_regressao.csv")
    if geral is not None:
        for _, r in geral.iterrows():
            extra = ["", "", ""]
            if reg is not None:
                s = reg[reg["variante"] == r["variante"]]
                if len(s):
                    s = s.iloc[0]
                    extra = [fmt(s["tempo_medio_ms"], 1), fmt(s["inclinacao_ms_por_alvo"], 2),
                             f"[{fmt(s['ic95_inf'], 2)}; {fmt(s['ic95_sup'], 2)}]"]
            rows.append([r["rotulo"], fmt(r["MOTA"], 2), fmt(r["IDF1"], 2), fmt(r["IDs"], 0), fmt(r["FP"], 0),
                         fmt(r["FN"], 0)] + extra)
    A(tabela(["Variante", "MOTA", "IDF1", "IDs", "FP", "FN", "Tempo médio (ms/quadro)", "Inclinação (ms/alvo)",
              "IC95% BCa"], rows))
    if reg is not None:
        for _, s in reg[reg["tipo"] == "diferenca"].iterrows():
            A(f"Diferença de inclinação {s['variante']}: {fmt(s['inclinacao_ms_por_alvo'], 2)} ms/alvo "
              f"[{fmt(s['ic95_inf'], 2)}; {fmt(s['ic95_sup'], 2)}], p (bootstrap) = {fmt(s['p_bootstrap'], 4)}.\n")
    A("Tempo por quadro medido em 28/09 (sem nova medição). Regressão MQO tempo = a + b·alvos com todos os quadros; "
      "IC por bootstrap de sequências (BCa, 10.000).\n")

    # 9. Predicao no MOT ---------------------------------------------------------------------------
    A("## 9. Predição de trajetória no MOT (correspondência CLEAR MOT)\n")
    pr = base.read_csv(p1 / "mot_pred_resumo.csv")
    labels = cfg["mot"]["variantes"]
    rows = []
    if pr is not None:
        for v in labels:
            for damp in (1.0, 0.95):
                s = pr[(pr["variante"] == v) & (pr["damping"] == damp)].set_index("H")
                if s.empty:
                    continue
                rows.append([labels[v], "1 (persist.)" if damp == 1.0 else "0,95 (amort.)"]
                            + [fmt(g(s.loc[h], "media_seq_norm") if h in s.index else NAN, 3) for h in (5, 10, 15, 30)]
                            + [fmt(g(s.loc[30], "media_seq_px") if 30 in s.index else NAN, 1),
                               fmt(100 * g(s.loc[30], "cobertura") if 30 in s.index else NAN, 1) + "%"])
    A(tabela(["Variante", "γ", "H5 norm.", "H10 norm.", "H15 norm.", "H30 norm.", "H30 (px)", "Cobertura H30"], rows))
    A("Normalizado = erro / diagonal do GT em t+H (média das médias por sequência). A correspondência pelo "
      "IDF1 está só no apêndice (`parte1/mot_pred_apendice_idf1_*.csv`).\n")
    sr = base.read_csv(p1 / "mot_suporte_comum_resumo.csv")
    if sr is not None and len(sr):
        rows = [[f"{labels.get(r['a'], r['a'])} × {labels.get(r['b'], r['b'])}",
                 "1" if r["damping"] == 1.0 else "0,95", int(r["H"]), fmt(r["pares_comuns"], 0),
                 fmt(r["media_norm_a"], 3), fmt(r["media_norm_b"], 3), fmt(r["dif_norm_media"], 3)]
                for _, r in sr[sr["H"] == 30].iterrows()]
        A("Suporte comum (mesmo alvo, mesmo t, mesmo H):\n")
        A(tabela(["Par (a × b)", "γ", "H", "Pares", "norm. a", "norm. b", "a − b"], rows))

    # 10. Estatistica ------------------------------------------------------------------------------
    A("## 10. Estatística (Holm por família; IC BCa 10.000; IC de Bonferroni 1 − 0,05/m)\n")
    est = base.read_csv(out / "estatistica_testes.csv")
    rows = []
    if est is not None:
        for _, r in est.iterrows():
            nd = 3 if str(r.get("unidade")) in ("fração da diagonal", "ρ") else 2
            rows.append([r["familia"], r["teste"], fmt(r["n"], 0), fmt(r["estimativa"], nd),
                         f"[{fmt(r['ic95_inf'], nd)}; {fmt(r['ic95_sup'], nd)}]",
                         f"[{fmt(r['ic_bonf_inf'], nd)}; {fmt(r['ic_bonf_sup'], nd)}]",
                         fmt(r["p_bruto"], 4), fmt(r["p_holm"], 4), fmt(g(r, "rank_biserial"), 2)])
    A(tabela(["Fam.", "Teste", "n", "Estimativa", "IC95% BCa", "IC Bonferroni", "p bruto", "p Holm", "r rank-biserial"],
             rows))

    # 11. Determinismo -------------------------------------------------------------------------------
    A("## 11. Determinismo\n")
    dt = base.read_csv(p2 / "determinismo.csv")
    rows = []
    if dt is not None:
        for _, r in dt.iterrows():
            rows.append([r["seq"], r["comparacao"], str(g(r, "identicos", "")), fmt(g(r, "max_dif_px"), 4),
                         fmt(g(r, "so_em_a"), 0), fmt(g(r, "so_em_b"), 0), fmt(g(r, "dif_mota"), 6),
                         fmt(g(r, "dif_idf1"), 6), fmt(g(r, "dif_num_switches"), 0)])
    A(tabela(["Seq.", "Comparação", "Idênticos", "máx |Δ| (px)", "só em A", "só em B", "Δ MOTA", "Δ IDF1", "Δ IDs"], rows))

    # 12. Cobertura e 13. Diagonal ----------------------------------------------------------------------
    A("## 12. Cobertura\n")
    rows = []
    if res is not None:
        for _, r in res.iterrows():
            rows.append(["DUT", r["rotulo"], fmt(100 * r["cobertura_global"], 1) + "%",
                         fmt(100 * r["H30_cob_pers"], 1) + "%", fmt(r["H30_n_pers"], 0)])
    if pr is not None:
        for v in labels:
            s = pr[(pr["variante"] == v) & (pr["damping"] == 1.0) & (pr["H"] == 30)]
            if len(s):
                s = s.iloc[0]
                rows.append(["MOT", labels[v], "—", fmt(100 * s["cobertura"], 1) + "%", fmt(s["pares"], 0)])
    A(tabela(["Cenário", "Método", "Quadros com saída", "Cobertura dos pares H30", "Pares H30"], rows))
    A("DUT: pares H30 / quadros com alvo em t+H. MOT: pares válidos / passos do filtro (trilha com correspondência "
      "CLEAR em t e alvo presente em t+H).\n")
    A("## 13. Diagonal do alvo\n")
    dj = p1 / "dut_diagonal_resumo.json"
    mj = p1 / "mot_alvos_resumo.json"
    rows = []
    if dj.exists():
        d = base.read_json(dj)
        q = d.get("por_quadro", {})
        rows.append(["DUT (GT oficial, sem −100)", fmt(d.get("quadros_com_alvo"), 0), fmt(q.get("largura_media_px"), 1),
                     fmt(q.get("altura_media_px"), 1), fmt(q.get("diagonal_media_px"), 1),
                     fmt(q.get("diagonal_mediana_px"), 1)])
        A(f"GT oficial: {fmt(d.get('linhas_total'), 0)} linhas, {fmt(d.get('linhas_-100'), 0)} com −100 (excluídas).\n")
    if mj.exists():
        m = base.read_json(mj)
        rows.append(["MOT Track 3 (GT, 30 seq.)", fmt(m.get("caixas_gt"), 0), fmt(m.get("largura_media_px"), 1),
                     fmt(m.get("altura_media_px"), 1), fmt(m.get("diagonal_media_px"), 1), "—"])
    A(tabela(["Cenário", "Caixas", "Largura média (px)", "Altura média (px)", "Diagonal média (px)",
              "Diagonal mediana (px)"], rows))

    # 14. Diferencas -----------------------------------------------------------------------------
    A("## 14. Diferenças em relação aos valores antigos\n")
    itens = []
    if rep is not None:
        for _, r in rep.iterrows():
            dv = r["dif_avaliador"] if np.isfinite(r["dif_avaliador"]) else r["dif_kalman_sige"]
            if np.isfinite(dv) and abs(dv) > (0.051 if "px" in str(r["metrica"]) else 0.0006):
                itens.append(f"{r['tabela']} — {r['metodo']} — {r['metrica']}: publicado {fmt(r['publicado'], 4)}, "
                             f"avaliador {fmt(r['recalc_avaliador'], 4)}, Kalman SIGE {fmt(r['recalc_kalman_sige'], 4)}.")
    if res is not None:
        for _, r in res[res["trajetoria_original"].astype(str) != "True"].iterrows():
            itens.append(f"{r['rotulo']}: trajetória original não encontrada; usada a reexecução "
                         f"(`{str(r['trajetoria']).split('TesteYolo26/')[-1]}`). F3 usa os erros por vídeo salvos.")
    if rr is not None:
        for _, r in rr.iterrows():
            if np.isfinite(g(r, "MOTA_antigo")):
                itens.append(f"{r['descricao']}: MOTA {fmt(r['MOTA_antigo'], 1)} → {fmt(r['MOTA'], 2)}, IDF1 "
                             f"{fmt(r['IDF1_antigo'], 1)} → {fmt(r['IDF1'], 2)}, IDs {fmt(r['IDs_antigo'], 0)} → "
                             f"{fmt(r['IDs'], 0)} (estado zerado por sequência).")
    if geral is not None:
        b = geral[geral["variante"] == "baseline_ref"]
        if len(b):
            b = b.iloc[0]
            itens.append(f"Linha de base BoT-SORT: publicado 75,9 / 71,2 / 2.327 → 28/09 (estado zerado) "
                         f"{fmt(b['MOTA'], 2)} / {fmt(b['IDF1'], 2)} / {fmt(b['IDs'], 0)}.")
    itens.append("FPS: nenhuma coluna de FPS nova; leaderboard e SAHI sem FPS. Custo × densidade usa os tempos de 28/09.")
    itens.append("Removidos do TCC: CLAHE/gama/unsharp/sharpen e B5. OSTrack/ODTrack/DeepSORT não reexecutados.")
    A("\n".join(f"- {x}" for x in itens) + "\n")

    md = "\n".join(L)
    base.write_text(out / "RESUMO_TCC.md", md)

    zpath = out / "resumo_tcc.zip"
    tmp = zpath.with_name(zpath.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out / "RESUMO_TCC.md", "RESUMO_TCC.md")
        for f in sorted(out.glob("*.csv")):
            z.write(f, f.name)
        for sub in ("parte1", "parte2", "figuras"):
            d = out / sub
            if not d.exists():
                continue
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix in (".csv", ".json", ".pdf", ".png", ".md"):
                    z.write(f, f"{sub}/{f.name}")
        z.write(base.HERE / "config.yaml", "config.yaml")
    tmp.replace(zpath)
    base.log(f"RESUMO_TCC.md e resumo_tcc.zip gravados em {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
