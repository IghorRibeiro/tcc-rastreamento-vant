"""Pacote do experimento mot_prediction (predicao de trajetoria no cenario MOT).

Modulos:
    common      configuracao, caminhos, importacao do SIGE e do LAFusion, E/S atomica
    data        leitura de GT/trajetorias, sequencias de validacao, video, densidade
    mot_eval    avaliador MOT identico ao evaluate_mot.py + correspondencias CLEAR/IDF1
    prediction  Kalman do SIGE por trilha e erro de predicao com identidade
    hybrid      arquitetura hibrida (YOLO26n + rastreador local + hungaro GM) para MOT
    reference   rodadas de referencia (linha de base e HIT-Tracker) com tempo por quadro
    drift       deriva do rastreador local (H4)
    dut_sige    copia instrumentada do hibrido do SIGE para o DUT Anti-UAV (H4)
    evaluate    avaliacao completa de uma trajetoria (MOT + predicao + suporte comum)
    stats       testes estatisticos
"""
