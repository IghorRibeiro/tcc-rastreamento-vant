#!/usr/bin/env bash
# =============================================================================
# ligar_dados.sh -- liga (links simbólicos) os dados, pesos, saídas salvas e o ambiente
# dos dois projetos originais a este repositório, sem copiar nada. Os links ficam
# fora do git (.gitignore).
#
#   bash ligar_dados.sh [pasta do DroneTrackingMOT] [pasta do TesteYolo26]
#   (padrão: ~/PycharmProjects/DroneTrackingMOT e ~/PycharmProjects/TesteYolo26)
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOT="${1:-$HOME/PycharmProjects/DroneTrackingMOT}"
SIGE="${2:-$HOME/PycharmProjects/TesteYolo26}"
M="$REPO/mot_infravermelho"

ligar() {   # ligar <alvo existente> <link>
  local alvo="$1" link="$2"
  if [[ ! -e "$alvo" ]]; then echo "AVISO: não encontrado: $alvo"; return 0; fi
  if [[ -e "$link" || -L "$link" ]]; then echo "já existe: $link"; return 0; fi
  ln -s "$alvo" "$link"
  echo "ok: ${link#$REPO/} -> $alvo"
}

# Track 3 (MOT no infravermelho)
ligar "$MOT/dataset"                 "$M/dataset"                    # vídeos, gt.txt e dataset YOLO
ligar "$MOT/weights"                 "$M/weights"                    # best.pt e best_hardmining.pt
ligar "$MOT/runs"                    "$M/runs"                       # runs/mot_eval (fatores da causa raiz e resultados antigos)
ligar "$MOT/mot_prediction/results"  "$M/mot_prediction/results"     # trajetórias e tempos de 28/09
ligar "$MOT/audit"                   "$M/audit"                      # manifestos lidos pelo check_setup do mot_prediction
ligar "$MOT/.venv"                   "$M/.venv"                      # ambiente usado pelo run_hoje.sh (opcional)

# DUT Anti-UAV (SIGE)
ligar "$SIGE/publicar_github/hybrid-uav-tracking-prediction/pesos_treinados" \
      "$REPO/sot_visivel/hybrid-uav-tracking-prediction/pesos_treinados"
for d in datasets avaliacao_tracking pesos_treinados hitratepos; do
  ligar "$SIGE/$d" "$REPO/sot_visivel/projeto_sige/$d"
done

CFG="$M/tcc_hoje/config.yaml"
echo
echo "A reexecução final (tcc_hoje) lê o projeto do SIGE pelo caminho paths.teste_yolo26 em:"
echo "  $CFG"
grep -n "teste_yolo26:" "$CFG" || true
if [[ "$SIGE" != "$HOME/PycharmProjects/TesteYolo26" ]]; then
  echo "Ajuste esse caminho para: $SIGE"
fi
