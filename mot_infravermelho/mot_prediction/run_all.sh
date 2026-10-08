#!/usr/bin/env bash
#
# Experimento mot_prediction (TCC): executa as etapas em ordem.
#
#   0  check_setup.py          verificacao do ambiente e dos dados (sempre roda; e rapida na 2a vez)
#   1  steps/s1_smoke.py       teste rapido em 1 sequencia + estimativa do tempo das demais etapas
#   2  steps/s2_exp_a.py       Experimento A (trajetorias salvas da linha de base e do HIT-Tracker)
#   3  steps/s3_exp_b.py       Experimento B principal (N_red = 15, CSRT) + referencia visivel de H4
#   4  steps/s4_sweeps.py      varreduras opcionais (N_red 5/10/30, KCF, MOSSE, HIT com tempo por quadro)
#   -> make_summary.py         results/summary/ (e results/summary_<data>.zip): o que volta para analise
#
# Uso (a partir da raiz do projeto LAFusion):
#   bash mot_prediction/run_all.sh --smoke          # so as etapas 0 e 1
#   bash mot_prediction/run_all.sh                  # tudo
#   bash mot_prediction/run_all.sh --skip-sweeps    # etapas 0 a 3 + resumo
#
# Retomavel: etapa ou sequencia ja concluida e pulada (marcadores _DONE_*.json e
# results/**/done/<seq>.json). Se a execucao cair, rode o MESMO comando de novo.
# Nada e apagado em nenhum caso.
#
# No macOS o script se reexecuta sob 'caffeinate -dimsu' para a maquina nao dormir.
# Ambiente: o .venv do LAFusion (raiz do projeto). O setup_env.sh confere e, se faltar o
# CSRT, troca o opencv-python pelo opencv-contrib-python da mesma versao (~1 min).
# Para usar outro interpretador: --python CAMINHO (ou a variavel PYTHON).

set -uo pipefail

# ---------------------------------------------------------------- caffeinate
if command -v caffeinate >/dev/null 2>&1 && [ -z "${MOTPRED_CAFFEINATED:-}" ]; then
  export MOTPRED_CAFFEINATED=1
  exec caffeinate -dimsu bash "$0" "$@"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

SMOKE=0
SKIP_SWEEPS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke)        SMOKE=1; shift ;;
    --skip-sweeps)  SKIP_SWEEPS=1; shift ;;
    --python)       PYTHON="$2"; shift 2 ;;
    -h|--help)      sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "opcao desconhecida: $1 (use --smoke, --skip-sweeps ou --python CAMINHO)"; exit 2 ;;
  esac
done

export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

RESULTS="$HERE/results"
LOGS="$HERE/logs"
mkdir -p "$RESULTS" "$LOGS"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGS/run_${STAMP}.log"
TIMES="$LOGS/stage_times.csv"
[ -f "$TIMES" ] || echo "etapa,inicio,fim,duracao_s,codigo_saida,log" > "$TIMES"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

# run_stage <nome> <marcador | -> <comando ...>
run_stage() {
  local name="$1" marker="$2"
  shift 2
  if [ "$marker" != "-" ] && [ -f "$marker" ]; then
    log "PULANDO etapa '$name' (ja concluida: ${marker#"$ROOT"/})"
    return 0
  fi
  log "INICIO etapa '$name'"
  log "  comando: $*"
  local start end status start_h
  start=$(date +%s)
  start_h="$(ts)"
  "$@" 2>&1 | tee -a "$LOG"
  status=${PIPESTATUS[0]}
  end=$(date +%s)
  echo "$name,$start_h,$(ts),$((end - start)),$status,$(basename "$LOG")" >> "$TIMES"
  if [ "$status" -ne 0 ]; then
    log "ERRO na etapa '$name' (codigo $status) depois de $((end - start)) s."
    log ">>> Nada foi apagado. Leia a mensagem logo acima, corrija e rode o MESMO comando de novo:"
    log ">>> etapas e sequencias ja concluidas serao puladas."
    log ">>> Log completo: ${LOG#"$ROOT"/}"
    exit "$status"
  fi
  log "FIM etapa '$name' ($((end - start)) s)"
}

log "mot_prediction -- raiz: $ROOT"

# ---------------------------------------------------------------- ambiente
# Sem --python, usa SEMPRE o .venv do LAFusion (conferido pelo setup_env.sh, que so
# acrescenta o OpenCV contrib se faltar o CSRT); nunca o python3 do sistema ou do Anaconda.
if [ -z "${PYTHON:-}" ]; then
  run_stage "ambiente" "-" bash "$HERE/setup_env.sh"
  PYTHON="$ROOT/.venv/bin/python"
fi
log "python: $PYTHON ($("$PYTHON" --version 2>&1))"
log "modo: $([ $SMOKE -eq 1 ] && echo 'teste rapido' || echo 'completo')$([ $SKIP_SWEEPS -eq 1 ] && echo ', sem varreduras')"

run_stage "0-check_setup" "-" "$PYTHON" "$HERE/check_setup.py"
run_stage "1-teste_rapido" "$RESULTS/smoke/_DONE_smoke.json" "$PYTHON" "$HERE/steps/s1_smoke.py"

# estimativa (tambem quando o teste rapido ja tinha sido feito antes)
"$PYTHON" - "$RESULTS/smoke/estimate.json" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
from datetime import datetime, timedelta
try:
    est = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(0)
print("\nEstimativa das etapas seguintes (min):")
for k, v in est["minutos"].items():
    print(f"  {k:<45s} {v:8.1f}")
m, s = est["total_sem_varreduras_min"], est["total_varreduras_min"]
now = datetime.now()
print(f"  etapas 2-3: {m:.0f} min (termino ~{(now + timedelta(minutes=m)):%H:%M}); "
      f"com varreduras: {m + s:.0f} min (termino ~{(now + timedelta(minutes=m + s)):%H:%M})\n")
PY

if [ "$SMOKE" -eq 1 ]; then
  log "Teste rapido concluido. Relatorio: ${RESULTS#"$ROOT"/}/smoke/smoke_report.md"
  log "Para a noite: bash mot_prediction/run_all.sh   (ou --skip-sweeps)"
  exit 0
fi

run_stage "2-experimento_A" "$RESULTS/exp_a/_DONE_exp_a.json" "$PYTHON" "$HERE/steps/s2_exp_a.py"
run_stage "3-experimento_B" "-" "$PYTHON" "$HERE/steps/s3_exp_b.py"
if [ "$SKIP_SWEEPS" -eq 0 ]; then
  run_stage "4-varreduras" "-" "$PYTHON" "$HERE/steps/s4_sweeps.py"
else
  log "Etapa 4 (varreduras) pulada por --skip-sweeps."
fi
run_stage "5-resumo" "-" "$PYTHON" "$HERE/make_summary.py"

ZIP="$RESULTS/summary_${STAMP}.zip"
( cd "$RESULTS" && zip -qr "$(basename "$ZIP")" summary ) && log "Pacote para analise: ${ZIP#"$ROOT"/}"
log "CONCLUIDO. Traga de volta a pasta mot_prediction/results/summary/ (ou o .zip acima)."
