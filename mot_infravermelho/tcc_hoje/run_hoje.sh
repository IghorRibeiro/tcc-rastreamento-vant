#!/usr/bin/env bash
# =============================================================================
# tcc_hoje/run_hoje.sh -- reexecucao enxuta do TCC (Partes 1 -> 2 -> 3 -> 4 + resumo)
#
#   bash tcc_hoje/run_hoje.sh            # completo  -> tcc_hoje/resultados/
#   bash tcc_hoje/run_hoje.sh --smoke    # teste rapido -> tcc_hoje/resultados_smoke/
#
# Retomavel: cada etapa grava _marcadores/<etapa>.done; a Parte 2 tambem pula as
# tarefas (config, sequencia) ja feitas. Para refazer uma etapa, apague o marcador.
# Roda sob caffeinate (macOS). Aborta se houver menos de 3 GB livres no disco.
# Usa o .venv do DroneTrackingMOT, sem instalar nada.
# =============================================================================
set -o pipefail      # sem -u: o bash 3.2 do macOS trata array vazio como variavel nao definida

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/.venv/bin/python"

SMOKE=0
for a in "$@"; do [[ "$a" == "--smoke" ]] && SMOKE=1; done

# -- caffeinate: impede o Mac de dormir durante a execucao -------------------------
if [[ "$(uname)" == "Darwin" && -z "${TCC_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null 2>&1; then
  export TCC_CAFFEINATED=1
  exec caffeinate -dimsu /bin/bash "$0" "$@"
fi

OUT="$HERE/resultados"; FLAG=""
if [[ $SMOKE == 1 ]]; then OUT="$HERE/resultados_smoke"; FLAG="--smoke"; fi
mkdir -p "$OUT/_marcadores" "$OUT/logs"
LOG="$OUT/logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1        # nada de __pycache__ em TesteYolo26 / publicar_github / src
export YOLO_VERBOSE=False
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

check_disk() {
  local kb
  kb=$(df -Pk "$HERE" | awk 'NR==2 {print $4}')
  if [[ -z "$kb" ]]; then say "AVISO: nao consegui ler o espaco livre"; return 0; fi
  if (( kb < 3 * 1024 * 1024 )); then
    say "ABORTANDO: menos de 3 GB livres em $(df -Pk "$HERE" | awk 'NR==2 {print $6}') ($((kb / 1024)) MB)."
    exit 2
  fi
  say "Disco livre: $((kb / 1024 / 1024)) GB"
}

# -- processos da Parte 2: 2 se RAM <= 8 GB, 3 se maior ------------------------------
MEM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
MEM_GB=$(( MEM_BYTES / 1024 / 1024 / 1024 ))
WORKERS=3
if (( MEM_GB <= 8 )); then WORKERS=2; fi

say "tcc_hoje | modo: $([[ $SMOKE == 1 ]] && echo 'TESTE RAPIDO' || echo 'COMPLETO') | saida: $OUT"
say "RAM: ${MEM_GB} GB -> ${WORKERS} processos na Parte 2 | log: $LOG"
if [[ ! -x "$PY" ]]; then say "ERRO: $PY nao encontrado"; exit 1; fi
"$PY" - <<'EOF'
import importlib.util, sys
need = ["numpy", "pandas", "scipy", "motmetrics", "ultralytics", "sahi", "cv2", "matplotlib", "yaml", "torch"]
miss = [m for m in need if importlib.util.find_spec(m) is None]
print("Python", sys.version.split()[0], "| faltando:", miss or "nada")
sys.exit(1 if miss else 0)
EOF
[[ $? -ne 0 ]] && { say "ERRO: pacotes faltando no .venv"; exit 1; }
check_disk

FALHAS=()
etapa() {   # etapa <nome> <modulo> [args...]
  local nome="$1" mod="$2"; shift 2
  local marker="$OUT/_marcadores/$nome.done"
  if [[ -f "$marker" ]]; then say "[pula] $nome (ja concluida)"; return 0; fi
  check_disk
  say "==> $nome"
  local t0=$SECONDS
  (cd "$HERE" && "$PY" -m "tcc.$mod" $FLAG "$@")
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') $((SECONDS - t0)) s" > "$marker"
    say "<== $nome ok ($((SECONDS - t0)) s)"
  else
    say "<== $nome FALHOU (codigo $rc; $((SECONDS - t0)) s) -- seguindo; rode de novo para retomar"
    FALHAS+=("$nome")
  fi
}

T0=$SECONDS
# Parte 1 (CPU, sem rastreamento)
etapa p1a_sot          p1a_sot
etapa p1b_mot          p1b_mot
etapa p1c_causa        p1c_causa
etapa p1d_custo        p1d_custo
# Parte 2 (rodadas novas, so acuracia)
etapa p2_rodadas       p2_rodadas --workers "$WORKERS"
# Parte 3, Parte 4 e resumo: baratos, refeitos sempre que chegarem aqui
rm -f "$OUT/_marcadores/p3_estatistica.done" "$OUT/_marcadores/p4_figuras.done" "$OUT/_marcadores/resumo.done"
etapa p3_estatistica   p3_estatistica
etapa p4_figuras       p4_figuras
etapa resumo           resumo

say "Tempo total: $(( (SECONDS - T0) / 60 )) min"
if (( ${#FALHAS[@]} )); then
  say "Etapas com falha: ${FALHAS[*]}  -> rode o mesmo comando de novo para retomar"
  exit 1
fi
say "Pronto: $OUT/RESUMO_TCC.md e $OUT/resumo_tcc.zip"
