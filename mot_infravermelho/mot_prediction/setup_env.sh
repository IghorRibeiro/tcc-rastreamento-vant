#!/usr/bin/env bash
#
# Prepara o ambiente do experimento usando o .venv NORMAL do LAFusion (raiz do projeto).
# Chamado sozinho pelo run_all.sh; tambem pode ser rodado a mao:
#
#   bash mot_prediction/setup_env.sh
#
# Unica mudanca no .venv, e so se faltar: o CSRT/KCF/MOSSE ficam no pacote contrib do
# OpenCV. Se o .venv tiver o 'opencv-python', ele e trocado pelo 'opencv-contrib-python'
# da MESMA versao (4.11.0.86 hoje). O contrib contem todo o opencv-python e mais os
# modulos extras, entao o codigo do LAFusion (GMC do BoT-SORT, SAHI, leitura de video)
# continua igual. Para desfazer:
#   .venv/bin/python -m pip uninstall -y opencv-contrib-python
#   .venv/bin/python -m pip install opencv-python==4.11.0.86

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/.venv/bin/python"

say()  { echo "[setup_env $(date '+%H:%M:%S')] $*"; }
fail() { echo; echo ">>> ERRO no preparo do ambiente: $*"; echo ">>> Nada foi apagado."; exit 1; }

[ -x "$PY" ] || fail "nao achei o .venv do LAFusion em $ROOT/.venv"

check() {   # imprime o que falta; codigo 0 se tudo ok, 2 se so falta o CSRT, 1 se falta pacote
  "$PY" - <<'EOF_PY'
import sys
missing = []
for name in ("cv2", "torch", "ultralytics", "motmetrics", "sahi", "scipy", "pandas", "yaml", "matplotlib", "lap"):
    try:
        __import__(name)
    except Exception as error:  # noqa: BLE001
        missing.append(f"{name} ({error})")
if missing:
    print("  faltam pacotes: " + "; ".join(missing))
    sys.exit(1)
import cv2, torch, ultralytics  # noqa: E401
legacy = getattr(cv2, "legacy", None)
if not (hasattr(cv2, "TrackerCSRT_create") or (legacy is not None and hasattr(legacy, "TrackerCSRT_create"))):
    print(f"  OpenCV {cv2.__version__} sem CSRT (falta o pacote contrib)")
    sys.exit(2)
mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
print(f"  .venv ok: python {sys.version.split()[0]} | torch {torch.__version__} (MPS={mps}) | "
      f"ultralytics {ultralytics.__version__} | OpenCV {cv2.__version__} com CSRT")
EOF_PY
}

check
status=$?
if [ "$status" -eq 0 ]; then
  say "ambiente pronto ($ROOT/.venv)"
  exit 0
fi
if [ "$status" -eq 1 ]; then
  fail "o .venv do LAFusion nao tem algum pacote listado acima; instale-o no .venv e rode de novo"
fi

# ---- so falta o CSRT: troca opencv-python pelo contrib da mesma versao
VERSION="$("$PY" -m pip show opencv-python 2>/dev/null | sed -n 's/^Version: //p')"
[ -n "$VERSION" ] || VERSION="$("$PY" -m pip show opencv-python-headless 2>/dev/null | sed -n 's/^Version: //p')"
[ -n "$VERSION" ] || VERSION="4.11.0.86"
say "trocando opencv-python $VERSION por opencv-contrib-python $VERSION no .venv (mesma versao, mais o modulo de rastreadores)"
"$PY" -m pip install --no-deps "opencv-contrib-python==$VERSION" || fail "pip install opencv-contrib-python==$VERSION"
"$PY" -m pip uninstall -y opencv-python opencv-python-headless >/dev/null 2>&1
"$PY" -m pip install --no-deps --force-reinstall "opencv-contrib-python==$VERSION" || \
  fail "reinstalacao do opencv-contrib-python==$VERSION"

check || fail "a troca do OpenCV terminou, mas a conferencia acima falhou"
say "ambiente pronto ($ROOT/.venv)"
