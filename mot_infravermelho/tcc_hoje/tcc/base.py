"""Base comum do tcc_hoje: caminhos, configuracao, log, E/S e acesso ao motpred.

Regras: TesteYolo26, publicar_github/, src/ do LAFusion e mot_prediction/ sao so
LIDOS. sys.dont_write_bytecode impede que a importacao grave __pycache__ neles.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

# Mesmos shims do mot_prediction (NumPy 1.26 do .venv): o avaliador do SIGE usa
# np.trapezoid e o py-motmetrics usa np.asfarray.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

HERE = Path(__file__).resolve().parents[1]      # .../DroneTrackingMOT/tcc_hoje
ROOT = HERE.parent                               # .../DroneTrackingMOT
MP_DIR = ROOT / "mot_prediction"
if str(MP_DIR) not in sys.path:
    sys.path.insert(0, str(MP_DIR))

_CACHE: dict = {}


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_cfg() -> dict:
    if "cfg" not in _CACHE:
        import yaml

        _CACHE["cfg"] = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    return _CACHE["cfg"]


def P(value) -> Path:
    """Resolve caminho do config: {root}, {teste}, ~, absoluto ou relativo a tcc_hoje/."""
    cfg = load_cfg()
    text = str(value).replace("{root}", str(ROOT)).replace("{teste}", str(cfg["paths"]["teste_yolo26"]))
    path = Path(os.path.expanduser(text))
    return path if path.is_absolute() else (HERE / path).resolve()


def out_dir(smoke: bool) -> Path:
    base = HERE / ("resultados_smoke" if smoke else "resultados")
    base.mkdir(parents=True, exist_ok=True)
    return base


def sub(smoke: bool, name: str) -> Path:
    path = out_dir(smoke) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def mp_cfg() -> dict:
    """Config do mot_prediction (caminhos do Track 3, pesos, LAFusion, SIGE)."""
    if "mp_cfg" not in _CACHE:
        from motpred import common as mc

        _CACHE["mp_cfg"] = mc.load_config()
    return _CACHE["mp_cfg"]


def val_sequences(smoke: bool) -> list[str]:
    from motpred import data

    seqs = data.val_sequences(mp_cfg())
    if smoke:
        wanted = load_cfg()["smoke"]["mot_sequences"]
        seqs = [s for s in wanted if s in seqs]
    return seqs


def load_module(alias: str, path: Path):
    if alias in _CACHE:
        return _CACHE[alias]
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Arquivo nao encontrado: {path}")
    for extra in (path.parent, path.parent.parent):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location(alias, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    _CACHE[alias] = module
    return module


# ---------------------------------------------------------------------------
# E/S atomica
# ---------------------------------------------------------------------------

def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path, obj) -> None:
    write_text(path, json.dumps(_clean(obj), indent=2, ensure_ascii=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(df, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_csv(path, **kw):
    import pandas as pd

    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception:  # noqa: BLE001
        return None


def smoke_flag(argv=None) -> bool:
    argv = sys.argv[1:] if argv is None else argv
    return "--smoke" in argv


def arg_value(name: str, default=None, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if name in argv:
        idx = argv.index(name)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return default
