"""Configuracao, caminhos, importacao do codigo do SIGE e do LAFusion, E/S atomica.

Regra do experimento: o codigo do SIGE (publicar_github/) e do LAFusion (src/)
e IMPORTADO, nunca alterado. Os modulos cujo nome comeca com digito
(04_rastreadores.py, 05_avaliar.py) sao carregados por importlib com um apelido.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Compatibilidade: o uav_common.py do SIGE usa np.trapezoid (NumPy >= 2.0); o .venv do
# LAFusion tem NumPy 1.26, que so tem np.trapz (mesma funcao, mesma assinatura (y, x)).
# O shim evita alterar o codigo do SIGE; o evaluate_mot.py do LAFusion faz o mesmo com asfarray.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

MP_DIR = Path(__file__).resolve().parents[1]      # .../mot_prediction
ROOT = MP_DIR.parent                               # raiz do projeto LAFusion

_MODULES: dict = {}


# ----------------------------------------------------------------------------
# Log e configuracao
# ----------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def load_config(path: str | Path | None = None) -> dict:
    import yaml

    cfg_path = Path(path) if path else MP_DIR / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(cfg_path.resolve())
    cfg["_config_sha256"] = sha256_file(cfg_path)
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Copia de base com as chaves de override sobrescritas recursivamente."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def rpath(value) -> Path | None:
    """Resolve um caminho do config: absoluto, ~ ou relativo a ROOT."""
    if value is None or str(value).strip() == "":
        return None
    path = Path(os.path.expanduser(str(value)))
    return path if path.is_absolute() else (ROOT / path)


def results_dir(cfg: dict) -> Path:
    return rpath(cfg["paths"]["results"])


def weights_path(cfg: dict, key: str) -> Path:
    return rpath(cfg["weights"][key])


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # noqa: BLE001 - torch e opcional nos passos de analise
        pass


def resolve_device(requested: str) -> str:
    """Mesma regra do SIGE (04_rastreadores.py::resolve_device)."""
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------------
# Hash e E/S atomica (uma queda no meio da escrita nao deixa arquivo truncado)
# ----------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def _clean_nan(obj):
    if isinstance(obj, dict):
        return {str(k): _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: str | Path, obj) -> None:
    atomic_write_text(path, json.dumps(_clean_nan(obj), indent=2, ensure_ascii=False,
                                       default=_json_default) + "\n")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(df, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def mark_done(path: str | Path, info: dict | None = None) -> None:
    write_json(path, {"concluido_em": now_iso(), **(info or {})})


def is_done(path: str | Path) -> bool:
    return Path(path).exists()


# ----------------------------------------------------------------------------
# Importacao do SIGE (publicar_github) -- sem modificar nada
# ----------------------------------------------------------------------------

def sige_src(cfg: dict) -> Path:
    return rpath(cfg["paths"]["sige_repo"]) / "src"


def _ensure_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def uav_common(cfg: dict):
    """publicar_github/.../src/uav_common.py (metricas e Kalman do SIGE)."""
    if "uav_common" not in _MODULES:
        _ensure_path(sige_src(cfg))
        import uav_common as module  # noqa: WPS433

        _MODULES["uav_common"] = module
    return _MODULES["uav_common"]


def _load_by_path(alias: str, path: Path):
    if alias in _MODULES:
        return _MODULES[alias]
    if not path.exists():
        raise SystemExit(f"Arquivo de origem nao encontrado: {path}")
    spec = importlib.util.spec_from_file_location(alias, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    _MODULES[alias] = module
    return module


def sige_rastreadores(cfg: dict):
    """publicar_github/.../src/04_rastreadores.py (make_local_tracker, detect_best)."""
    uav_common(cfg)
    return _load_by_path("sige_rastreadores", sige_src(cfg) / "04_rastreadores.py")


def sige_avaliar(cfg: dict):
    """publicar_github/.../src/05_avaliar.py (bootstrap_ci)."""
    uav_common(cfg)
    return _load_by_path("sige_avaliar", sige_src(cfg) / "05_avaliar.py")


# ----------------------------------------------------------------------------
# Importacao do LAFusion (src/) -- sem modificar nada
# ----------------------------------------------------------------------------

def lafusion_root(cfg: dict) -> Path:
    return rpath(cfg["paths"]["lafusion_root"])


def lafusion_common(cfg: dict):
    """src/inference/common.py (split de validacao, leitura MOT, nomes de video)."""
    if "lafusion_common" not in _MODULES:
        root = lafusion_root(cfg)
        _ensure_path(root / "src" / "inference")
        _MODULES["lafusion_common"] = _load_by_path("lafusion_inference_common",
                                                    root / "src" / "inference" / "common.py")
    return _MODULES["lafusion_common"]


def lafusion_flit(cfg: dict):
    """src/postprocessing/robust_flit_association.py (custo Geman-McClure do HIT-Tracker)."""
    root = lafusion_root(cfg)
    _ensure_path(root / "src" / "postprocessing")
    return _load_by_path("lafusion_robust_flit", root / "src" / "postprocessing" / "robust_flit_association.py")


def lafusion_nsa(cfg: dict):
    """src/postprocessing/nsa_kalman_patch.py (NSA-Kalman do HIT-Tracker)."""
    root = lafusion_root(cfg)
    _ensure_path(root / "src" / "postprocessing")
    return _load_by_path("lafusion_nsa_kalman", root / "src" / "postprocessing" / "nsa_kalman_patch.py")


def lafusion_sahi_runner(cfg: dict):
    """src/inference/run_sahi_flit_nsa.py (runner canonico do HIT-Tracker)."""
    root = lafusion_root(cfg)
    _ensure_path(root / "src" / "inference")
    _ensure_path(root / "src")
    _ensure_path(root / "src" / "postprocessing")
    return _load_by_path("lafusion_run_sahi_flit_nsa", root / "src" / "inference" / "run_sahi_flit_nsa.py")


def lafusion_density(cfg: dict):
    """paper_figures/metrics/density.py (definicao de densidade do artigo)."""
    root = lafusion_root(cfg)
    lafusion_common(cfg)
    return _load_by_path("lafusion_density", root / "paper_figures" / "metrics" / "density.py")


# ----------------------------------------------------------------------------
# Estrutura de resultados
# ----------------------------------------------------------------------------

def out_dirs(cfg: dict) -> dict:
    base = results_dir(cfg)
    return {
        "base": base,
        "setup": base / "setup",
        "smoke": base / "smoke",
        "exp_a": base / "exp_a",
        "exp_b": base / "exp_b",
        "h4_dut": base / "h4_dut",
        "summary": base / "summary",
    }


def selection_file(cfg: dict) -> Path:
    return out_dirs(cfg)["setup"] / "trajectory_selection.json"


def selected_trajectories(cfg: dict) -> dict:
    """Le a escolha feita pelo check_setup (quais arquivos entram no Experimento A)."""
    path = selection_file(cfg)
    if not path.exists():
        raise SystemExit(
            f"{path} nao existe. Rode primeiro: python mot_prediction/check_setup.py\n"
            "(ele avalia as trajetorias salvas e escolhe as que reproduzem o artigo)."
        )
    return read_json(path)


def environment_info() -> dict:
    """Versoes de bibliotecas e maquina, para o hand-off de reprodutibilidade."""
    import platform
    import subprocess

    info = {
        "python": sys.version.split()[0],
        "executavel": sys.executable,
        "plataforma": platform.platform(),
        "maquina": platform.machine(),
    }
    try:
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        if brand:
            info["cpu"] = brand
    except Exception:  # noqa: BLE001
        pass
    packages = {}
    try:
        from importlib import metadata

        for name in ("ultralytics", "torch", "torchvision", "numpy", "pandas", "scipy", "motmetrics",
                     "sahi", "matplotlib", "PyYAML", "lap", "lapx", "opencv-python",
                     "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
    except Exception:  # noqa: BLE001
        pass
    info["pacotes"] = packages
    try:
        import cv2

        info["cv2_version"] = cv2.__version__
    except Exception:  # noqa: BLE001
        info["cv2_version"] = None
    return info
