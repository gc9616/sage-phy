"""Configuration loading. TOML only, so the core needs no third-party packages."""
from __future__ import annotations

import copy
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "configs" / "default.toml"


def repo_path(p: str | Path) -> Path:
    """Paths in configs, object files and CLI defaults are relative to the repo root,
    so the repo runs unchanged on any machine and from any working directory."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load the default config, overlaid with `path` if given."""
    with open(DEFAULT_PATH, "rb") as f:
        cfg = tomllib.load(f)
    if path is not None:
        with open(path, "rb") as f:
            cfg = _merge(cfg, tomllib.load(f))
    return cfg
