"""Configuration loading. TOML only, so the core needs no third-party packages."""
from __future__ import annotations

import copy
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.toml"


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
