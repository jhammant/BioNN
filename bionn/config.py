"""YAML configuration loader with defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(_DEFAULT_PATH) as f:
        cfg = yaml.safe_load(f)
    if path is not None:
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, overrides)
    return cfg
