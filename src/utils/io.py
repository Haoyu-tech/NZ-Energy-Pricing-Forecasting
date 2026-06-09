"""Config loading and common IO helpers."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# Project root = three levels up from this file (src/utils/io.py -> project root)
ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | None = None) -> dict:
    """Read config.yaml and expand its relative paths to absolute paths."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def abspath(rel: str) -> Path:
    """Turn a project-root-relative path into an absolute path."""
    p = ROOT / rel
    return p


def ensure_dir(path: str | os.PathLike) -> Path:
    """Ensure a directory exists. Accepts a directory path, or a file path (creates its parent)."""
    p = Path(path)
    target = p if p.suffix == "" else p.parent
    target.mkdir(parents=True, exist_ok=True)
    return p
