"""Configuration and path resolution.

The data tree lives outside the checkout when SH_DATA_ROOT is set, which is how
the containers are wired. Everything else resolves relative to the repository
root so that a module works the same whether it is imported from a test or run
as a pipeline step.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


def data_root() -> Path:
    return Path(os.environ.get("SH_DATA_ROOT", REPO_ROOT / "data"))


def raw_dir() -> Path:
    return data_root() / "raw"


def interim_dir() -> Path:
    return data_root() / "interim"


def derived_dir() -> Path:
    return data_root() / "derived"


def display_dir() -> Path:
    return data_root() / "display_items"


def docs_dir() -> Path:
    return REPO_ROOT / "docs"


def positives_path() -> Path:
    """The collapsed disease-gene table every model reads."""
    return derived_dir() / "analysis_set" / "positives_collapsed.parquet"


def manifest_path() -> Path:
    return data_root() / "MANIFEST.json"


@cache
def load(name: str) -> dict[str, Any]:
    """Load a YAML file from config/ by stem."""
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sources() -> dict[str, Any]:
    return load("sources")


def analysis() -> dict[str, Any]:
    return load("analysis")


def ensure_dirs() -> None:
    for path in (raw_dir(), interim_dir(), derived_dir(), display_dir()):
        path.mkdir(parents=True, exist_ok=True)
