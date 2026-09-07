"""Resolve HPO terms against the ontology rather than from memory.

A wrong ontology identifier is the most dangerous class of error this pipeline
can carry: it changes a result by an amount that looks plausible and it fails no
downstream assertion. One was already found here, where HP:0011712 was taken for
sudden cardiac death when it is complete right bundle branch block.

The response is to stop writing identifiers down. Term sets are defined by the
concept name, resolved against `hp.obo` at load time, and expanded over the
ontology's own subclass relations. The resolved identifier and label are carried
into the output so that any later reader can check what was actually used.
"""

from __future__ import annotations

from pathlib import Path

from src import config, ontology

Ontology = ontology.Ontology
PREFIX = "HP"


def _path() -> Path:
    return config.raw_dir() / "hpo" / "hp.obo"


def load(path: str | None = None) -> Ontology:
    """The HPO release, read through the generic OBO reader."""
    return ontology.load(str(Path(path) if path else _path()), PREFIX)


load.cache_clear = ontology.load.cache_clear  # type: ignore[attr-defined]


def resolve_set(labels: list[str], path: str | None = None) -> dict[str, set[str]]:
    """Map each concept name to the subtree of terms beneath it."""
    release = load(path)
    return {label: release.subtree_by_name(label)[1] for label in labels}


def provenance(labels: list[str], path: str | None = None) -> list[dict]:
    """What each concept resolved to, for the record."""
    rows = ontology.provenance(load(path), labels)
    return [{**row, "hpo_id": row.pop("term_id")} for row in rows]
