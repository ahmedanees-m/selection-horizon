"""Ingest for sources that arrive as a single delimited table.

GeneBayes s_het, gnomAD constraint at v4.1 and v2.1.1, the HPO annotation file
and the GENCODE annotation all fall into this shape. Each is downloaded once,
hashed into the manifest, and normalised to a Parquet file with the gene
identifier the rest of the pipeline joins on.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pandas as pd

from src import config
from src.ingest import fetch


def _download(group: str, name: str) -> Path:
    spec = config.sources()[group]
    entry = spec["files"][name]
    dest = config.raw_dir() / group / Path(entry["url"].split("?")[0]).name
    if dest.name == "content":
        dest = dest.with_name(f"{name}.tsv")
    return fetch.download(
        entry["url"],
        dest,
        source=group,
        version=entry.get("version", spec.get("version", "")),
        licence=spec["licence"],
        redistribute=spec["redistribute"],
    )


def _open_text(path: Path):
    if path.suffix in {".gz", ".bgz"}:
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def genebayes() -> pd.DataFrame:
    """Per-gene selection coefficient on heterozygotes. Primary constraint metric."""
    path = _download("genebayes", "s_het")
    frame = pd.read_csv(path, sep="\t")
    frame.columns = [column.strip().lower() for column in frame.columns]
    return frame


def gnomad_constraint(version: str = "v4") -> pd.DataFrame:
    """LOEUF, pLI and missense Z. Secondary constraint metrics."""
    name = "constraint_v4" if version == "v4" else "constraint_v2"
    path = _download("gnomad", name)
    with _open_text(path) as handle:
        frame = pd.read_csv(handle, sep="\t", low_memory=False)
    return frame


def hpo_annotations() -> pd.DataFrame:
    """phenotype.hpoa, including the onset and clinical-course annotations.

    The clinical-course sub-ontology supplies the per-case severity term of the
    fitness-cost model, since Orphanet does not publish an age of death.
    """
    path = _download("hpo", "annotations")
    with _open_text(path) as handle:
        frame = pd.read_csv(handle, sep="\t", comment="#", low_memory=False)
    frame.columns = [column.strip().lstrip("#").lower() for column in frame.columns]
    return frame


def hpo_ontology() -> Path:
    """The ontology file itself, kept in the raw layer for label verification."""
    return _download("hpo", "ontology")


def run() -> None:
    out_dir = config.interim_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    hpo_ontology()

    for name, loader in (
        ("genebayes_s_het", genebayes),
        ("gnomad_constraint_v4", lambda: gnomad_constraint("v4")),
        ("gnomad_constraint_v2", lambda: gnomad_constraint("v2")),
        ("hpo_annotations", hpo_annotations),
    ):
        frame = loader()
        target = out_dir / f"{name}.parquet"
        frame.to_parquet(target, index=False)
        print(f"{name}: {len(frame):,} rows, {frame.shape[1]} columns")


def decompress(source: Path, target: Path) -> Path:
    with gzip.open(source, "rb") as reader, target.open("wb") as writer:
        shutil.copyfileobj(reader, writer)
    return target


if __name__ == "__main__":
    config.ensure_dirs()
    run()
