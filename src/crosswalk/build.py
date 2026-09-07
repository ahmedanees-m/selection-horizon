"""Ontology crosswalk.

Target vocabulary is Mondo. Release 26.06 of the Platform ships EFO 3.88.0, in
which a block of EFO disease terms was replaced by Mondo terms, so anything
keyed on a deprecated EFO identifier drops rows without raising.

The ORPHA leg is cheap because both ends publish it: the Platform disease index
carries Orphanet cross-references, and Orphanet product1 carries references out
to OMIM, ICD-10, MeSH, MedDRA, UMLS and GARD with an explicit mapping relation.
The expensive legs, Pharmaprojects MeSH and FinnGen ICD-10, are built in
src/crosswalk/adjudicate.py against the manual decisions recorded in
config/ontology.yaml.
"""

from __future__ import annotations

import re

import pandas as pd

from src import config
from src.ingest import open_targets

ORPHA_XREF = re.compile(r"^Orphanet[:_](\d+)$")


def orpha_to_platform() -> pd.DataFrame:
    """ORPHAcode to Platform disease identifier, from the Platform disease index.

    One ORPHAcode can reach several Platform terms and one Platform term can
    carry several ORPHAcodes. Both are kept; the consumer decides how to
    resolve, and the multiplicity is reported so that a fan-out is visible in
    the row-count assertions.
    """
    disease = open_targets.read("disease", columns=["id", "name", "dbXRefs", "therapeuticAreas"])
    exploded = disease.explode("dbXRefs").dropna(subset=["dbXRefs"])
    matched = exploded["dbXRefs"].astype(str).str.extract(ORPHA_XREF, expand=False)
    frame = exploded.loc[matched.notna()].copy()
    frame["orpha_code"] = matched.loc[matched.notna()].astype(int)
    frame = frame.rename(columns={"id": "disease_id", "name": "disease_name"})
    return frame[["orpha_code", "disease_id", "disease_name", "therapeuticAreas"]].reset_index(
        drop=True
    )


def platform_vocabulary_report() -> pd.DataFrame:
    """Prefix counts over the Platform disease index.

    The Mondo resolution threshold applies to the great majority of diseases
    that carry evidence. This reports the composition of the whole index; the
    check applies to the diseases actually used.
    """
    disease = open_targets.read("disease", columns=["id"])
    prefixes = disease["id"].str.split("_").str[0]
    return prefixes.value_counts().rename_axis("prefix").reset_index(name="n_terms")


def write() -> None:
    out_dir = config.interim_dir() / "crosswalk"
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = orpha_to_platform()
    mapping.to_parquet(out_dir / "orpha_to_platform.parquet", index=False)
    print(
        f"orpha_to_platform: {len(mapping):,} rows, "
        f"{mapping['orpha_code'].nunique():,} ORPHAcodes, "
        f"{mapping['disease_id'].nunique():,} Platform terms"
    )

    report = platform_vocabulary_report()
    report.to_parquet(out_dir / "platform_vocabulary.parquet", index=False)


if __name__ == "__main__":
    config.ensure_dirs()
    write()
