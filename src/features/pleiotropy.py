"""The pleiotropy covariate.

Spence et al. showed that genome-wide association studies prioritise genes near
trait-specific variants and can therefore surface highly pleiotropic genes, while
burden tests prioritise trait-specific genes and generally cannot. Pleiotropy
drives constraint. So evidence source, through pleiotropy, to constraint is a
published pathway, and it is exactly
that account. That is why this covariate is central to the mediation in
the mediation rather than a control among many.

Two independent measures, because they fail differently.

**Effective number of traits**, from the Spence per-gene effect estimates across
roughly two hundred UK Biobank traits. A gene acting on one trait has an
effective number near one; a gene acting evenly across many has an effective
number near the trait count. This is the Hill number of order two, so it needs
no significance threshold and no arbitrary cut:

    effective = ( sum_t g_t )^2 / sum_t g_t^2

**Distinct disease parents**, from the Open Targets association index: how many
different top-level disease areas a gene is associated with. This is coarser but
covers genes the UK Biobank traits do not reach.

The Spence estimates are unbiased and so can be negative, which is meaningful
for a single estimate and meaningless for a concentration measure. They are
clipped at zero before the concentration is taken, and the share clipped is
reported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.ingest import open_targets

SPENCE_FILE = "unbiased_gamma_sq_by_trait_MAF1_hgncIDs.txt"


def _spence_path() -> Path:
    return config.raw_dir() / "spence" / "specificity_length_luck" / "data" / SPENCE_FILE


def effective_traits(values: np.ndarray) -> float:
    """Hill number of order two over a gene's per-trait effects."""
    positive = np.clip(values[np.isfinite(values)], 0.0, None)
    total = positive.sum()
    if total <= 0:
        return float("nan")
    return float(total**2 / np.sum(positive**2))


def spence_pleiotropy(path: Path | None = None) -> pd.DataFrame:
    """Effective number of traits per gene, keyed on HGNC identifier."""
    source = path or _spence_path()
    frame = pd.read_csv(source, sep="\t")

    trait_columns = [column for column in frame.columns if column != "hgnc_id"]
    values = frame[trait_columns].to_numpy(dtype=float)

    negative_share = float(np.mean(values < 0)) if values.size else float("nan")

    rows = {
        "hgnc_id": frame["hgnc_id"].astype(str).str.replace("HGNC:", "", regex=False),
        "pleiotropy_effective_traits": [effective_traits(row) for row in values],
        "pleiotropy_total_effect": np.nansum(np.clip(values, 0.0, None), axis=1),
        "n_traits_measured": len(trait_columns),
    }
    table = pd.DataFrame(rows)
    table.attrs["negative_estimate_share"] = negative_share
    table.attrs["n_traits"] = len(trait_columns)
    return table


def platform_disease_breadth() -> pd.DataFrame:
    """Distinct therapeutic areas a gene carries evidence for.

    Taken from the evidence sets rather than from an association table, so the
    count reflects the same rows the primary model treats as positives.
    """
    disease = open_targets.read("disease", columns=["id", "therapeuticAreas"])
    areas = (
        disease.explode("therapeuticAreas")
        .dropna(subset=["therapeuticAreas"])
        .rename(columns={"id": "diseaseId"})
    )

    frames = []
    for dataset in (
        "evidence_orphanet",
        "evidence_gwas_credible_sets",
        "evidence_eva",
        "evidence_gene_burden",
    ):
        try:
            evidence = open_targets.read(dataset, columns=["targetId", "diseaseId"])
        except FileNotFoundError:
            continue
        frames.append(evidence.dropna(subset=["targetId", "diseaseId"]).drop_duplicates())
    if not frames:
        return pd.DataFrame(columns=["ensembl_gene_id", "n_therapeutic_areas", "n_diseases"])

    pairs = pd.concat(frames, ignore_index=True).drop_duplicates()
    joined = pairs.merge(areas, on="diseaseId", how="inner")

    grouped = joined.groupby("targetId")
    return pd.DataFrame(
        {
            "ensembl_gene_id": grouped.size().index,
            "n_therapeutic_areas": grouped["therapeuticAreas"].nunique().to_numpy(),
            "n_diseases": grouped["diseaseId"].nunique().to_numpy(),
        }
    ).reset_index(drop=True)


def build() -> pd.DataFrame:
    from src.features import gene_table

    genes = gene_table.load()[["ensembl_gene_id", "gene_symbol", "hgnc_id"]]

    spence = spence_pleiotropy()
    breadth = platform_disease_breadth()

    table = genes.merge(breadth, on="ensembl_gene_id", how="left")

    # The Spence table is keyed on HGNC and the bridge runs on that identifier
    # directly. The symbol lookup shipped alongside it is a synonym table
    # carrying up to thirty-two aliases per gene, so joining through gene
    # symbols fans one Spence row out across many genes and assigns values that
    # do not belong to them. GeneBayes carries Ensembl and HGNC on the same row,
    # which is a one-to-one bridge.
    before = len(table)
    table = table.merge(
        spence[["hgnc_id", "pleiotropy_effective_traits", "pleiotropy_total_effect"]],
        on="hgnc_id",
        how="left",
    )
    if len(table) != before:
        raise ValueError(
            f"the HGNC join changed the row count from {before} to {len(table)}; "
            "the bridge is not one to one"
        )

    table.attrs.update(spence.attrs)
    return table.drop_duplicates("ensembl_gene_id").reset_index(drop=True)


def run() -> pd.DataFrame:
    table = build()
    target = config.interim_dir() / "pleiotropy.parquet"
    table.to_parquet(target, index=False)

    print(f"pleiotropy: {len(table):,} genes")
    print(f"   traits in the Spence table: {table.attrs.get('n_traits')}")
    print(
        f"   share of per-trait estimates below zero, clipped for the "
        f"concentration measure: {table.attrs.get('negative_estimate_share', float('nan')):.3f}"
    )
    for column in (
        "pleiotropy_effective_traits",
        "pleiotropy_total_effect",
        "n_therapeutic_areas",
        "n_diseases",
    ):
        if column in table.columns:
            print(f"   {column}: {int(table[column].notna().sum()):,} non-null")
    print()
    print(
        table[["pleiotropy_effective_traits", "n_therapeutic_areas", "n_diseases"]]
        .describe()
        .to_string()
    )
    return table


def load() -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / "pleiotropy.parquet")


if __name__ == "__main__":
    config.ensure_dirs()
    run()
