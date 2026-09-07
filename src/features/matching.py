"""Matched control genes.

For each positive, k controls are drawn matched on confounders only: CDS length
decile, expected loss-of-function count decile, and GC content. Nothing that the
causal graph classifies as a mediator or a collider enters the matching, because
matching on a mediator would remove part of the effect being estimated.

Matching is without replacement within a disease. Standardised mean differences
are reported for every covariate, including the ones deliberately left
unmatched, so that the balance achieved and the balance not attempted are both
visible.

Both the matched and the unmatched analyses are primary outputs. They bound the
answer from opposite directions: unmatched risks confounding, matched risks
over-adjustment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GC_BINS = 10


def _strata(genes: pd.DataFrame) -> pd.Series:
    """Coarse exact-matching strata over the three confounders."""
    gc_decile = np.ceil(genes["gc_content"].rank(method="average", pct=True) * GC_BINS).clip(
        1, GC_BINS
    )
    return (
        genes["cds_length_decile"].astype("Int64").astype(str)
        + "|"
        + genes["expected_lof_decile"].astype("Int64").astype(str)
        + "|"
        + gc_decile.astype("Int64").astype(str)
    )


def draw_controls(
    positives: pd.DataFrame,
    genes: pd.DataFrame,
    k: int,
    seed: int,
    gene_column: str = "gene_id",
    disease_column: str = "disease_id",
) -> pd.DataFrame:
    """Return positives and their matched controls, labelled by `y`.

    Controls are drawn from genes in the same confounder stratum that are not
    positives for that disease. Where a stratum cannot supply k controls, the
    shortfall is recorded rather than filled from a different stratum, because
    silently relaxing the match is how a matched analysis stops being one.
    """
    rng = np.random.default_rng(seed)

    pool = genes.dropna(subset=["cds_length_decile", "expected_lof_decile", "gc_content"]).copy()
    pool["stratum"] = _strata(pool)
    by_stratum = {
        name: group["ensembl_gene_id"].to_numpy() for name, group in pool.groupby("stratum")
    }
    stratum_of = dict(zip(pool["ensembl_gene_id"], pool["stratum"], strict=True))

    rows = []
    shortfalls = 0
    for disease, group in positives.groupby(disease_column):
        positive_genes = set(group[gene_column]) & set(stratum_of)
        if not positive_genes:
            continue
        taken = set(positive_genes)
        for gene in sorted(positive_genes):
            rows.append({disease_column: disease, "gene_id": gene, "y": 1})
            candidates = by_stratum[stratum_of[gene]]
            available = np.array([g for g in candidates if g not in taken], dtype=object)
            if len(available) < k:
                shortfalls += 1
            chosen = rng.choice(available, size=min(k, len(available)), replace=False)
            for control in chosen:
                taken.add(control)
                rows.append({disease_column: disease, "gene_id": control, "y": 0})

    matched = pd.DataFrame(rows)
    matched.attrs["control_shortfalls"] = shortfalls
    return matched


def draw_random_controls(
    positives: pd.DataFrame,
    genes: pd.DataFrame,
    k: int,
    seed: int,
    gene_column: str = "gene_id",
    disease_column: str = "disease_id",
) -> pd.DataFrame:
    """The unmatched comparator: k controls drawn at random, ignoring confounders.

    Unmatched means the controls were not selected on the confounders, not that
    every gene in the genome is a control for every disease. Drawing the same
    number of controls by a different rule keeps the two arms the same size and
    makes the contrast between them a contrast in control selection alone,
    which is what bounds the answer from opposite directions.
    """
    rng = np.random.default_rng(seed)
    pool = genes["ensembl_gene_id"].dropna().to_numpy()

    rows = []
    for disease, group in positives.groupby(disease_column):
        positive_genes = sorted(set(group[gene_column].dropna()) & set(pool))
        if not positive_genes:
            continue
        taken = set(positive_genes)
        for gene in positive_genes:
            rows.append({disease_column: disease, "gene_id": gene, "y": 1})
        available = np.array([g for g in pool if g not in taken], dtype=object)
        wanted = min(k * len(positive_genes), len(available))
        for control in rng.choice(available, size=wanted, replace=False):
            rows.append({disease_column: disease, "gene_id": control, "y": 0})

    return pd.DataFrame(rows)


def standardised_mean_differences(
    matched: pd.DataFrame, genes: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Standardised mean difference per covariate, positives against controls."""
    joined = matched.merge(
        genes.rename(columns={"ensembl_gene_id": "gene_id"}), on="gene_id", how="left"
    )
    rows = []
    for column in columns:
        if column not in joined.columns:
            continue
        treated = joined.loc[joined["y"] == 1, column].astype(float)
        control = joined.loc[joined["y"] == 0, column].astype(float)
        pooled = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
        smd = (treated.mean() - control.mean()) / pooled if pooled > 0 else np.nan
        rows.append(
            {
                "covariate": column,
                "mean_positive": treated.mean(),
                "mean_control": control.mean(),
                "smd": smd,
            }
        )
    return pd.DataFrame(rows)
