"""Tests for the matched inheritance-stratum comparison.

The matched figures answer a different question from the pooled ones, and the
answer depends entirely on where the controls come from. What has to hold is
that a control shares its gene's confounder stratum, that no disease gene is
used as a control, and that a stratum too thin to supply k is reported rather
than filled from elsewhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import discrimination

METRICS = {"s_het": "s_het_log", "loeuf": "loeuf", "missense_z": "missense_z"}


def genes(n: int = 120, seed: int = 3) -> pd.DataFrame:
    """A scored gene table with two confounder strata of equal size."""
    rng = np.random.default_rng(seed)
    decile = np.where(np.arange(n) % 2 == 0, 1, 10)
    return pd.DataFrame(
        {
            "ensembl_gene_id": [f"ENSG{i:011d}" for i in range(n)],
            "s_het_log": rng.normal(size=n),
            "loeuf": rng.normal(size=n),
            "missense_z": rng.normal(size=n),
            "cds_length_decile": decile,
            "expected_lof_decile": decile,
            "gc_content": np.where(decile == 1, 0.3, 0.7),
            "cds_length": np.where(decile == 1, 500.0, 5000.0),
            "expected_lof": np.where(decile == 1, 10.0, 100.0),
        }
    )


def test_a_control_shares_the_stratum_of_the_gene_it_matches():
    pool = discrimination.scored_pool(genes(), METRICS)
    disease = {pool["ensembl_gene_id"][i] for i in range(0, 40, 2)}
    blocks, _ = discrimination.matched_blocks(disease, pool, disease, k=3, seed=1)
    deciles = pool["cds_length_decile"].to_numpy()
    for position, controls in blocks:
        assert all(deciles[control] == deciles[position] for control in controls)


def test_a_disease_gene_is_never_drawn_as_a_control():
    pool = discrimination.scored_pool(genes(), METRICS)
    disease = {pool["ensembl_gene_id"][i] for i in range(0, 60)}
    blocks, _ = discrimination.matched_blocks(disease, pool, disease, k=3, seed=1)
    drawn = {control for _, controls in blocks for control in controls}
    assert not drawn & {i for i in range(60)}


def test_a_stratum_too_thin_to_fill_k_is_reported():
    pool = discrimination.scored_pool(genes(n=24), METRICS)
    disease = set(pool["ensembl_gene_id"])
    blocks, shortfalls = discrimination.matched_blocks(disease, pool, disease, k=5, seed=1)
    assert shortfalls == len(blocks)
    assert all(not controls for _, controls in blocks)


def test_a_separating_score_reaches_one_against_matched_controls():
    table = genes(n=200)
    positives = {gene for gene in table["ensembl_gene_id"][:40]}
    # s_het is made to separate the positives from everything else.
    table["s_het_log"] = np.where(table["ensembl_gene_id"].isin(positives), 10.0, 0.0)
    frame = pd.DataFrame(
        {"gene_id": sorted(positives), "inheritance_mode": "recessive", "evidence_source": "a"}
    )
    rows, _, balance = discrimination.by_inheritance_matched(frame, table, METRICS, seed=1, k=3)
    measured = rows.set_index("metric").loc["s_het"]
    assert measured["auroc"] == 1.0
    assert measured["n_disease_genes"] == len(positives)
    assert set(balance["covariate"]) == set(discrimination.MATCHING_COVARIATES)
