"""Tests for the analysis-set record of the autosomal restriction."""

from __future__ import annotations

import pandas as pd
import pytest

from src.diagnostics import analysis_set


@pytest.fixture
def positives_and_genes(monkeypatch):
    """Four onset levels, with the sex-chromosome rows concentrated early."""
    rows = []
    for index in range(40):
        level = ["perinatal", "infancy", "childhood", "adult"][index % 4]
        rows.append(
            {
                "disease_id": f"Orphanet_{index}",
                "gene_id": f"ENSG{index:05d}",
                "evidence_source": "a_mendelian",
                "onset_bin": level,
            }
        )
    positives = pd.DataFrame(rows)

    # Eight genes sit on the X, and every one of them carries a perinatal or
    # infancy row, so the exclusion is differential by construction.
    early = positives["onset_bin"].isin(["perinatal", "infancy"]) & (positives.index < 16)
    genes = pd.DataFrame(
        {
            "ensembl_gene_id": positives["gene_id"],
            "chrom": ["chrX" if flag else "chr1" for flag in early],
        }
    )
    features = pd.DataFrame(
        {
            "ensembl_gene_id": positives["gene_id"],
            "phylop_241way_fraction_above": 0.5,
            "s_het_log": -2.0,
        }
    )

    monkeypatch.setattr("src.features.gene_table.load", lambda: genes)
    monkeypatch.setattr("src.features.evolutionary.load", lambda: features)
    return positives


def test_the_exclusion_record_counts_rows_genes_and_share(positives_and_genes):
    record = analysis_set.autosomal_exclusion_record(positives_and_genes)
    assert record["n_rows"] == 40
    assert record["n_rows_excluded"] == 8
    assert record["n_genes_excluded"] == 8
    assert record["excluded_share"] == pytest.approx(0.2)


def test_the_exclusion_record_reports_the_onset_position_of_each_side(positives_and_genes):
    record = analysis_set.autosomal_exclusion_record(positives_and_genes)
    # The excluded rows are the earliest ones here, so their mean position must
    # sit below the retained mean and the test must see the difference.
    assert record["mean_onset_position_excluded"] < record["mean_onset_position_retained"]
    assert record["mann_whitney_p"] < 0.05


def test_the_exclusion_record_omits_the_test_when_one_side_is_empty(
    positives_and_genes, monkeypatch
):
    genes = pd.DataFrame(
        {
            "ensembl_gene_id": positives_and_genes["gene_id"],
            "chrom": "chr1",
        }
    )
    monkeypatch.setattr("src.features.gene_table.load", lambda: genes)
    record = analysis_set.autosomal_exclusion_record(positives_and_genes)
    assert record["n_rows_excluded"] == 0
    assert "mann_whitney_p" not in record
