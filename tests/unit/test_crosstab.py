"""Tests for the identifiability crosstab and the minimum sizes."""

from __future__ import annotations

import pandas as pd

from src.diagnostics import crosstab


def positives(rows: list[tuple[str, str, str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["disease_id", "gene_id", "evidence_source", "onset_bin"])


def test_single_gene_diseases_reach_the_crosstab():
    """No minimum-gene filter applies. Most Mendelian diseases have one gene."""
    frame = positives(
        [
            ("D1", "G1", "a_mendelian", "adult"),
            ("D2", "G2", "a_mendelian", "adult"),
        ]
    )
    cells = crosstab.build(frame)
    row = cells[cells["onset_bin"] == "adult"].iloc[0]
    assert row["n_diseases"] == 2
    assert row["n_pairs"] == 2
    assert row["median_genes_per_disease"] == 1.0


def test_duplicate_pairs_are_counted_once():
    frame = positives(
        [
            ("D1", "G1", "a_mendelian", "adult"),
            ("D1", "G1", "a_mendelian", "adult"),
        ]
    )
    assert crosstab.build(frame).iloc[0]["n_pairs"] == 1


def test_missing_onset_is_visible_rather_than_dropped():
    frame = positives([("D1", "G1", "b_gwas", None)])
    cells = crosstab.build(frame)
    assert set(cells["onset_bin"]) == {"unassigned"}


def test_bins_are_returned_in_collapsed_onset_order():
    frame = positives(
        [
            ("D1", "G1", "a_mendelian", "adult"),
            ("D2", "G2", "a_mendelian", "perinatal"),
            ("D3", "G3", "a_mendelian", "childhood"),
        ]
    )
    assert list(crosstab.build(frame)["onset_bin"]) == ["perinatal", "childhood", "adult"]


def test_missing_column_raises():
    frame = pd.DataFrame({"disease_id": ["D1"], "gene_id": ["G1"]})
    try:
        crosstab.build(frame)
    except ValueError as error:
        assert "evidence_source" in str(error)
    else:
        raise AssertionError("expected a ValueError")


def _span(source: str, bins: list[str], diseases: int, pairs: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "evidence_source": source,
            "onset_bin": bins,
            "n_diseases": diseases,
            "n_pairs": pairs,
            "median_genes_per_disease": 1.0,
        }
    )


def test_gate_fails_when_neither_primary_source_spans_the_range():
    cells = pd.concat(
        [
            _span("a_mendelian", ["neonatal"], 400, 900),
            _span("b_gwas", ["adult"], 400, 900),
        ],
        ignore_index=True,
    )
    result = crosstab.assess(cells)
    assert result.verdict == "fail"


def test_gate_is_marginal_when_one_source_spans_the_range():
    cells = pd.concat(
        [
            _span("a_mendelian", ["neonatal", "childhood", "adult"], 400, 900),
            _span("b_gwas", ["adult"], 400, 900),
        ],
        ignore_index=True,
    )
    result = crosstab.assess(cells)
    assert result.verdict == "marginal"
    assert result.pivotal_cell_diseases == 400


def test_gate_reports_the_pivotal_cell_even_when_both_sources_span():
    cells = pd.concat(
        [
            _span("a_mendelian", ["neonatal", "childhood", "adult"], 400, 900),
            _span("b_gwas", ["neonatal", "childhood", "adult"], 400, 900),
        ],
        ignore_index=True,
    )
    result = crosstab.assess(cells)
    assert result.verdict == "pass"
    assert result.pivotal_cell_diseases == 400
    assert any("pivotal" in reason for reason in result.reasons)


def test_thin_pivotal_cell_downgrades_a_passing_gate():
    cells = pd.concat(
        [
            _span("a_mendelian", ["neonatal", "childhood", "adolescent"], 400, 900),
            _span("a_mendelian", ["adult"], 20, 40),
            _span("b_gwas", ["neonatal", "childhood", "adult"], 400, 900),
        ],
        ignore_index=True,
    )
    result = crosstab.assess(cells)
    assert result.verdict == "marginal"
    assert result.pivotal_cell_diseases == 20
