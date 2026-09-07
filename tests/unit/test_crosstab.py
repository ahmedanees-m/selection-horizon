"""Tests for the identifiability crosstab and its summary."""

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


def test_a_source_confined_to_one_level_does_not_span_the_range():
    cells = pd.concat(
        [
            _span("a_mendelian", ["perinatal"], 400, 900),
            _span("b_gwas", ["adult"], 400, 900),
        ],
        ignore_index=True,
    )
    summary = crosstab.summarise(cells).per_source_bins.set_index("evidence_source")
    assert summary.loc["a_mendelian", "occupied_bins"] == 1
    assert not summary.loc["a_mendelian", "spans_range"]
    assert not summary.loc["b_gwas", "spans_range"]


def test_a_source_occupying_three_levels_spans_the_range():
    cells = pd.concat(
        [
            _span("a_mendelian", ["perinatal", "childhood", "adult"], 400, 900),
            _span("b_gwas", ["adult"], 400, 900),
        ],
        ignore_index=True,
    )
    summary = crosstab.summarise(cells)
    bins = summary.per_source_bins.set_index("evidence_source")
    assert bins.loc["a_mendelian", "spans_range"]
    assert not bins.loc["b_gwas", "spans_range"]
    assert summary.adult_mendelian_diseases == 400


def test_sources_with_no_occupied_level_are_listed():
    cells = _span("a_mendelian", ["perinatal", "childhood", "adult"], 400, 900)
    bins = crosstab.summarise(cells).per_source_bins
    assert set(bins["evidence_source"]) == set(crosstab.SOURCE_LABELS)
    assert int(bins.set_index("evidence_source").loc["d_gene_burden", "occupied_bins"]) == 0


def test_a_level_below_the_minimum_sizes_is_not_occupied():
    cells = pd.concat(
        [
            _span("a_mendelian", ["perinatal", "childhood", "adolescent"], 400, 900),
            _span("a_mendelian", ["adult"], 20, 40),
        ],
        ignore_index=True,
    )
    summary = crosstab.summarise(cells)
    bins = summary.per_source_bins.set_index("evidence_source")
    assert bins.loc["a_mendelian", "occupied_bins"] == 3
    assert summary.adult_mendelian_diseases == 20
