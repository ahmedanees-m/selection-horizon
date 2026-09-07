"""Tests for the T3 onset tier.

The appendix itself is not in the repository, so the tests that need it are the
ones the build already enforces at run time: both tables must parse to 308
conditions and every abbreviated term must resolve. What is tested here is
everything that can go wrong without the file, which is most of it: the cell
grammar, the text normalisation, the alias map, and the concordance arithmetic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.onset import kuan


def test_the_cell_pattern_accepts_the_three_released_forms():
    for cell, median in (
        ("63     (52, 72)", "63"),
        ("70.5     (59.75, 78.25)", "70.5"),
        ("NA     (NA, NA)", "NA"),
    ):
        match = kuan.CELL.match(cell)
        assert match is not None, cell
        assert match.group(1) == median


def test_the_cell_pattern_rejects_a_condition_name():
    assert kuan.CELL.match("Myocardial Infarction") is None
    assert kuan.CELL.match("Cardiovascular") is None


def test_the_unresolved_character_is_normalised_to_a_dash():
    assert kuan._clean("Benign Neo " + kuan.UNRESOLVED_CHARACTER + " Brain") == "Benign Neo - Brain"


def test_whitespace_is_collapsed_so_the_join_is_not_defeated_by_wrapping():
    assert kuan._clean("Myelodysplastic \n Syndrome") == "Myelodysplastic Syndrome"
    assert kuan._clean(None) == ""


def test_every_alias_points_somewhere_else():
    for source, target in kuan.TERM_ALIASES.items():
        assert source != target


def test_the_alias_map_is_one_to_one():
    targets = list(kuan.TERM_ALIASES.values())
    assert len(targets) == len(set(targets))


def test_the_concordance_reports_too_few_matches_rather_than_correlating_them():
    matched = pd.DataFrame({"kuan_median_age": [40.0, 50.0], "risteys_median_age": [42.0, 51.0]})
    statistics = kuan.concordance_statistics(matched)
    assert statistics["n_matched"] == 2
    assert "note" in statistics


def test_the_concordance_recovers_a_monotone_relationship():
    matched = pd.DataFrame(
        {
            "kuan_median_age": [10.0, 20.0, 30.0, 40.0, 55.0],
            "risteys_median_age": [12.0, 21.0, 33.0, 39.0, 60.0],
        }
    )
    statistics = kuan.concordance_statistics(matched)
    assert statistics["n_matched"] == 5
    assert statistics["spearman"] == pytest.approx(1.0)
    assert statistics["mean_absolute_difference_years"] < 6


def test_the_concordance_reports_a_signed_and_an_absolute_difference():
    """A registry that records everything two years later is not a registry that
    disagrees by two years at random, and the two statistics separate them."""
    offset = pd.DataFrame(
        {
            "kuan_median_age": [30.0, 40.0, 50.0, 60.0],
            "risteys_median_age": [32.0, 42.0, 52.0, 62.0],
        }
    )
    scattered = pd.DataFrame(
        {
            "kuan_median_age": [30.0, 40.0, 50.0, 60.0],
            "risteys_median_age": [32.0, 38.0, 52.0, 58.0],
        }
    )
    assert kuan.concordance_statistics(offset)["median_difference_years"] == pytest.approx(-2.0)
    assert kuan.concordance_statistics(scattered)["median_difference_years"] == pytest.approx(0.0)
    assert kuan.concordance_statistics(scattered)["mean_absolute_difference_years"] > 0
