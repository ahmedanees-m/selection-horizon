"""Tests for inheritance-mode classification.

Mode of inheritance is a confounder of the H1 contrast: s_het is a coefficient
on heterozygotes, so it is close to blind to recessive disease genes, and the
dominant share of Orphanet diseases rises with onset. Misclassifying a disease
here moves the primary stratum, so the parsing is tested rather than assumed.
"""

from __future__ import annotations

import pandas as pd

from src.onset import inheritance


def modes(values: list[str | None]) -> list[str]:
    return list(inheritance.classify(pd.Series(values)))


def test_single_modes_classify_directly():
    assert modes(["Autosomal dominant"]) == [inheritance.DOMINANT]
    assert modes(["Autosomal recessive"]) == [inheritance.RECESSIVE]


def test_x_linked_recessive_counts_as_recessive():
    assert modes(["X-linked recessive"]) == [inheritance.RECESSIVE]


def test_a_disease_reported_both_ways_is_mixed_not_forced():
    assert modes(["Autosomal dominant|Autosomal recessive"]) == [inheritance.MIXED]
    assert modes(["Autosomal dominant|Autosomal recessive|X-linked recessive"]) == [
        inheritance.MIXED
    ]


def test_modes_outside_the_dominant_recessive_axis_are_not_guessed():
    assert modes(["Mitochondrial inheritance"]) == [inheritance.OTHER]
    assert modes(["Not applicable"]) == [inheritance.OTHER]
    assert modes([None]) == [inheritance.OTHER]


def test_dominant_plus_not_applicable_stays_dominant():
    assert modes(["Autosomal dominant|Not applicable"]) == [inheritance.DOMINANT]


def test_composition_reports_the_dominant_share_per_bin():
    diseases = pd.DataFrame(
        {
            "onset_bin": ["perinatal"] * 4 + ["adult"] * 4,
            "inheritance_mode": [
                inheritance.DOMINANT,
                inheritance.RECESSIVE,
                inheritance.RECESSIVE,
                inheritance.RECESSIVE,
                inheritance.DOMINANT,
                inheritance.DOMINANT,
                inheritance.DOMINANT,
                inheritance.RECESSIVE,
            ],
        }
    )
    table = inheritance.by_onset(diseases, ["perinatal", "adult"]).set_index("onset_bin")
    assert table.loc["perinatal", "dominant_share"] == 0.25
    assert table.loc["adult", "dominant_share"] == 0.75
    assert table.loc["perinatal", "total"] == 4


def test_composition_leaves_the_share_undefined_when_no_mode_is_resolved():
    diseases = pd.DataFrame(
        {"onset_bin": ["adult", "adult"], "inheritance_mode": [inheritance.OTHER] * 2}
    )
    table = inheritance.by_onset(diseases, ["adult"])
    assert pd.isna(table.loc[0, "dominant_share"])
