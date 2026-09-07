"""Tests for the per-case severity term.

The label check here is not ceremony. A wrong HPO identifier in the interval
table places deaths in the wrong decade, changes kappa_d, and produces a
fitness-cost index that is wrong in a direction nothing downstream would catch.
HP:0005268 is Miscarriage, and an early draft of the table had it as a
young-adult death term.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.onset import clinical_course

OBO = """format-version: 1.2

[Term]
id: HP:0003811
name: Neonatal death

[Term]
id: HP:0001522
name: Death in infancy

[Term]
id: HP:0003819
name: Death in childhood

[Term]
id: HP:0011421
name: Death in adolescence

[Term]
id: HP:0100613
name: Death in early adulthood

[Term]
id: HP:0033763
name: Death in adulthood

[Term]
id: HP:0033764
name: Death in middle age

[Term]
id: HP:0033765
name: Death in late adulthood

[Term]
id: HP:0003826
name: Stillbirth

[Term]
id: HP:0005268
name: Miscarriage

[Term]
id: HP:0034241
name: Prenatal death

[Term]
id: HP:0001699
name: Sudden death

[Term]
id: HP:0001645
name: Sudden cardiac death

[Term]
id: HP:0033258
name: Sudden unexpected death in epilepsy
"""


@pytest.fixture
def obo(tmp_path):
    path = tmp_path / "hp.obo"
    path.write_text(OBO, encoding="utf-8")
    return path


def annotations(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["database_id", "hpo_id"])


def test_labels_match_the_ontology(obo):
    clinical_course.verify_labels(obo)


def test_a_drifted_label_raises(obo, tmp_path):
    path = tmp_path / "drifted.obo"
    path.write_text(OBO.replace("name: Miscarriage", "name: Something else"), encoding="utf-8")
    with pytest.raises(ValueError, match="HP:0005268"):
        clinical_course.verify_labels(path)


def test_intervals_are_ordered_and_do_not_overlap_within_the_ladder():
    ladder = [
        "HP:0003811",
        "HP:0001522",
        "HP:0003819",
        "HP:0011421",
        "HP:0100613",
    ]
    bounds = [clinical_course.DEATH_INTERVALS[term][1:] for term in ladder]
    for (_, upper), (lower, _) in zip(bounds[:-1], bounds[1:], strict=True):
        assert upper == lower


def test_earliest_interval_sets_the_point_estimate():
    frame = annotations([("ORPHA:1", "HP:0001522"), ("ORPHA:1", "HP:0033763")])
    table = clinical_course.per_disease(frame)
    row = table.iloc[0]
    assert row["death_lower"] == pytest.approx(0.0767)
    assert row["death_upper"] == pytest.approx(65.0)
    assert row["death_midpoint"] < 2.0
    assert row["n_terms"] == 2


def test_prenatal_loss_carries_no_interval_but_is_flagged():
    frame = annotations([("OMIM:1", "HP:0003826")])
    table = clinical_course.per_disease(frame)
    assert bool(table.iloc[0]["prenatal_loss"])
    assert pd.isna(table.iloc[0]["death_midpoint"])


def test_orpha_codes_are_extracted_and_omim_ids_are_not():
    frame = annotations([("ORPHA:558", "HP:0001522"), ("OMIM:123456", "HP:0001522")])
    table = clinical_course.per_disease(frame).set_index("database_id")
    assert table.loc["ORPHA:558", "orpha_code"] == 558
    assert pd.isna(table.loc["OMIM:123456", "orpha_code"])


def test_coverage_separates_located_from_unlocated():
    frame = annotations(
        [
            ("ORPHA:1", "HP:0001522"),
            ("OMIM:2", "HP:0003826"),
            ("OMIM:3", "HP:0001699"),
            ("OMIM:4", "HP:0000001"),
        ]
    )
    counts = clinical_course.coverage(frame)
    assert counts["diseases_in_hpoa"] == 4
    assert counts["with_age_of_death_interval"] == 1
    assert counts["with_prenatal_loss_only"] == 1
    assert counts["with_unlocated_severity_only"] == 1
    assert counts["orpha_keyed_with_interval"] == 1
