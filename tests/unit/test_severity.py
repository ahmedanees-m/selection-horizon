"""Tests for the severity index and the ontology resolver.

The resolver exists because a wrong ontology identifier changes a result by a
plausible amount and fails no downstream assertion. These tests check that
concepts resolve by name, that an unknown name raises rather than matching
nothing, and that the rarefaction actually removes the depth effect it was added
for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.onset import hpo_ontology, severity

OBO = """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0000707
name: Abnormality of the nervous system
is_a: HP:0000118

[Term]
id: HP:0001626
name: Abnormality of the cardiovascular system
is_a: HP:0000118

[Term]
id: HP:0001249
name: Intellectual disability
is_a: HP:0000707

[Term]
id: HP:0000789
name: Infertility

[Term]
id: HP:0003251
name: Male infertility
is_a: HP:0000789

[Term]
id: HP:0009999
name: Retired term
is_obsolete: true
"""


@pytest.fixture
def obo(tmp_path):
    path = tmp_path / "hp.obo"
    path.write_text(OBO, encoding="utf-8")
    hpo_ontology.load.cache_clear()
    yield str(path)
    hpo_ontology.load.cache_clear()


def test_concepts_resolve_by_name(obo):
    ontology = hpo_ontology.load(obo)
    assert ontology.resolve("Infertility") == "HP:0000789"
    assert ontology.resolve("infertility") == "HP:0000789"


def test_an_unknown_name_raises_rather_than_matching_nothing(obo):
    ontology = hpo_ontology.load(obo)
    with pytest.raises(KeyError, match="Oligospermia"):
        ontology.resolve("Oligospermia")


def test_obsolete_terms_are_not_resolvable(obo):
    ontology = hpo_ontology.load(obo)
    assert "HP:0009999" not in ontology.name


def test_subtree_includes_descendants_and_self(obo):
    ontology = hpo_ontology.load(obo)
    term, subtree = ontology.subtree_by_name("Infertility")
    assert term == "HP:0000789"
    assert subtree == {"HP:0000789", "HP:0003251"}


def test_organ_systems_are_the_children_of_phenotypic_abnormality(obo):
    ontology = hpo_ontology.load(obo)
    root = ontology.resolve("Phenotypic abnormality")
    assert set(ontology.children[root]) == {"HP:0000707", "HP:0001626"}


def test_provenance_records_what_each_concept_resolved_to(obo):
    rows = hpo_ontology.provenance(["Infertility"], obo)
    assert rows[0]["hpo_id"] == "HP:0000789"
    assert rows[0]["ontology_label"] == "Infertility"
    assert rows[0]["n_terms_in_subtree"] == 2


def test_rarefaction_returns_observed_richness_below_the_sample_depth():
    counts = np.array([2.0, 1.0, 0.0])
    assert severity.rarefied_richness(counts, sample=10) == pytest.approx(2.0)


def test_rarefaction_is_bounded_by_observed_richness():
    counts = np.array([50.0, 30.0, 20.0])
    rarefied = severity.rarefied_richness(counts, sample=10)
    assert 0 < rarefied <= 3.0


def test_rarefaction_removes_the_depth_effect_it_was_added_for():
    """Two diseases with the same composition but different curation depth.

    The raw count separates them; the rarefied estimate should not.
    """
    shallow = np.array([5.0, 5.0, 0.0, 0.0])
    deep = np.array([50.0, 50.0, 1.0, 1.0])

    assert (shallow > 0).sum() < (deep > 0).sum()

    rarefied_shallow = severity.rarefied_richness(shallow, sample=10)
    rarefied_deep = severity.rarefied_richness(deep, sample=10)
    assert abs(rarefied_shallow - rarefied_deep) < 0.5


def test_rarefaction_is_undefined_for_a_disease_with_no_annotations():
    assert np.isnan(severity.rarefied_richness(np.zeros(3), sample=10))


def test_unknown_frequency_label_raises():
    with pytest.raises(ValueError, match="frequency labels"):
        severity.frequency_weights(pd.Series(["Sometimes (about half)"]))


def test_missing_frequency_falls_back_to_the_modal_category():
    weights = severity.frequency_weights(pd.Series([None]))
    assert weights.iloc[0] == pytest.approx(0.545)


def test_frequency_weights_are_ordered_by_the_interval_they_name():
    labels = [
        "Obligate (100%)",
        "Very frequent (99-80%)",
        "Frequent (79-30%)",
        "Occasional (29-5%)",
        "Very rare (<4-1%)",
        "Excluded (0%)",
    ]
    values = severity.frequency_weights(pd.Series(labels)).to_numpy()
    assert np.all(np.diff(values) < 0)
