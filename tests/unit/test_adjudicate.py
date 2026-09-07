"""Tests for the crosswalk adjudication.

The adjudication decides whether a queued mapping enters the complex arm, so its
failures are asymmetric. Accepting a wrong mapping attaches one disease's genes
to another disease's onset, silently. Leaving a right one queued costs only
sample size. The tests are written around that asymmetry: what must be rejected,
what must not be accepted on thin evidence, and what must stay queued.
"""

from __future__ import annotations

import pandas as pd

from src import ontology
from src.crosswalk import adjudicate


def _ontology() -> ontology.Ontology:
    """A small hierarchy, two branches deep on each side.

    MONDO:0000001 disease
      MONDO:0000100 infection
        MONDO:0000110 viral infection
          MONDO:0000111 rabies
            MONDO:0000112 furious rabies
              MONDO:0000113 a very specific rabies
      MONDO:0000200 neoplasm
        MONDO:0000210 carcinoma
    """
    parents = {
        "MONDO:0000001": (),
        "MONDO:0000100": ("MONDO:0000001",),
        "MONDO:0000110": ("MONDO:0000100",),
        "MONDO:0000111": ("MONDO:0000110",),
        "MONDO:0000112": ("MONDO:0000111",),
        "MONDO:0000113": ("MONDO:0000112",),
        "MONDO:0000200": ("MONDO:0000001",),
        "MONDO:0000210": ("MONDO:0000200",),
    }
    names = {
        "MONDO:0000001": "disease",
        "MONDO:0000100": "infection",
        "MONDO:0000110": "viral infection",
        "MONDO:0000111": "rabies",
        "MONDO:0000112": "furious rabies",
        "MONDO:0000113": "a very specific rabies",
        "MONDO:0000200": "neoplasm",
        "MONDO:0000210": "carcinoma",
    }
    children: dict[str, list[str]] = {}
    for term, above in parents.items():
        for parent in above:
            children.setdefault(parent, []).append(term)
    return ontology.Ontology(
        name=names,
        parents=parents,
        children={key: tuple(value) for key, value in children.items()},
        prefix="MONDO",
        synonyms={"MONDO:0000111": ("hydrophobia",)},
        xrefs={"MONDO:0000111": ("DOID:11260",)},
    )


def _row(longname: str, disease_id: str, doid: str | None = None) -> pd.Series:
    return pd.Series({"longname": longname, "disease_id": disease_id, "doid": doid})


def test_a_non_disease_target_is_rejected():
    mondo = _ontology()
    for target in ("GO_0036273", "OBA_2040163"):
        verdict = adjudicate.adjudicate_row(_row("Dermatophytosis", target), mondo, {}, {})
        assert verdict["verdict"] == adjudicate.REJECTED
        assert "not a disease" in verdict["reason"]


def test_a_condition_that_does_not_resolve_stays_queued():
    mondo = _ontology()
    verdict = adjudicate.adjudicate_row(
        _row("Other fluke infections", "MONDO_0000111"), mondo, {}, {}
    )
    assert verdict["verdict"] == adjudicate.QUEUED


def test_a_condition_resolves_through_its_disease_ontology_identifier():
    mondo = _ontology()
    verdict = adjudicate.adjudicate_row(
        _row("Rabies", "MONDO_0000110", doid="DOID:11260"), mondo, {}, mondo.by_xref("DOID")
    )
    assert verdict["verdict"] == adjudicate.ACCEPTED
    assert verdict["condition_term"] == "MONDO:0000111"


def test_a_condition_resolves_through_an_exact_synonym():
    mondo = _ontology()
    verdict = adjudicate.adjudicate_row(
        _row("Hydrophobia", "MONDO_0000111"), mondo, mondo.by_any_name(), {}
    )
    assert verdict["verdict"] == adjudicate.ACCEPTED


def test_a_target_one_step_above_the_condition_is_accepted():
    mondo = _ontology()
    verdict = adjudicate.adjudicate_row(
        _row("Rabies", "MONDO_0000110"), mondo, mondo.by_any_name(), {}
    )
    assert verdict["verdict"] == adjudicate.ACCEPTED
    assert "1 step" in verdict["reason"]


def test_a_target_far_above_the_condition_is_not_accepted():
    """Mapping a condition onto a broad category would attach that category's
    genes to this condition's onset."""
    mondo = _ontology()
    verdict = adjudicate.adjudicate_row(
        _row("A very specific rabies", "MONDO_0000100"), mondo, mondo.by_any_name(), {}
    )
    assert verdict["verdict"] == adjudicate.QUEUED
    assert f"more than {adjudicate.MAX_STEPS} steps" in verdict["reason"]


def test_a_target_in_a_disjoint_category_is_rejected():
    mondo = _ontology()
    verdict = adjudicate.adjudicate_row(
        _row("Rabies", "MONDO_0000210"), mondo, mondo.by_any_name(), {}
    )
    assert verdict["verdict"] == adjudicate.REJECTED
    assert "disjoint" in verdict["reason"]


def test_names_are_normalised_from_the_most_specific_form_outward():
    forms = adjudicate.normalise("Other Human immunodeficiency virus [HIV] disease, unspecified")
    assert forms[0] == "other human immunodeficiency virus [hiv] disease, unspecified"
    assert "human immunodeficiency virus disease" in forms[-1]


def test_normalisation_never_returns_an_empty_form():
    for name in ("", None, "Other", "   "):
        assert all(form for form in adjudicate.normalise(name))


def test_the_step_count_is_the_shortest_path():
    mondo = _ontology()
    assert adjudicate.steps_between(mondo, "MONDO:0000111", "MONDO:0000110", 3) == 1
    assert adjudicate.steps_between(mondo, "MONDO:0000111", "MONDO:0000100", 3) == 2
    assert adjudicate.steps_between(mondo, "MONDO:0000111", "MONDO:0000100", 1) is None
    assert adjudicate.steps_between(mondo, "MONDO:0000111", "MONDO:0000210", 5) is None


def test_only_accepted_mappings_become_confident(tmp_path, monkeypatch):
    crosswalk = pd.DataFrame(
        {
            "endpoint": ["E1", "E2", "E3", "E4"],
            "tier": ["review", "review", "review", "confident"],
            "confident": [False, False, False, True],
            "usable_for_onset": [True, True, True, True],
        }
    )
    directory = tmp_path / "crosswalk"
    directory.mkdir(parents=True)
    crosswalk.to_parquet(directory / "finngen_to_platform.parquet", index=False)
    monkeypatch.setattr(adjudicate.config, "interim_dir", lambda: tmp_path)

    queued = pd.DataFrame(
        {
            "endpoint": ["E1", "E2", "E3"],
            "verdict": [adjudicate.ACCEPTED, adjudicate.REJECTED, adjudicate.QUEUED],
        }
    )
    updated = adjudicate.apply_to_crosswalk(queued).set_index("endpoint")

    assert updated.loc["E1", "tier"] == "confident"
    assert updated.loc["E2", "tier"] == "rejected"
    assert updated.loc["E3", "tier"] == "review", "a queued mapping is never promoted"
    assert updated.loc["E4", "tier"] == "confident"
    assert bool(updated.loc["E1", "confident"])
    assert not bool(updated.loc["E3", "confident"])


def test_the_crosswalk_records_which_rows_were_adjudicated(tmp_path, monkeypatch):
    """A confident mapping that arrived by adjudication must stay tellable from
    one that passed the name check."""
    crosswalk = pd.DataFrame(
        {"endpoint": ["E1", "E2"], "tier": ["review", "confident"], "confident": [False, True]}
    )
    directory = tmp_path / "crosswalk"
    directory.mkdir(parents=True)
    crosswalk.to_parquet(directory / "finngen_to_platform.parquet", index=False)
    monkeypatch.setattr(adjudicate.config, "interim_dir", lambda: tmp_path)

    queued = pd.DataFrame({"endpoint": ["E1"], "verdict": [adjudicate.ACCEPTED]})
    updated = adjudicate.apply_to_crosswalk(queued).set_index("endpoint")
    assert updated.loc["E1", "adjudicated"] == adjudicate.ACCEPTED
    assert pd.isna(updated.loc["E2", "adjudicated"])
