"""Tests for the disease-name matcher shared by both onset cross-checks.

Every test here is a pairing the matcher actually made and should not have. The
matcher places free-text condition names against curated registry endpoints, and
a wrong pairing does not fail loudly: it imports the wrong endpoint's age
distribution and quietly degrades the concordance it was built to measure.
"""

from __future__ import annotations

from src.onset import donertas


def test_a_kind_mismatch_is_vetoed():
    """Token overlap alone pairs a stomach disorder with a stomach cancer."""
    assert donertas.similarity("stomach disorder", "malignant neoplasm of stomach") == 0.0


def test_an_endpoint_broader_than_the_condition_is_vetoed():
    """These three were the largest disagreements in the T2 to T3 concordance."""
    assert donertas.similarity("Postcoital Bleeding", "Bleeding") == 0.0
    assert donertas.similarity("Type 1 Diabetes Mellitus", "Diabetes mellitus") == 0.0
    assert donertas.similarity("Motor Neurone Disease", "Motor disorders") == 0.0


def test_an_endpoint_carrying_extra_qualifiers_is_not_vetoed():
    """The breadth veto is one directional. Registry endpoints routinely carry
    qualifying words the condition does not, and vetoing that direction would
    reject the matches that work."""
    assert donertas.similarity("Asthma", "Asthma, more controls excluded") > 0.0


def test_an_exact_name_matches_itself():
    assert donertas.similarity("Hydrocele", "Hydrocele") == 1.0


def test_administrative_qualifiers_do_not_carry_a_match():
    """Two unrelated conditions both marked not otherwise specified were paired
    on those three words alone."""
    assert (
        donertas.similarity(
            "Stroke - not otherwise specified", "Mental disorders, not otherwise specified"
        )
        == 0.0
    )


def test_a_stated_subtype_disagreement_is_vetoed():
    assert (
        donertas.similarity("Type 2 Diabetes Mellitus", "Type 1 diabetes, strict definition") == 0.0
    )
    assert donertas.similarity("Type 1 Diabetes Mellitus", "Type 2 diabetes") == 0.0


def test_a_matching_subtype_is_not_vetoed():
    assert donertas.similarity("Type 2 diabetes", "Type 2 diabetes, wide") > 0.0


def test_roman_and_arabic_subtypes_are_the_same_subtype():
    assert donertas.subtypes("Type II diabetes") == donertas.subtypes("Type 2 diabetes")
    assert donertas.subtypes("Type IV renal tubular acidosis") == {"4"}


def test_a_name_with_no_subtype_yields_none():
    assert donertas.subtypes("Dermatitis") == set()
    assert donertas.subtypes("Asthma") == set()


def test_a_subtype_on_one_side_only_does_not_veto():
    """Only a stated disagreement vetoes. One side being silent is not a
    disagreement, and the breadth veto already handles the case where the silent
    side is also the broader one."""
    score = donertas.similarity("Diabetes mellitus type 2", "Diabetes insipidus")
    assert donertas.subtypes("Diabetes insipidus") == set()
    assert score > 0.0, "no veto fires when only one side states a subtype"
    assert score < donertas.MATCH_THRESHOLD, "and the shared word alone is not a match"


def test_the_tokeniser_drops_words_that_carry_no_disease_information():
    tokens = donertas._tokens("Other unspecified disorders of the ear")
    assert "other" not in tokens
    assert "unspecified" not in tokens
    assert "disorders" not in tokens
    assert "ear" in tokens
