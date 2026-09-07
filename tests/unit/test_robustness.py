"""Tests for the robustness suite, run against synthetic positives only.

The suite runs the primary estimand many times. Doing that on the real gene sets
before the pre-registration is filed would compound the early exposure the
registration discloses, so correctness is checked here against a generator with a
known effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models import robustness
from tests.fixtures import synthetic

SEED = 11


@pytest.fixture(scope="module")
def synthetic_inputs():
    positives, features = synthetic.generate(gamma=0.12, seed=SEED)
    features = features.assign(
        is_autosomal=True,
        is_hla=False,
        is_olfactory=False,
        loeuf=-features[synthetic.CONSTRAINT_COLUMN],
        phylop_100way_fraction_above=features[synthetic.CONSERVATION_COLUMN],
    )
    return positives, features


def test_every_arm_declares_a_kind_and_a_rationale():
    for arm in robustness.arms():
        assert arm.kind
        assert len(arm.rationale) > 20, f"{arm.name} has no real rationale"


def test_arm_names_are_unique():
    names = [arm.name for arm in robustness.arms()]
    assert len(names) == len(set(names))


def test_the_suite_carries_a_reference_arm():
    assert any(arm.name == "primary" for arm in robustness.arms())


def test_arms_that_cannot_run_are_declared_rather_than_omitted():
    assert robustness.NOT_RUN
    for reason in robustness.NOT_RUN.values():
        assert len(reason) > 20


def test_a_score_swap_changes_which_columns_are_used(synthetic_inputs):
    positives, features = synthetic_inputs
    arm = next(a for a in robustness.arms() if a.name == "conservation_100way")
    row = robustness.run_arm(positives, features, arm, SEED, default_k=10)
    assert row["conservation"] == "phylop_100way_fraction_above"
    assert np.isfinite(row["gamma"])


def test_a_missing_score_is_reported_rather_than_crashing(synthetic_inputs):
    positives, features = synthetic_inputs
    arm = robustness.Arm(
        "absent",
        "score swap",
        "a score that is not in the matrix at all",
        conservation="not_a_column",
    )
    row = robustness.run_arm(positives, features, arm, SEED, default_k=10)
    assert np.isnan(row["gamma"])
    assert "not in the feature matrix" in row["note"]


def test_changing_k_changes_the_design_size(synthetic_inputs):
    positives, features = synthetic_inputs
    small = robustness.run_arm(
        positives, features, robustness.Arm("k5", "design swap", "five controls", k=5), SEED, 10
    )
    large = robustness.run_arm(
        positives, features, robustness.Arm("k20", "design swap", "twenty controls", k=20), SEED, 10
    )
    assert small["n_rows"] < large["n_rows"]
    assert small["n_positives"] == large["n_positives"]


def test_the_summary_flags_a_sign_flip_as_material():
    rows = [
        {"arm": "primary", "kind": "reference", "gamma": 0.8, "ci_low": 0.4, "ci_high": 1.2},
        {"arm": "flipped", "kind": "score swap", "gamma": -0.7, "ci_low": -1.1, "ci_high": -0.3},
        {"arm": "steady", "kind": "score swap", "gamma": 0.75, "ci_low": 0.35, "ci_high": 1.15},
    ]
    table = robustness.summarise(rows).set_index("arm")
    assert bool(table.loc["flipped", "large_change"])
    assert not bool(table.loc["steady", "large_change"])
    assert not bool(table.loc["primary", "large_change"])


def test_the_summary_flags_a_large_move():
    rows = [
        {"arm": "primary", "kind": "reference", "gamma": 1.0, "ci_low": 0.5, "ci_high": 1.5},
        {"arm": "halved", "kind": "design swap", "gamma": 0.2, "ci_low": -0.1, "ci_high": 0.5},
    ]
    table = robustness.summarise(rows).set_index("arm")
    assert bool(table.loc["halved", "large_change"])


def test_the_suite_recovers_a_consistent_sign_across_score_swaps(synthetic_inputs):
    """Swapping which column stands for a family should not flip the estimate."""
    positives, features = synthetic_inputs
    table = robustness.run("synthetic", positives=positives, features=features)
    usable = table[table["gamma"].notna() & table["kind"].isin(["reference", "score swap"])]
    assert len(usable) >= 3
    assert usable["gamma"].gt(0).all()


def test_the_reference_arm_is_never_flagged(synthetic_inputs):
    positives, features = synthetic_inputs
    table = robustness.run("synthetic", positives=positives, features=features).set_index("arm")
    assert not bool(table.loc["primary", "large_change"])


def test_a_restriction_arm_leaves_the_design_no_larger(synthetic_inputs):
    positives, features = synthetic_inputs
    base = robustness.run_arm(
        positives, features, robustness.Arm("primary", "reference", "the reference arm"), SEED, 10
    )
    restricted = robustness.run_arm(
        positives,
        features.assign(is_hla=lambda f: f.index < 500),
        robustness.Arm(
            "no_hla", "restriction", "dropping a gene family", gene_filter="no_hla_no_olfactory"
        ),
        SEED,
        10,
    )
    assert restricted["n_rows"] <= base["n_rows"]


def test_summarise_handles_an_arm_that_failed(synthetic_inputs):
    rows = [
        {"arm": "primary", "kind": "reference", "gamma": 0.8, "ci_low": 0.4, "ci_high": 1.2},
        {
            "arm": "broken",
            "kind": "score swap",
            "gamma": float("nan"),
            "ci_low": np.nan,
            "ci_high": np.nan,
        },
    ]
    table = robustness.summarise(rows)
    assert len(table) == 2
    assert pd.isna(table.loc[table["arm"] == "broken", "gamma"].iloc[0])


def test_a_restriction_that_cannot_find_its_column_raises(synthetic_inputs):
    """Three arms once returned the reference estimate to six decimal places,
    which is what a filter that quietly matches nothing looks like."""
    positives, features = synthetic_inputs
    bare = features.drop(columns=["is_autosomal", "is_hla", "is_olfactory"], errors="ignore")
    arm = robustness.Arm("autosomal_only", "restriction", "a" * 40, gene_filter="autosomal")
    with pytest.raises(KeyError, match="is_autosomal"):
        robustness.apply_filters(positives, bare, arm)


def test_the_oncology_filter_raises_without_disease_names(synthetic_inputs):
    positives, features = synthetic_inputs
    arm = robustness.Arm("oncology", "restriction", "a" * 40, disease_filter="no_oncology")
    with pytest.raises(KeyError, match="disease_name"):
        robustness.apply_filters(
            positives.drop(columns=["disease_name"], errors="ignore"), features, arm
        )


def test_a_gene_family_restriction_removes_genes(synthetic_inputs):
    positives, features = synthetic_inputs
    marked = features.assign(
        is_hla=lambda f: f.index < 40,
        is_olfactory=lambda f: (f.index >= 40) & (f.index < 70),
    )
    arm = robustness.Arm(
        "no_hla_no_olfactory", "restriction", "a" * 40, gene_filter="no_hla_no_olfactory"
    )
    _, restricted = robustness.apply_filters(positives, marked, arm)
    assert len(restricted) == len(marked) - 70


def test_an_autosomal_restriction_removes_the_sex_chromosomes(synthetic_inputs):
    positives, features = synthetic_inputs
    marked = features.assign(is_autosomal=lambda f: f.index >= 25)
    arm = robustness.Arm("autosomal_only", "restriction", "a" * 40, gene_filter="autosomal")
    _, restricted = robustness.apply_filters(positives, marked, arm)
    assert len(restricted) == len(marked) - 25


def test_an_arm_whose_inputs_are_absent_is_recorded_rather_than_raising(
    synthetic_inputs, tmp_path, monkeypatch
):
    """Arms differ in what they read, and not every input is present everywhere.

    The all-ages arm needs the Orphanet onset table, which a checkout without
    ingested data does not have. The suite reports the arm with its reason
    instead of failing.
    """
    positives, features = synthetic_inputs
    monkeypatch.setattr(config, "interim_dir", lambda: tmp_path / "absent")
    row = robustness.run_arm(
        positives,
        features,
        robustness.Arm(
            "all_ages_included",
            "inclusion",
            "diseases annotated only as spanning every age",
            onset_encoding="span_value",
            include_all_ages=True,
        ),
        SEED,
        10,
    )
    assert not np.isfinite(row["gamma"])
    assert row["note"]
