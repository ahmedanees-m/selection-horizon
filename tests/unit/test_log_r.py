"""Tests for the second primary estimand.

The construction rests on one claim: a ratio of lifts is invariant to the
multiplicative attenuation that label noise imposes, and a difference of lifts is
not. That claim is what makes the ratio the estimand and the difference a
sensitivity, so it is tested directly rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import log_r

SEED = 5
ALPHA = 0.05


def test_a_separable_feature_is_scored_well_above_chance():
    rng = np.random.default_rng(SEED)
    y = np.repeat([0, 1], 200).astype(float)
    x = (y * 2.0 + rng.normal(size=400)).reshape(-1, 1)
    assert log_r.cross_validated_auroc(x, y, SEED) > 0.85


def test_a_feature_carrying_nothing_is_scored_near_chance():
    rng = np.random.default_rng(SEED)
    y = np.repeat([0, 1], 200).astype(float)
    x = rng.normal(size=(400, 1))
    assert log_r.cross_validated_auroc(x, y, SEED) == pytest.approx(0.5, abs=0.1)


def test_a_disease_with_too_few_positives_is_not_scored():
    y = np.array([1.0, 1.0] + [0.0] * 40)
    x = np.random.default_rng(SEED).normal(size=(42, 2))
    assert np.isnan(log_r.cross_validated_auroc(x, y, SEED))


def test_a_disease_with_one_class_is_not_scored():
    y = np.ones(40)
    x = np.random.default_rng(SEED).normal(size=(40, 2))
    assert np.isnan(log_r.cross_validated_auroc(x, y, SEED))


def test_cross_validation_within_matched_sets_falls_below_chance():
    """Why the estimand uses random controls rather than matched ones.

    When every positive has controls carrying the same covariate values, a model
    trained on the other folds scores them identically and the held-out control
    outranks its own positive often enough to push the pooled AUROC below
    chance. The quantity is then not a weak signal but an artefact of the
    design, and no restriction on it can be satisfied.
    """
    rng = np.random.default_rng(SEED)
    covariates = rng.normal(size=(60, 3))
    # Each positive is paired with three controls carrying its own covariates.
    x = np.vstack([covariates, np.repeat(covariates, 3, axis=0)])
    y = np.concatenate([np.ones(60), np.zeros(180)])
    assert log_r.cross_validated_auroc(x, y, SEED) < 0.5


def test_the_estimand_is_computed_against_random_controls():
    import inspect

    assert inspect.signature(log_r.estimate).parameters["matched"].default is False


def test_the_ratio_survives_multiplicative_attenuation():
    """Label noise attenuates both models, and the ratio has to divide it out.

    This is the reason the estimand is a ratio. If it were a difference, a
    disease whose labels are noisier would look like a disease where evolution
    adds less, and the estimand would be measuring annotation quality along the
    onset axis rather than biology.
    """
    lift_m0, lift_m1 = 0.20, 0.30
    for attenuation in (1.0, 0.6, 0.3):
        ratio = (lift_m1 * attenuation) / (lift_m0 * attenuation)
        assert ratio == pytest.approx(lift_m1 / lift_m0)

    differences = [(lift_m1 - lift_m0) * attenuation for attenuation in (1.0, 0.6, 0.3)]
    assert differences[0] != pytest.approx(differences[-1])


def test_the_slope_recovers_a_planted_gradient():
    onset = np.linspace(0.0, 1.0, 80)
    values = 0.4 - 0.6 * onset
    table = pd.DataFrame({"onset": onset, "log_ratio": values})
    fitted = log_r.slope(table, "log_ratio", ALPHA)
    assert fitted["slope"] == pytest.approx(-0.6, abs=1e-6)
    assert fitted["intercept"] == pytest.approx(0.4, abs=1e-6)


def test_a_flat_relationship_gives_an_interval_that_covers_zero():
    rng = np.random.default_rng(SEED)
    onset = rng.uniform(size=200)
    table = pd.DataFrame({"onset": onset, "log_ratio": rng.normal(size=200)})
    fitted = log_r.slope(table, "log_ratio", ALPHA)
    assert not fitted["excludes_zero"]


def test_a_slope_with_too_few_points_is_reported_rather_than_fitted():
    table = pd.DataFrame({"onset": [0.0, 1.0], "log_ratio": [1.0, 2.0]})
    assert "note" in log_r.slope(table, "log_ratio", ALPHA)


def test_diseases_with_a_missing_value_do_not_enter_the_slope():
    table = pd.DataFrame(
        {
            "onset": [0.0, 0.25, 0.5, 0.75, 1.0],
            "log_ratio": [0.4, np.nan, 0.2, np.nan, 0.0],
        }
    )
    fitted = log_r.slope(table, "log_ratio", ALPHA)
    assert fitted["n_diseases"] == 3
    assert fitted["slope"] == pytest.approx(-0.4, abs=1e-6)


def test_the_non_evolutionary_comparator_holds_no_evolutionary_score():
    """M0 must not contain conservation or constraint under another name."""
    forbidden = ("phylop", "phastcons", "s_het", "loeuf", "missense_z", "pli")
    for column in log_r.BASELINE_COLUMNS:
        assert not any(term in column.lower() for term in forbidden), column


def test_the_baseline_lift_floor_matches_the_plan():
    assert log_r.MIN_BASELINE_LIFT == 0.05
