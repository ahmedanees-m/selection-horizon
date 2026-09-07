"""Tests for the control analyses, run against synthetic positives.

The controls decide whether the real result can be read at all, so what has to
be right is that they detect a gradient when one is there and report none when
it is not. Both are testable on a generator: a column built to decay along the
onset axis must come back with a positive decay, and a column of noise must come
back flat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import controls
from tests.fixtures import synthetic

SEED = 7


@pytest.fixture(scope="module")
def synthetic_inputs():
    positives, features = synthetic.generate(gamma=0.12, seed=SEED)
    rng = np.random.default_rng(SEED)

    # A column with no relation to anything, which every negative control is
    # meant to look like, and a column built to carry the onset gradient the
    # positive control is meant to find.
    features = features.assign(
        is_autosomal=True,
        is_hla=False,
        is_olfactory=False,
        noise=rng.normal(size=len(features)),
    )
    onset = positives[["gene_id", "onset_bin"]].drop_duplicates("gene_id")
    position = onset["onset_bin"].map(
        {name: index for index, name in enumerate(synthetic.ONSET_LEVELS)}
    )
    gradient = pd.DataFrame(
        {
            "ensembl_gene_id": onset["gene_id"].to_numpy(),
            "declining": (4.0 - position.to_numpy()) + rng.normal(scale=0.5, size=len(onset)),
        }
    )
    features = features.merge(gradient, on="ensembl_gene_id", how="left")
    features["declining"] = features["declining"].fillna(float(np.nanmean(features["declining"])))
    return positives, features


def test_every_control_declares_a_kind_an_expectation_and_a_rationale():
    for control in controls.controls():
        assert control.kind in {"negative", "positive", "placebo"}
        assert control.expectation
        assert len(control.rationale) > 30


def test_control_names_are_unique():
    names = [control.name for control in controls.controls()]
    assert len(names) == len(set(names))


def test_the_suite_carries_a_positive_control():
    assert any(control.kind == "positive" for control in controls.controls())


def test_controls_not_run_as_specified_are_declared():
    assert controls.NOT_RUN
    for reason in controls.NOT_RUN.values():
        assert len(reason) > 30


def test_a_score_absent_from_the_matrix_is_reported_rather_than_crashing(synthetic_inputs):
    positives, features = synthetic_inputs
    control = controls.Control("absent", "negative", "not_a_column", "flat", "a" * 40)
    row = controls.run_control(positives, features, control, SEED, 10)
    assert np.isnan(row["decay"])
    assert "not in the feature matrix" in row["note"]


def test_a_noise_column_comes_back_flat(synthetic_inputs):
    positives, features = synthetic_inputs
    control = controls.Control("noise", "negative", "noise", "flat", "a" * 40)
    row = controls.run_control(positives, features, control, SEED, 10)
    assert np.isfinite(row["decay"])
    assert not row["excludes_zero"]
    assert controls.as_expected(row)


def test_a_column_built_to_decay_is_detected(synthetic_inputs):
    positives, features = synthetic_inputs
    control = controls.Control("declining", "positive", "declining", "steep decay", "a" * 40)
    row = controls.run_control(positives, features, control, SEED, 10)
    assert row["decay"] > 0
    assert row["excludes_zero"]
    assert controls.as_expected(row)


def test_a_positive_control_that_stays_flat_stops_the_gate(synthetic_inputs):
    positives, features = synthetic_inputs
    control = controls.Control("noise", "positive", "noise", "steep decay", "a" * 40)
    row = controls.run_control(positives, features, control, SEED, 10)
    assert not controls.as_expected(row)


def test_the_evolutionary_scores_are_not_in_the_control_design(synthetic_inputs):
    """A negative control conditioned on the scores it must be independent of
    would pass by construction."""
    from src.features import evolutionary
    from src.models import primary

    positives, features = synthetic_inputs
    design = primary.build_design(
        positives,
        features,
        evolutionary.CONSERVATION_PRIMARY,
        evolutionary.CONSTRAINT_PRIMARY,
        SEED,
        10,
        True,
    )
    values = np.ones(len(design.gene)) * np.arange(len(design.gene))
    recast = controls.control_design(design, values, ())
    assert "z_conservation" not in recast["names"]
    assert "conservation_x_onset" not in recast["names"]
    assert "control_x_onset" in recast["names"]
    assert "cds_length_decile" in recast["names"]


def test_a_collinear_covariate_is_dropped_when_it_is_the_tested_score(synthetic_inputs):
    from src.features import evolutionary
    from src.models import primary

    positives, features = synthetic_inputs
    design = primary.build_design(
        positives,
        features,
        evolutionary.CONSERVATION_PRIMARY,
        evolutionary.CONSTRAINT_PRIMARY,
        SEED,
        10,
        True,
    )
    recast = controls.control_design(design, np.arange(len(design.gene)), ("cds_length_decile",))
    assert "cds_length_decile" not in recast["names"]


def _null(mean: float, spread: float, draws: int, seed: int = SEED) -> np.ndarray:
    """Draws with exactly the requested mean, so a test of the criterion is not
    also a test of how far a sample mean wanders."""
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=spread, size=draws)
    return values - values.mean() + mean


def test_a_centred_null_is_recognised():
    summary = controls.summarise_null(_null(0.0, 0.2, 400))
    assert summary["centred"]
    assert summary["offset_in_null_standard_deviations"] < controls.CENTRED_TOLERANCE


def test_the_centring_criterion_does_not_tighten_with_the_draw_count():
    """A null offset by a negligible amount must pass at any number of draws.

    Measuring the offset against the standard error of the mean would fail this,
    because that error shrinks with the draw count while the offset does not.
    """
    offsets = []
    for draws in (200, 2000, 20000):
        summary = controls.summarise_null(_null(0.002, 0.2, draws))
        assert summary["centred"], draws
        offsets.append(summary["offset_in_null_standard_deviations"])
    # The offset is a property of the null and not of how many times it was
    # drawn, so it must not drift systematically with the draw count.
    assert max(offsets) / min(offsets) < 1.5


def test_a_null_too_short_to_resolve_an_offset_is_not_called_a_stop():
    """The mean of a handful of draws wanders by a large fraction of the spread,
    and the threshold must not read that as model misspecification."""
    rng = np.random.default_rng(SEED)
    summary = controls.summarise_null(rng.normal(scale=0.2, size=5))
    assert summary["centred"]
    assert not summary["mean_differs_from_zero"]


def test_a_large_and_resolved_offset_is_called_a_stop():
    summary = controls.summarise_null(_null(0.5, 0.2, 400))
    assert summary["offset_is_material"]
    assert summary["mean_differs_from_zero"]
    assert not summary["centred"]


def test_an_off_centre_null_is_recognised():
    summary = controls.summarise_null(_null(0.5, 0.2, 400))
    assert not summary["centred"]


def test_a_small_but_real_offset_is_visible_without_stopping_the_gate():
    """An offset of a twentieth of the spread is negligible against the threshold and
    still large enough to be resolved at twenty thousand draws."""
    summary = controls.summarise_null(_null(0.01, 0.2, 20000))
    assert summary["centred"]
    assert summary["mean_differs_from_zero"]


def test_a_null_with_too_few_draws_is_reported_rather_than_summarised():
    summary = controls.summarise_null(np.array([0.1]))
    assert "note" in summary


def test_the_permutation_is_confined_to_the_stratum(synthetic_inputs):
    """Onset must move within therapeutic area and not across it."""
    from src.features import evolutionary
    from src.models import primary

    positives, features = synthetic_inputs
    design = primary.build_design(
        positives,
        features,
        evolutionary.CONSERVATION_PRIMARY,
        evolutionary.CONSTRAINT_PRIMARY,
        SEED,
        10,
        True,
    )
    # One stratum per disease leaves nothing to permute, so the null must be a
    # spike at the observed value rather than a spread.
    strata = pd.Series(
        {disease: f"area_{index}" for index, disease in enumerate(np.unique(design.disease))}
    )
    values = controls.stratified_permutation_null(design, strata, draws=3, seed=SEED)
    assert np.allclose(values, values[0])
