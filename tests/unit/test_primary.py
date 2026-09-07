"""Tests for the primary model, run against synthetic positives only.

The real estimands are not fitted here. Pipeline correctness is checked against
a generator with a known Gamma, so that development does not repeatedly inspect
the quantity the pre-registration fixes in advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import estimator, primary
from tests.fixtures import synthetic

SEED = 7


def design_for(gamma: float, matched: bool = True, seed: int = SEED) -> primary.Design:
    positives, features = synthetic.generate(gamma=gamma, seed=seed)
    return primary.build_design(
        positives,
        features,
        synthetic.CONSERVATION_COLUMN,
        synthetic.CONSTRAINT_COLUMN,
        seed=seed,
        k=10,
        matched=matched,
    )


# Measured on the synthetic generator over ten seeds per point: the fitted
# contrast is linear in the generated effect through the origin, at about nine
# and a third logit units per AUROC-equivalent unit. The pre-registered minimum
# detectable effect of 0.035 AUROC-equivalent is therefore about 0.33 on the
# scale the model reports.
LOGIT_PER_AUROC = 9.33


def test_the_model_recovers_a_null_when_none_was_generated():
    estimates = [primary.estimate_gamma(design_for(0.0, seed=seed))["gamma"] for seed in range(8)]
    spread = float(np.std(estimates, ddof=1)) / np.sqrt(len(estimates))
    assert abs(float(np.mean(estimates))) < 3 * spread


def test_the_model_recovers_the_sign_of_a_generated_effect():
    """The sign convention has been wrong once. Gamma is a contrast of decays.

    A score whose discrimination falls with onset has a negative interaction
    coefficient, so the decay is the negative of it. Writing the contrast
    straight onto the interaction coefficients reverses the primary estimand.
    """
    estimates = [primary.estimate_gamma(design_for(0.12, seed=seed))["gamma"] for seed in range(4)]
    assert float(np.mean(estimates)) > 0.0


def test_gamma_is_the_decay_contrast_and_not_its_negation():
    assert primary.GAMMA_CONTRAST["constraint_x_onset"] > 0
    assert primary.GAMMA_CONTRAST["conservation_x_onset"] < 0


def test_the_estimate_scales_with_the_generated_effect():
    points = {
        gamma: float(
            np.mean([primary.estimate_gamma(design_for(gamma, seed=s))["gamma"] for s in range(4)])
        )
        for gamma in (0.05, 0.20)
    }
    assert points[0.20] > points[0.05]
    implied = (points[0.20] - points[0.05]) / 0.15
    assert 0.5 * LOGIT_PER_AUROC < implied < 2.0 * LOGIT_PER_AUROC


def test_the_two_control_arms_carry_the_same_positives():
    matched = design_for(0.10, matched=True)
    unmatched = design_for(0.10, matched=False)
    assert matched.metadata["n_positives"] == unmatched.metadata["n_positives"]
    # The matched arm is smaller because a confounder stratum cannot always
    # supply k controls. That shortfall is a property of matching,
    # and it is recorded rather than filled from another stratum.
    assert matched.metadata["n_rows"] <= unmatched.metadata["n_rows"]


def test_inheritance_terms_appear_only_where_the_mode_varies():
    positives, features = synthetic.generate(gamma=0.1, seed=SEED)
    design = primary.build_design(
        positives, features, synthetic.CONSERVATION_COLUMN, synthetic.CONSTRAINT_COLUMN, SEED, 10
    )
    assert "conservation_x_dominant" in design.names

    single_mode = positives.assign(inheritance_mode="recessive")
    flat = primary.build_design(
        single_mode, features, synthetic.CONSERVATION_COLUMN, synthetic.CONSTRAINT_COLUMN, SEED, 10
    )
    assert "conservation_x_dominant" not in flat.names
    assert flat.metadata["inheritance_terms"] is False


def test_onset_position_spans_the_unit_interval():
    import pandas as pd

    positions = primary.onset_position(pd.Series(synthetic.ONSET_LEVELS))
    assert positions.min() == pytest.approx(0.0)
    assert positions.max() == pytest.approx(1.0)
    assert np.all(np.diff(positions) > 0)


def test_the_permutation_null_is_centred_on_zero():
    design = design_for(0.12)
    null = primary.permutation_null(design, draws=25, seed=SEED)
    assert abs(float(np.mean(null))) < 0.1
    assert float(np.std(null)) > 0.0


def test_a_rank_deficient_design_is_refused_rather_than_fitted():
    x = np.column_stack([np.ones(50), np.ones(50)])
    y = np.repeat([0.0, 1.0], 25)
    with pytest.raises(ValueError, match="rank deficient"):
        estimator.fit(x, y, ["a", "b"])


def test_clustering_every_row_separately_reproduces_the_unclustered_sandwich():
    """An exact identity, and the sharpest available check on the meat matrix."""
    design = design_for(0.10)
    plain = estimator.fit(design.x, design.y, design.names)
    singleton = estimator.fit(
        design.x, design.y, design.names, clusters={"row": np.arange(len(design.y))}
    )
    assert np.allclose(plain.covariance, singleton.covariance, atol=1e-10)


def test_clustering_changes_the_interval_and_keeps_it_finite():
    design = design_for(0.10)
    plain = estimator.fit(design.x, design.y, design.names)
    clustered = estimator.fit(
        design.x, design.y, design.names, clusters={"d": design.disease, "g": design.gene}
    )
    plain_se = plain.contrast(primary.GAMMA_CONTRAST)[1]
    clustered_se = clustered.contrast(primary.GAMMA_CONTRAST)[1]
    assert np.isfinite(clustered_se)
    assert clustered_se != pytest.approx(plain_se, rel=1e-6)
    assert clustered.n_clusters == {
        "d": len(np.unique(design.disease)),
        "g": len(np.unique(design.gene)),
    }


def test_contrast_reports_the_difference_of_the_two_interaction_terms():
    design = design_for(0.10)
    fit = estimator.fit(design.x, design.y, design.names)
    estimate, _ = fit.contrast(primary.GAMMA_CONTRAST)
    manual = (
        fit.coefficients[fit.index("constraint_x_onset")]
        - fit.coefficients[fit.index("conservation_x_onset")]
    )
    assert estimate == pytest.approx(manual)


def test_the_mendelian_arm_reads_the_interval_column():
    positives = pd.DataFrame({"evidence_source": ["a_mendelian"] * 3})
    assert primary.onset_column_for(positives) == "onset_bin"


def test_the_platform_arms_read_the_decade_column():
    """These arms are measured as a median age in years, not as an interval.

    Reading the Mendelian column for them keeps only the diseases that happen to
    carry an Orphanet interval as well, which is about a fifth of the arm, and
    puts those on an axis they were not measured against.
    """
    for arm in ("b_gwas", "c_clinvar", "d_gene_burden"):
        positives = pd.DataFrame({"evidence_source": [arm] * 3})
        assert primary.onset_column_for(positives) == "onset_decade"


def test_mixing_arms_with_different_axes_is_refused():
    positives = pd.DataFrame({"evidence_source": ["a_mendelian", "b_gwas"]})
    with pytest.raises(ValueError, match="mix onset axes"):
        primary.onset_column_for(positives)


def test_onset_position_scales_both_vocabularies_onto_the_unit_interval():
    from src.onset import tiers

    named = pd.Series(tiers.collapsed_order())
    decades = pd.Series(["0-10", "10-20", "30-40", "70-100"])
    for series in (named, decades):
        position = primary.onset_position(series)
        assert np.nanmin(position) == pytest.approx(0.0)
        assert np.nanmax(position) == pytest.approx(1.0)
        assert np.all(np.diff(position) > 0)


def test_onset_position_returns_not_a_number_for_an_unknown_bin():
    position = primary.onset_position(pd.Series(["perinatal", "not a bin", "adult"]))
    assert np.isnan(position[1])
    assert np.isfinite(position[0]) and np.isfinite(position[2])


def test_the_design_records_which_onset_column_it_used():
    design = design_for(0.10)
    assert design.metadata["onset_column"] in {"onset_bin", "onset_decade"}


def _crosstab(tmp_path, source: str, n_diseases: int, name: str = "crosstab.parquet"):
    directory = tmp_path / "analysis_set"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "evidence_source": [source, source],
            "onset_bin": ["perinatal", "unassigned"],
            "n_diseases": [n_diseases, 9999],
            "n_pairs": [10, 10],
        }
    ).to_parquet(directory / name, index=False)
    return tmp_path


def test_a_design_that_lost_most_of_its_arm_is_refused(tmp_path, monkeypatch):
    """Reading the wrong onset column produces a design
    with a fifth of the complex arm, correct in every local respect."""
    crosstab = _crosstab(tmp_path, "b_gwas", 521, "crosstab_complex.parquet")
    monkeypatch.setattr(primary.config, "derived_dir", lambda: crosstab)
    positives = pd.DataFrame({"evidence_source": ["b_gwas"] * 3})
    with pytest.raises(ValueError, match="crosstab"):
        primary.check_design_against_crosstab(positives, "onset_decade", kept=105)


def test_a_design_that_merely_trimmed_is_allowed(tmp_path, monkeypatch):
    """Losing genes with no score is normal and must not raise."""
    crosstab = _crosstab(tmp_path, "b_gwas", 521, "crosstab_complex.parquet")
    monkeypatch.setattr(primary.config, "derived_dir", lambda: crosstab)
    positives = pd.DataFrame({"evidence_source": ["b_gwas"] * 3})
    primary.check_design_against_crosstab(positives, "onset_decade", kept=517)


def test_the_unassigned_row_is_not_counted_as_an_expectation(tmp_path, monkeypatch):
    crosstab = _crosstab(tmp_path, "b_gwas", 521, "crosstab_complex.parquet")
    monkeypatch.setattr(primary.config, "derived_dir", lambda: crosstab)
    assert primary.expected_diseases("b_gwas", "onset_decade") == 521


def test_the_guard_is_silent_when_there_is_nothing_to_check_against(tmp_path, monkeypatch):
    """Synthetic positives have no crosstab, and the guard must not invent one."""
    monkeypatch.setattr(primary.config, "derived_dir", lambda: tmp_path)
    positives = pd.DataFrame({"evidence_source": ["synthetic"] * 3})
    primary.check_design_against_crosstab(positives, "onset_bin", kept=1)


def test_the_guard_is_silent_when_arms_are_mixed(tmp_path, monkeypatch):
    crosstab = _crosstab(tmp_path, "b_gwas", 521, "crosstab_complex.parquet")
    monkeypatch.setattr(primary.config, "derived_dir", lambda: crosstab)
    positives = pd.DataFrame({"evidence_source": ["b_gwas", "a_mendelian"]})
    primary.check_design_against_crosstab(positives, "onset_decade", kept=1)


def test_the_metric_comparison_refuses_a_matrix_missing_a_score():
    """The comparison is only meaningful on genes carrying every metric, so a
    matrix that cannot supply one has to say so rather than silently compare
    different gene sets."""
    from src.models import metric_mechanism

    with pytest.raises(KeyError, match="lacks"):
        metric_mechanism.common_gene_set(pd.DataFrame({"ensembl_gene_id": ["ENSG00000000001"]}))


def test_the_metric_comparison_keeps_only_genes_carrying_every_score():
    from src.features import evolutionary
    from src.models import metric_mechanism

    columns = [evolutionary.CONSERVATION_PRIMARY, *metric_mechanism.CONSTRAINT_METRICS.values()]
    frame = pd.DataFrame({"ensembl_gene_id": [f"ENSG{i:08d}" for i in range(4)]})
    for name in columns:
        frame[name] = [1.0, 2.0, 3.0, 4.0]
    frame.loc[2, columns[1]] = None
    kept = metric_mechanism.common_gene_set(frame)
    assert len(kept) == 3


def test_every_onset_encoding_lands_on_the_unit_interval():
    """An encoding left in years returns a Gamma smaller by the width of the axis.

    That reads as a precise null rather than as a different unit, so the scaling
    is held here rather than repeated in each encoding.
    """
    bins = pd.Series(["perinatal", "infancy", "childhood", "adolescent", "adult"])
    for encoded in (primary.onset_position(bins), primary.onset_midpoint(bins)):
        assert np.nanmin(encoded) == pytest.approx(0.0)
        assert np.nanmax(encoded) == pytest.approx(1.0)

    years = np.array([-0.4, 0.5, 6.0, 35.8, 72.0])
    scaled = primary.scale_to_unit(years)
    assert np.nanmin(scaled) == pytest.approx(0.0)
    assert np.nanmax(scaled) == pytest.approx(1.0)
    assert primary.scale_to_unit(np.array([3.0, 3.0])).tolist() == [0.0, 0.0]
