"""Tests for the power simulation and its supporting arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.diagnostics import power
from src.features import matching
from src.onset import tiers


def test_auroc_and_d_are_inverses():
    for auroc in (0.55, 0.65, 0.75, 0.90):
        assert power.d_to_auroc(power.auroc_to_d(auroc)) == pytest.approx(auroc)


def test_no_discrimination_maps_to_zero_effect():
    assert power.auroc_to_d(0.5) == pytest.approx(0.0)


def test_rank_auroc_matches_a_hand_computed_case():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([0, 0, 1, 1])
    assert power.rank_auroc(scores, labels) == pytest.approx(1.0)
    assert power.rank_auroc(-scores, labels) == pytest.approx(0.0)


def test_rank_auroc_handles_ties_at_one_half():
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    labels = np.array([0, 1, 0, 1])
    assert power.rank_auroc(scores, labels) == pytest.approx(0.5)


def test_rank_auroc_is_undefined_without_both_classes():
    assert np.isnan(power.rank_auroc(np.array([1.0, 2.0]), np.array([1, 1])))


def test_design_reads_pairs_off_the_positives():
    positives = pd.DataFrame(
        {
            "disease_id": ["D1", "D1", "D2", "D3"],
            "gene_id": ["G1", "G2", "G3", "G4"],
            "onset_bin": ["perinatal", "perinatal", "perinatal", "adult"],
        }
    )
    design = power.design_from_positives(positives, k=10)
    assert design.bins == ["perinatal", "adult"]
    assert design.diseases_per_bin == [2, 1]
    assert design.pairs_per_bin == [3, 1]


def test_two_level_design_merges_everything_before_adulthood():
    positives = pd.DataFrame(
        {
            "disease_id": ["D1", "D2", "D3", "D4"],
            "gene_id": ["G1", "G2", "G3", "G4"],
            "onset_bin": ["perinatal", "childhood", "adolescent", "adult"],
        }
    )
    design = power.two_level_design(positives, k=10)
    assert design.bins == ["early", "adult"]
    assert design.diseases_per_bin == [3, 1]


def test_simulation_recovers_a_null_on_average():
    design = power.Design(
        bins=["perinatal", "adult"],
        diseases_per_bin=[200, 200],
        genes_per_disease={"perinatal": [1] * 200, "adult": [1] * 200},
        k_controls=10,
    )
    estimates = [power._simulate_once((design, 0.70, 0.0, 0.4, 10, seed))[0] for seed in range(30)]
    assert abs(float(np.nanmean(estimates))) < 0.15


def test_simulation_recovers_a_positive_effect_on_average():
    design = power.Design(
        bins=["perinatal", "adult"],
        diseases_per_bin=[400, 400],
        genes_per_disease={"perinatal": [1] * 400, "adult": [1] * 400},
        k_controls=10,
    )
    estimates = [power._simulate_once((design, 0.70, 0.10, 0.4, 10, seed))[0] for seed in range(30)]
    assert float(np.nanmean(estimates)) > 0.0


def test_mde_interpolates_between_grid_points():
    summary = pd.DataFrame(
        {
            "score_correlation": [0.4] * 4,
            "gamma": [0.0, 0.02, 0.04, 0.06],
            "power": [0.03, 0.40, 0.90, 0.99],
        }
    )
    mde = power.minimum_detectable_effect(summary, target=0.80)
    assert 0.02 < float(mde.loc[0, "mde_auroc_equivalent"]) < 0.04


def test_mde_is_undefined_when_the_grid_never_reaches_the_target():
    summary = pd.DataFrame(
        {"score_correlation": [0.4] * 3, "gamma": [0.0, 0.05, 0.10], "power": [0.02, 0.2, 0.5]}
    )
    assert np.isnan(power.minimum_detectable_effect(summary).loc[0, "mde_auroc_equivalent"])


def test_matched_controls_come_from_the_same_confounder_stratum():
    genes = pd.DataFrame(
        {
            "ensembl_gene_id": [f"G{i}" for i in range(40)],
            "cds_length_decile": [1] * 20 + [9] * 20,
            "expected_lof_decile": [1] * 20 + [9] * 20,
            "gc_content": [0.40] * 20 + [0.60] * 20,
        }
    )
    positives = pd.DataFrame({"disease_id": ["D1"], "gene_id": ["G0"]})
    matched = matching.draw_controls(positives, genes, k=5, seed=1)
    controls = set(matched.loc[matched["y"] == 0, "gene_id"])
    assert controls <= {f"G{i}" for i in range(1, 20)}
    assert "G0" not in controls


def test_smd_is_zero_when_positives_and_controls_agree():
    genes = pd.DataFrame(
        {
            "ensembl_gene_id": ["G0", "G1"],
            "cds_length": [1000.0, 1000.0],
            "gc_content": [0.5, 0.5],
        }
    )
    matched = pd.DataFrame({"gene_id": ["G0", "G1"], "y": [1, 0]})
    smd = matching.standardised_mean_differences(matched, genes, ["cds_length"])
    assert pd.isna(smd.loc[0, "smd"]) or smd.loc[0, "smd"] == pytest.approx(0.0)


def test_collapse_map_covers_every_raw_interval():
    from src.diagnostics import crosstab

    mapped = tiers.collapse(pd.Series(crosstab.RAW_ONSET_ORDER))
    assert mapped.notna().all()
    assert set(mapped) == set(tiers.collapsed_order())


def test_collapse_rejects_an_interval_it_does_not_know():
    with pytest.raises(ValueError, match="collapse map"):
        tiers.collapse(pd.Series(["middle age"]))


def test_collapse_counts_flag_a_bin_that_is_too_thin():
    counts = pd.Series({"perinatal": 1749, "adult": 12})
    warnings = tiers.check_collapse_counts(counts)
    assert len(warnings) == 1
    assert "adult" in warnings[0]
