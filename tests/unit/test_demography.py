"""Tests for the selection-gradient module.

Each test corresponds to a mistake that was made or nearly made while the
design was being written.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models import demography


def simple_history(max_age: int = 60) -> demography.LifeHistory:
    """A life history with constant survivorship and a rectangular fertile span.

    Constant survivorship over the fertile span makes the closed form for W
    tractable by hand, which is the point.
    """
    ages = np.arange(max_age + 1, dtype=float)
    survivorship = np.ones_like(ages)
    survivorship[ages > 45] = 0.0
    fertility = np.where((ages >= 15) & (ages < 45), 0.1, 0.0)
    return demography.LifeHistory(ages, survivorship, fertility, label="rectangular")


def declining_history(max_age: int = 90) -> demography.LifeHistory:
    ages = np.arange(max_age + 1, dtype=float)
    survivorship = np.exp(-0.02 * ages)
    fertility = np.where((ages >= 16) & (ages < 45), 0.2, 0.0)
    return demography.LifeHistory(ages, survivorship, fertility, label="declining")


def test_residual_weight_is_monotone_non_increasing():
    history = declining_history()
    weight = demography.residual_reproductive_weight(history, r=0.0)
    assert np.all(np.diff(weight) <= 1e-12)


def test_residual_weight_reaches_zero_past_the_fertile_span():
    history = declining_history()
    weight = demography.residual_reproductive_weight(history, r=0.0)
    assert weight[history.ages >= 45][0] == pytest.approx(0.0, abs=1e-12)


def test_reproductive_value_peaks_inside_the_fertile_span():
    """The peak criterion belongs to v, not to W."""
    history = declining_history()
    value = demography.reproductive_value(history, r=0.0)
    peak_age = history.ages[int(np.argmax(value))]
    assert 14 <= peak_age <= 25


def test_discount_term_is_applied():
    """The v1.0 formula omitted exp(-r x). A positive r must lower W."""
    history = simple_history()
    undiscounted = demography.residual_reproductive_weight(history, r=0.0)
    discounted = demography.residual_reproductive_weight(history, r=0.03)
    assert discounted[0] < undiscounted[0]
    assert np.all(discounted <= undiscounted + 1e-12)


def test_euler_lotka_solves_to_unit_net_reproduction():
    history = declining_history()
    r = demography.intrinsic_growth_rate(history)
    characteristic = np.sum(np.exp(-r * history.ages) * history.survivorship * history.fertility)
    assert characteristic == pytest.approx(1.0, abs=1e-8)


def test_kappa_is_one_when_death_removes_all_remaining_reproduction():
    history = simple_history()
    assert demography.kappa_from_death_age(history, 20.0, 46.0, r=0.0) == pytest.approx(1.0)


def test_kappa_is_zero_when_onset_follows_the_fertile_span():
    history = simple_history()
    assert demography.kappa_from_death_age(history, 50.0, 60.0, r=0.0) == pytest.approx(0.0)


def test_kappa_rejects_death_before_onset():
    history = simple_history()
    with pytest.raises(ValueError):
        demography.kappa_from_death_age(history, 40.0, 30.0, r=0.0)


def test_cif_to_hazard_recovers_a_constant_hazard():
    rate = 0.05
    ages = np.arange(0, 40)
    cif = 1.0 - np.exp(-rate * ages)
    hazard = demography.cif_to_hazard(cif)
    assert np.allclose(hazard[:-1], 1.0 - np.exp(-rate), atol=1e-9)


def test_cif_to_hazard_rejects_a_decreasing_input():
    with pytest.raises(ValueError):
        demography.cif_to_hazard(np.array([0.0, 0.2, 0.1]))


def test_early_onset_costs_more_than_late_onset():
    history = declining_history()
    ages = history.ages
    early = np.where(ages == 10, 1.0, 0.0)
    late = np.where(ages == 60, 1.0, 0.0)
    cost_early = demography.disease_fitness_cost(history, early, kappa=1.0, r=0.0)
    cost_late = demography.disease_fitness_cost(history, late, kappa=1.0, r=0.0)
    assert cost_early > cost_late
    assert cost_late == pytest.approx(0.0, abs=1e-12)


def test_allele_cost_scales_with_penetrance_differential():
    assert demography.allele_fitness_cost(0.0, 0.4) == pytest.approx(0.0)
    assert demography.allele_fitness_cost(0.5, 0.4) == pytest.approx(0.2)


def test_life_history_rejects_increasing_survivorship():
    ages = np.arange(5, dtype=float)
    with pytest.raises(ValueError):
        demography.LifeHistory(ages, np.array([0.5, 0.6, 0.7, 0.8, 0.9]), np.zeros(5))
