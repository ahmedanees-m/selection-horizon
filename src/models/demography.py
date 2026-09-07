"""Hamilton's selection gradient and the fitness cost of a disease.

Two quantities are computed and they are not the same thing.

Residual reproductive weight, the tail of discounted future reproduction:

    W(a) = sum_{x >= a} exp(-r x) l(x) m(x)

W is a tail sum of non-negative terms, so it is monotone non-increasing in a by
construction. It is the weight Hamilton's gradient places on an age.

Fisher reproductive value, the same tail renormalised by the chance of reaching
age a:

    v(a) = (exp(r a) / l(a)) * sum_{x >= a} exp(-r x) l(x) m(x)

v rises through the pre-reproductive years, because surviving them makes future
reproduction more certain, peaks around the age at first reproduction, and falls
to zero past the end of the fertile span. The two are related by
W(a) = exp(-r a) l(a) v(a).

The execution plan states the validation criterion for this module as a peak in
the late teens or early twenties. That is a property of v, not of W. Both are
returned and both are checked.

The exponential discount is not optional. An earlier draft of the project
omitted it; tests/unit/test_demography.py fails if it is dropped again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LifeHistory:
    """Age-indexed survivorship and fertility on a one-year grid.

    survivorship  l(x), probability of surviving from birth to exact age x
    fertility     m(x), expected offspring produced between age x and x + 1
    """

    ages: np.ndarray
    survivorship: np.ndarray
    fertility: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        if not (len(self.ages) == len(self.survivorship) == len(self.fertility)):
            raise ValueError("ages, survivorship and fertility must be the same length")
        if np.any(np.diff(self.survivorship) > 1e-12):
            raise ValueError("survivorship must be non-increasing in age")
        if np.any(self.fertility < 0):
            raise ValueError("fertility must be non-negative")


def intrinsic_growth_rate(history: LifeHistory, tol: float = 1e-12) -> float:
    """Solve the Euler-Lotka equation for r by bisection."""

    def characteristic(rate: float) -> float:
        discounted = np.exp(-rate * history.ages) * history.survivorship * history.fertility
        return float(np.sum(discounted))

    net_reproduction = characteristic(0.0)
    if net_reproduction <= 0:
        raise ValueError("net reproduction is zero; r is undefined")

    low, high = -1.0, 1.0
    while characteristic(low) < 1.0:
        low *= 2
        if low < -10:
            raise ValueError("failed to bracket r from below")
    while characteristic(high) > 1.0:
        high *= 2
        if high > 10:
            raise ValueError("failed to bracket r from above")

    while high - low > tol:
        mid = (low + high) / 2
        if characteristic(mid) > 1.0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def residual_reproductive_weight(history: LifeHistory, r: float | None = None) -> np.ndarray:
    """W(a) on the age grid of the life history."""
    rate = intrinsic_growth_rate(history) if r is None else r
    contribution = np.exp(-rate * history.ages) * history.survivorship * history.fertility
    return np.flip(np.cumsum(np.flip(contribution)))


def reproductive_value(history: LifeHistory, r: float | None = None) -> np.ndarray:
    """v(a) on the age grid, zero where survivorship has reached zero."""
    rate = intrinsic_growth_rate(history) if r is None else r
    weight = residual_reproductive_weight(history, rate)
    scale = np.exp(rate * history.ages)
    value = np.zeros_like(weight)
    alive = history.survivorship > 0
    value[alive] = scale[alive] * weight[alive] / history.survivorship[alive]
    return value


def kappa_from_death_age(
    history: LifeHistory,
    onset_age: float,
    death_age: float,
    r: float | None = None,
) -> float:
    """Fraction of residual reproductive weight destroyed between onset and death.

    kappa_d = 1 - W(age_death) / W(age_onset)

    Returns 1.0 when the disease removes all remaining reproduction and 0.0 when
    it removes none. Death before onset is treated as an input error.
    """
    if death_age < onset_age:
        raise ValueError("age at death precedes age at onset")
    weight = residual_reproductive_weight(history, r)
    at_onset = float(np.interp(onset_age, history.ages, weight))
    if at_onset <= 0:
        return 0.0
    at_death = float(np.interp(death_age, history.ages, weight))
    return float(np.clip(1.0 - at_death / at_onset, 0.0, 1.0))


def disease_fitness_cost(
    history: LifeHistory,
    hazard: np.ndarray,
    kappa: float,
    r: float | None = None,
) -> float:
    """s_d = kappa_d * sum_a h_d(a) W(a) / W(0).

    hazard is the age-specific hazard of the disease on the same age grid as the
    life history. It must be a hazard, not a cumulative incidence: applying an
    ancestral life table to a modern cumulative incidence double counts modern
    survival and forces the modern and ancestral indices to agree. The check for
    that mistake lives in src/models/demography.py::correlation_diagnostic.
    """
    if len(hazard) != len(history.ages):
        raise ValueError("hazard and life history must share an age grid")
    weight = residual_reproductive_weight(history, r)
    if weight[0] <= 0:
        return 0.0
    return float(kappa * np.sum(hazard * weight) / weight[0])


def allele_fitness_cost(delta_penetrance: float, disease_cost: float) -> float:
    """s_allele = delta_pi * s_d.

    Without the penetrance differential the complex-disease arm is flat for a
    trivial reason: an allele with an odds ratio of 1.05 carries almost no
    fitness cost at any onset.
    """
    return float(delta_penetrance * disease_cost)


def cif_to_hazard(cumulative_incidence: np.ndarray) -> np.ndarray:
    """Discrete-time hazard from an Aalen-Johansen cumulative incidence function.

    h(a) = [F(a + 1) - F(a)] / [1 - F(a)]

    The Risteys cumulative incidence already treats death as a competing event
    and is stratified by sex with age as the timescale, which is the input this
    conversion assumes.
    """
    cif = np.asarray(cumulative_incidence, dtype=float)
    if np.any(np.diff(cif) < -1e-9):
        raise ValueError("cumulative incidence must be non-decreasing")
    increments = np.diff(cif, append=cif[-1])
    at_risk = np.clip(1.0 - cif, 1e-12, None)
    return np.clip(increments / at_risk, 0.0, 1.0)


def correlation_diagnostic(modern_index: np.ndarray, ancestral_index: np.ndarray) -> float:
    """Spearman correlation between the modern and ancestral fitness indices.

    A value above the ceiling in config/analysis.yaml means the hazard
    conversion was skipped somewhere upstream. The two indices should differ;
    if they do not, the ancestral arm carries no information and H4 cannot be
    tested.
    """
    from scipy.stats import spearmanr

    return float(spearmanr(modern_index, ancestral_index).statistic)
