"""Tests for the Evo 2 benchmark comparison.

The comparison decides whether the arm is used at all, so the two ways it can be wrong
are both tested: reading the score upside down, which would turn a good arm into
a failing one, and passing on too little data to mean anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evo2 import validation

SEED = 4


def _table(n: int = 2000, separation: float = 1.0, seed: int = SEED) -> pd.DataFrame:
    """Scored genes where essential ones carry a more negative Evo 2 score."""
    rng = np.random.default_rng(seed)
    essential = rng.random(n) < 0.1
    # A premature stop in an essential gene should surprise the model more, so
    # its score is more negative.
    score = rng.normal(size=n) - separation * essential
    return pd.DataFrame(
        {
            "ensembl_gene_id": [f"ENSG{i:08d}" for i in range(n)],
            "evo2_score": score,
            "essential": essential,
            "cds_length": rng.lognormal(mean=7.0, sigma=0.5, size=n),
            "gc_content": rng.uniform(0.3, 0.7, size=n),
        }
    )


def test_a_score_that_separates_essential_genes_is_read_the_right_way_up():
    """A more negative score means more essential, so the ranking is negated.

    Getting this backwards reports one minus the answer, which for a good arm
    lands near 0.3 and would be read as a failure to reproduce.
    """
    results = validation.evaluate(_table(), SEED)
    assert results["evo2_auroc"] > 0.5


def test_a_score_carrying_nothing_lands_at_chance():
    results = validation.evaluate(_table(separation=0.0), SEED)
    assert results["evo2_auroc"] == pytest.approx(0.5, abs=0.05)


def test_too_few_genes_is_reported_rather_than_scored():
    results = validation.evaluate(_table(n=100), SEED)
    assert "note" in results
    assert "evo2_auroc" not in results


def test_too_few_essential_genes_is_reported_rather_than_scored():
    table = _table()
    table["essential"] = False
    results = validation.evaluate(table, SEED)
    assert "note" in results


def test_the_tolerance_decides_whether_the_figure_is_reproduced():
    results = validation.evaluate(_table(separation=1.0), SEED)
    assert results["tolerance"] == validation.TOLERANCE
    assert results["published_auroc"] == validation.PUBLISHED_AUROC
    expected = abs(results["evo2_auroc"] - validation.PUBLISHED_AUROC) <= validation.TOLERANCE
    assert results["within_tolerance"] is expected


def test_the_baselines_are_reported_alongside():
    """An AUROC is only informative next to what length and composition reach."""
    results = validation.evaluate(_table(), SEED)
    for name in validation.BASELINES:
        assert f"{name}_auroc" in results
    assert "baselines_together_auroc" in results
    assert "baselines_plus_evo2_auroc" in results


def test_adding_a_real_score_to_the_baselines_improves_them():
    results = validation.evaluate(_table(separation=1.5), SEED)
    assert results["baselines_plus_evo2_auroc"] > results["baselines_together_auroc"]
