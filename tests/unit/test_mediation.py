"""Tests for the mediation blocks and the component split.

The component split exists to say which covariate carries the movement under
conditioning, so the only thing that makes it readable is that every component
is fitted on identical rows. These tests hold that property in place, and hold
the component list in step with the blocks it is meant to cover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import mediation, metric_mechanism, primary
from tests.fixtures import synthetic

SEED = 11


def design_and_covariates(missing: int = 0) -> tuple[primary.Design, pd.DataFrame]:
    positives, features = synthetic.generate(gamma=0.0, seed=SEED)
    design = primary.build_design(
        positives,
        features,
        synthetic.CONSERVATION_COLUMN,
        synthetic.CONSTRAINT_COLUMN,
        seed=SEED,
        k=10,
        matched=True,
    )
    genes = pd.unique(design.gene)
    rng = np.random.default_rng(SEED)
    frame = pd.DataFrame({"ensembl_gene_id": genes})
    for column in metric_mechanism.COMPONENT_ORIGIN:
        frame[column] = rng.normal(size=len(genes))
    if missing:
        for column in metric_mechanism.COMPONENT_ORIGIN:
            frame.loc[rng.choice(len(genes), missing, replace=False), column] = np.nan
    return design, frame


def test_every_block_covariate_is_classified_by_origin():
    """A covariate added to a block without an origin would vanish from the split."""
    covered = set(mediation.BLOCKS["pleiotropy"]) | set(mediation.BLOCKS["gene_class"])
    assert covered <= set(metric_mechanism.COMPONENT_ORIGIN)


def test_the_withdrawn_covariate_is_classified_but_not_fitted():
    """Disease breadth is reported on its own and entered in no block."""
    covered = set(mediation.BLOCKS["pleiotropy"]) | set(mediation.BLOCKS["gene_class"])
    assert set(metric_mechanism.COMPONENT_ORIGIN) == covered | set(mediation.WITHDRAWN)
    assert set(mediation.WITHDRAWN) and not (set(mediation.WITHDRAWN) & covered)


def test_restricting_on_complete_covariates_keeps_every_row():
    design, covariates = design_and_covariates(missing=0)
    columns = list(metric_mechanism.COMPONENT_ORIGIN)
    restricted = mediation.restrict(design, covariates, columns)
    assert restricted.x.shape == design.x.shape


def test_restricting_drops_the_rows_a_covariate_does_not_cover():
    design, covariates = design_and_covariates(missing=40)
    columns = list(metric_mechanism.COMPONENT_ORIGIN)
    restricted = mediation.restrict(design, covariates, columns)
    assert restricted.x.shape[0] < design.x.shape[0]
    assert restricted.x.shape[1] == design.x.shape[1]
    assert restricted.names == design.names


def test_each_component_is_fitted_on_the_same_rows_after_restriction():
    """The property the split rests on.

    Conditioning is compared across components, so a component that covers fewer
    genes must not also be a change of sample. Once the design is restricted to
    rows complete on every component, adding any one of them leaves the row
    count alone.
    """
    design, covariates = design_and_covariates(missing=40)
    columns = list(metric_mechanism.COMPONENT_ORIGIN)
    restricted = mediation.restrict(design, covariates, columns)
    for column in columns:
        widened = mediation.augment(restricted, covariates, [column])
        assert widened.x.shape[0] == restricted.x.shape[0]
        assert widened.x.shape[1] == restricted.x.shape[1] + 3


def test_augmenting_without_restriction_changes_the_sample():
    """Why the restriction is needed rather than assumed.

    Without it, a specification is fitted on the rows its covariate happens to
    cover and the comparison carries a change of sample inside it.
    """
    design, covariates = design_and_covariates(missing=40)
    widened = mediation.augment(design, covariates, list(metric_mechanism.COMPONENT_ORIGIN))
    assert widened.x.shape[0] < design.x.shape[0]


def test_a_covariate_built_to_track_onset_is_detected_as_varying():
    """The gradient check has to be able to see a gradient that is there.

    A null from a covariate that does not vary with onset is not evidence, so the
    check that separates the two cases has to be shown to work in both.
    """
    design, covariates = design_and_covariates(missing=0)
    onset_for_gene = pd.Series(design.onset, index=design.gene).groupby(level=0).mean()
    # The gene order follows the design rows, which follow onset, so an unshuffled
    # index would track the axis rather than being independent of it.
    flat = np.arange(len(covariates), dtype=float)
    np.random.default_rng(SEED).shuffle(flat)
    covariates = covariates.assign(
        tracks=covariates["ensembl_gene_id"].map(onset_for_gene), flat=flat
    )
    table = mediation.covariate_gradient(
        design, covariates, ["tracks", "flat"], seed=SEED, draws=40
    )
    among_genes = table[table["stratum"] == "positives"].set_index("covariate")
    assert bool(among_genes.loc["tracks", "varies_with_onset"])
    assert abs(float(among_genes.loc["tracks", "spearman_with_onset"])) > 0.5
    assert abs(float(among_genes.loc["flat", "spearman_with_onset"])) < 0.1
