"""Logistic regression with multi-way cluster-robust covariance.

The design specifies cross-classified random effects for disease and gene,
because genes recur across diseases through pleiotropy and diseases cluster
their genes. What those random effects are there to do is get the standard
errors right on a contrast estimated across correlated rows.

A two-way cluster-robust sandwich does the same job without fitting a mixed
model: it is consistent under arbitrary correlation within a disease and within
a gene, it makes no distributional assumption about either, and it is fast
enough to run a thousand-replicate permutation null on. The estimator used here
is the Cameron, Gelbach and Miller construction, which adds the two one-way
meats and subtracts the intersection meat.

The trade is that the coefficients are population-averaged rather than
conditional on a random effect. For a contrast of two interaction terms, which
is what Gamma is, that is the quantity the design wants anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_ITERATIONS = 100
TOLERANCE = 1e-9
LINEAR_PREDICTOR_LIMIT = 30.0


@dataclass
class Fit:
    """A fitted model with its cluster-robust covariance."""

    coefficients: np.ndarray
    covariance: np.ndarray
    names: list[str]
    n_observations: int
    n_clusters: dict[str, int]
    converged: bool

    def index(self, name: str) -> int:
        return self.names.index(name)

    def contrast(self, weights: dict[str, float]) -> tuple[float, float]:
        """Point estimate and standard error for a linear combination."""
        vector = np.zeros(len(self.names))
        for name, weight in weights.items():
            vector[self.index(name)] = weight
        estimate = float(vector @ self.coefficients)
        variance = float(vector @ self.covariance @ vector)
        return estimate, float(np.sqrt(variance)) if variance > 0 else float("nan")


def _irls(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Newton-Raphson for the logistic likelihood, returning the bread matrix too."""
    beta = np.zeros(x.shape[1])
    converged = False
    for _ in range(MAX_ITERATIONS):
        eta = np.clip(x @ beta, -LINEAR_PREDICTOR_LIMIT, LINEAR_PREDICTOR_LIMIT)
        mu = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(mu * (1.0 - mu), 1e-10, None)
        gradient = x.T @ (y - mu)
        hessian = (x * weights[:, None]).T @ x
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return beta, np.full((x.shape[1], x.shape[1]), np.nan), False
        beta = beta + step
        if np.max(np.abs(step)) < TOLERANCE:
            converged = True
            break

    eta = np.clip(x @ beta, -LINEAR_PREDICTOR_LIMIT, LINEAR_PREDICTOR_LIMIT)
    mu = 1.0 / (1.0 + np.exp(-eta))
    weights = np.clip(mu * (1.0 - mu), 1e-10, None)
    try:
        bread = np.linalg.inv((x * weights[:, None]).T @ x)
    except np.linalg.LinAlgError:
        return beta, np.full((x.shape[1], x.shape[1]), np.nan), False
    return beta, bread, converged


def _meat(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Sum of outer products of within-cluster score totals."""
    order = np.argsort(groups, kind="mergesort")
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    total = np.zeros((scores.shape[1], scores.shape[1]))
    for block in np.split(scores[order], boundaries):
        aggregate = block.sum(axis=0)
        total += np.outer(aggregate, aggregate)
    return total


def fit(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    clusters: dict[str, np.ndarray] | None = None,
) -> Fit:
    """Fit a logistic model with cluster-robust covariance over one or two dimensions."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("design matrix and response disagree on the number of rows")
    if x.shape[1] != len(names):
        raise ValueError("design matrix and coefficient names disagree on width")

    rank = int(np.linalg.matrix_rank(x))
    if rank < x.shape[1]:
        raise ValueError(
            f"design matrix is rank deficient: rank {rank} over {x.shape[1]} columns. "
            "A term is collinear with another, or constant within this analysis set."
        )

    beta, bread, converged = _irls(x, y)
    if not np.all(np.isfinite(bread)):
        return Fit(
            coefficients=beta,
            covariance=np.full((x.shape[1], x.shape[1]), np.nan),
            names=list(names),
            n_observations=int(x.shape[0]),
            n_clusters={},
            converged=False,
        )

    eta = np.clip(x @ beta, -LINEAR_PREDICTOR_LIMIT, LINEAR_PREDICTOR_LIMIT)
    mu = 1.0 / (1.0 + np.exp(-eta))
    scores = (y - mu)[:, None] * x

    clusters = clusters or {}
    if not clusters:
        meat = scores.T @ scores
    elif len(clusters) == 1:
        meat = _meat(scores, next(iter(clusters.values())))
    elif len(clusters) == 2:
        first, second = list(clusters.values())
        # Cameron, Gelbach and Miller: add the one-way meats and subtract the
        # meat of their intersection, which would otherwise be counted twice.
        pair = np.char.add(first.astype(str), np.char.add("|", second.astype(str)))
        meat = _meat(scores, first) + _meat(scores, second) - _meat(scores, pair)
    else:
        raise ValueError("only one or two clustering dimensions are supported")

    covariance = bread @ meat @ bread

    # A two-way sandwich is not guaranteed positive semi-definite in finite
    # samples. Negative eigenvalues are clipped to zero, which is the standard
    # repair, and the fact that it was needed is not hidden: a variance of zero
    # on a contrast surfaces as a missing standard error rather than as a
    # confident estimate.
    if np.all(np.isfinite(covariance)):
        covariance = (covariance + covariance.T) / 2.0
        try:
            eigenvalues, vectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            covariance = np.full_like(covariance, np.nan)
        else:
            if np.any(eigenvalues < 0):
                covariance = vectors @ np.diag(np.clip(eigenvalues, 0.0, None)) @ vectors.T

    return Fit(
        coefficients=beta,
        covariance=covariance,
        names=list(names),
        n_observations=int(x.shape[0]),
        n_clusters={name: int(len(np.unique(values))) for name, values in clusters.items()},
        converged=converged,
    )
