"""Primary estimand 2: the slope of log R on onset.

The first estimand asks whether deep-time conservation and recent human
constraint lose discrimination at different rates as disease appears later. This
one asks a blunter question: how much do the evolutionary scores add over a
model that does not use them at all, and does that increment shrink with onset.

    R(d) = ( AUC_M1(d) - 0.5 ) / ( AUC_M0(d) - 0.5 )

M0 holds the non-evolutionary features and M1 adds the two evolutionary scores,
both fitted per disease and both scored out of fold. The estimand is the slope
of log R on onset.

The ratio, rather than the difference, is the point of the construction. Label
noise attenuates discrimination multiplicatively, and label noise is exactly
what varies along the onset axis: late-onset gene-disease assertions rest on
weaker evidence than Mendelian ones. A difference in AUROC would shrink with
onset for that reason alone and the estimand would be measuring annotation
quality. A ratio divides that attenuation out, because it hits both models. The
difference version is reported as a sensitivity, not as the estimand, so that
the two can be compared rather than conflated.

The controls are drawn at random rather than matched, and that is a departure
from the design the first estimand uses. Matching balances the confounders
exactly, and those confounders are part of M0, so within a matched set a
positive and its controls are near identical on everything M0 can see. A model
trained on the other folds then scores them alike, the held-out control
outranks its own positive about as often as not, and the pooled AUROC lands
below chance rather than at it. Measured on the generator, M0 scores 0.283
matched against 0.485 unmatched. Below-chance discrimination makes the lift
restriction unsatisfiable and the ratio meaningless, so the estimand is computed
against random controls, which the plan already names as a primary output
alongside the matched design rather than as a robustness pair.

Two restrictions travel with it. A disease whose non-evolutionary model barely
discriminates gives a ratio with a near-zero denominator and an unbounded value,
so diseases with AUC_M0 - 0.5 at or below 0.05 are excluded, as the plan
specifies. And a disease needs enough positives for a stratified five-fold split
to mean anything, so a floor on positives applies and is reported.

The slope is fitted with heteroskedasticity-robust standard errors. Each disease
contributes one point, so there is no clustering left to account for, but the
points are estimated with very different precision because diseases differ by an
order of magnitude in how many genes they implicate.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config
from src.diagnostics.power import rank_auroc
from src.features import evolutionary
from src.models import primary

# Below this, the non-evolutionary model is not discriminating and the ratio has
# no stable denominator. The plan sets the value.
MIN_BASELINE_LIFT = 0.05

# A stratified five-fold split needs at least one positive per fold, and a split
# that thin gives an AUROC with no useful precision.
MIN_POSITIVES = 10

FOLDS = 5

# The non-evolutionary comparator. The confounders are already in every design;
# the rest come from the baseline matrix, which is built from experimentally
# determined interactions, bulk expression and CRISPR essentiality, and
# deliberately excludes anything carrying comparative-genomics signal.
BASELINE_COLUMNS = (
    "cds_length_decile",
    "expected_lof_decile",
    "gc_content",
    "ppi_degree",
    "ppi_betweenness",
    "gtex_mean_expression",
    "expression_breadth_tau",
    "depmap_gene_effect",
)


def _standardise(matrix: np.ndarray) -> np.ndarray:
    centre = np.nanmean(matrix, axis=0)
    spread = np.nanstd(matrix, axis=0)
    spread[~np.isfinite(spread) | (spread == 0)] = 1.0
    return np.nan_to_num((matrix - centre) / spread, nan=0.0)


def cross_validated_auroc(x: np.ndarray, y: np.ndarray, seed: int, folds: int = FOLDS) -> float:
    """Out-of-fold AUROC of a logistic model, scored on pooled predictions.

    The predictions are pooled across folds and scored once rather than scored
    per fold and averaged. Per-fold AUROCs on a handful of positives are so
    noisy that their mean is dominated by the folds that happened to draw an
    easy split.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    positives = int(y.sum())
    if positives < folds or (len(y) - positives) < folds:
        return float("nan")

    predictions = np.full(len(y), np.nan)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(x, y):
        if len(np.unique(y[train])) < 2:
            return float("nan")
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(x[train], y[train])
        predictions[test] = model.predict_proba(x[test])[:, 1]

    if not np.all(np.isfinite(predictions)):
        return float("nan")
    return rank_auroc(predictions, y)


def per_disease(
    design: primary.Design,
    features: pd.DataFrame,
    seed: int,
    min_positives: int = MIN_POSITIVES,
) -> pd.DataFrame:
    """AUROC of the nested models for every disease with enough positives."""
    lookup = features.drop_duplicates("ensembl_gene_id").set_index("ensembl_gene_id")
    available = [column for column in BASELINE_COLUMNS if column in lookup.columns]
    if not available:
        raise ValueError("none of the non-evolutionary columns are in the feature matrix")

    baseline_values = np.column_stack(
        [pd.Series(design.gene).map(lookup[column]).to_numpy(dtype=float) for column in available]
    )
    conservation = design.x[:, design.names.index("z_conservation")]
    constraint = design.x[:, design.names.index("z_constraint")]

    rows = []
    for disease in pd.unique(design.disease):
        mask = design.disease == disease
        y = design.y[mask]
        if int(y.sum()) < min_positives:
            continue

        m0 = _standardise(baseline_values[mask])
        m1 = np.column_stack([m0, conservation[mask], constraint[mask]])

        auc_m0 = cross_validated_auroc(m0, y, seed)
        auc_m1 = cross_validated_auroc(m1, y, seed)
        rows.append(
            {
                "disease_id": disease,
                "n_positives": int(y.sum()),
                "n_rows": int(mask.sum()),
                "onset": float(design.onset[mask][0]),
                "auc_m0": auc_m0,
                "auc_m1": auc_m1,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table["lift_m0"] = table["auc_m0"] - 0.5
    table["lift_m1"] = table["auc_m1"] - 0.5
    table["usable"] = table["lift_m0"] > MIN_BASELINE_LIFT
    with np.errstate(invalid="ignore", divide="ignore"):
        table["ratio"] = np.where(table["usable"], table["lift_m1"] / table["lift_m0"], np.nan)
        table["log_ratio"] = np.where(table["ratio"] > 0, np.log(table["ratio"]), np.nan)
    table["difference"] = table["lift_m1"] - table["lift_m0"]
    return table


def slope(table: pd.DataFrame, column: str, alpha: float) -> dict:
    """Slope of a per-disease quantity on onset, with a robust interval."""
    usable = table[table[column].notna() & table["onset"].notna()]
    if len(usable) < 3:
        return {"n_diseases": len(usable), "note": "too few diseases to fit a slope"}

    onset = usable["onset"].to_numpy(dtype=float)
    values = usable[column].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(onset)), onset])

    xtx_inverse = np.linalg.pinv(x.T @ x)
    coefficients = xtx_inverse @ x.T @ values
    residuals = values - x @ coefficients

    # Each disease is one point estimated from a different number of genes, so
    # the residual variance is not constant across them. HC1 is the small-sample
    # correction to the sandwich; with a few hundred diseases the correction is
    # small but it costs nothing to apply it.
    n, k = len(onset), x.shape[1]
    meat = (x * (residuals**2)[:, None]).T @ x * (n / max(n - k, 1))
    covariance = xtx_inverse @ meat @ xtx_inverse

    estimate = float(coefficients[1])
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    critical = norm.ppf(1 - alpha / 2)
    return {
        "n_diseases": int(n),
        "intercept": float(coefficients[0]),
        "slope": estimate,
        "standard_error": standard_error,
        "ci_low": estimate - critical * standard_error,
        "ci_high": estimate + critical * standard_error,
        "excludes_zero": bool(
            (estimate - critical * standard_error) * (estimate + critical * standard_error) > 0
        ),
    }


def estimate(
    positives: pd.DataFrame,
    features: pd.DataFrame,
    seed: int,
    k: int,
    matched: bool = False,
) -> tuple[pd.DataFrame, dict]:
    design = primary.build_design(
        positives,
        features,
        evolutionary.CONSERVATION_PRIMARY,
        evolutionary.CONSTRAINT_PRIMARY,
        seed,
        k,
        matched,
    )
    table = per_disease(design, features, seed)
    alpha = config.analysis()["estimands"]["gamma"]["alpha"]

    summary = {
        "n_diseases_scored": int(len(table)),
        "n_diseases_usable": int(table["usable"].sum()) if not table.empty else 0,
        "min_baseline_lift": MIN_BASELINE_LIFT,
        "min_positives": MIN_POSITIVES,
        "matched": matched,
        "log_ratio_slope": slope(table, "log_ratio", alpha) if not table.empty else {},
        "difference_slope": slope(table, "difference", alpha) if not table.empty else {},
    }
    return table, summary


def run(
    evidence_source: str = "a_mendelian",
    positives: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    settings = config.analysis()
    seed = settings["seed"]
    k = settings["matching"]["k"]

    if positives is None:
        everything = pd.read_parquet(config.positives_path())
        positives = everything[everything["evidence_source"] == evidence_source]
    if features is None:
        from src.features import baseline

        features = evolutionary.load().merge(baseline.load(), on="ensembl_gene_id", how="left")

    table, summary = estimate(positives, features, seed, k)

    print(f"Estimand 2 over the {evidence_source} arm")
    print(f"   diseases with at least {MIN_POSITIVES} positives: {summary['n_diseases_scored']:,}")
    print(f"   of those, discriminating without evolution: {summary['n_diseases_usable']:,}")
    if not table.empty:
        print()
        print(table[["auc_m0", "auc_m1", "ratio", "difference"]].describe().to_string())
    print()
    print("   slope of log R on onset")
    for name, value in summary["log_ratio_slope"].items():
        print(f"      {name}: {value}")
    print()
    print("   slope of the AUROC difference on onset, reported as a sensitivity")
    for name, value in summary["difference_slope"].items():
        print(f"      {name}: {value}")

    out_dir = config.derived_dir() / "estimands"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"log_r_{evidence_source}.parquet", index=False)
    with (out_dir / f"log_r_{evidence_source}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)
        handle.write("\n")
    return table, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the log R slope on onset")
    parser.add_argument("--source", default="a_mendelian")
    args = parser.parse_args()
    config.ensure_dirs()
    run(evidence_source=args.source)


if __name__ == "__main__":
    main()
