"""Conditioning the onset contrast on pleiotropy and gene class.

One account of the relationship is that early-onset disease genes are developmental,
pleiotropic and haploinsufficient by definition of what causes severe early
disease. Constraint would then track gene class, which co-varies with onset for
reasons that have nothing to do with the fitness cost of the disease. Spence et
al. supply a published mechanism for the pleiotropy half of that: genome-wide
association studies prioritise genes near trait-specific variants and can
surface highly pleiotropic genes, while burden tests generally cannot, so
evidence source to pleiotropy to constraint is a real pathway.

Gamma is estimated under four specifications, each adding a block of covariates
and its interactions with both scores:

    base            the primary specification
    pleiotropy      plus trait breadth
    gene class      plus developmental role and haploinsufficiency
    both            plus everything

A second pleiotropy covariate, disease breadth, is built but not fitted. See
WITHDRAWN below.

The attenuation is reported as a proportion, with the qualification that
conditioning on a mediator removes part of the effect being estimated, and the
causal graph classifies expression breadth and network degree as ambiguous
between mediator and gene-class marker. Those are reported adjusted and
unadjusted and are never used in the primary matching.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config
from src.models import estimator, primary

# Each block is a set of covariates entered as a main effect and interacted with
# both scores, so the block is allowed to change how well each score
# discriminates rather than only the baseline odds.
BLOCKS = {
    "pleiotropy": ("pleiotropy_effective_traits",),
    "gene_class": ("developmental_share", "haploinsufficiency"),
}

# Disease breadth was specified as the second pleiotropy covariate and is not
# fitted. It counts the distinct therapeutic areas a gene carries evidence for,
# assembled in features/pleiotropy.py from the Open Targets evidence sets, the
# first of which is evidence_orphanet. Orphanet is where the Mendelian positives
# come from, so the covariate counts among a gene's therapeutic areas the
# association that makes it a positive here. That was deliberate when it was
# written, so that the count would match the rows the model treats as positives,
# and it is what makes the covariate unusable as an adjustment: a variable with
# the outcome among its inputs cannot be conditioned on, whichever way it moves
# the estimate.
#
# It is still built and still loaded, because the component split reports it.
WITHDRAWN = {
    "n_therapeutic_areas": (
        "built from the evidence sets that supply the positives, so it counts the outcome"
    ),
}


def load_blocks() -> pd.DataFrame:
    """Gene-level covariates for the mediation blocks, on one row per gene."""
    from src.features import gene_class, pleiotropy

    left = pleiotropy.load()[
        ["ensembl_gene_id", "pleiotropy_effective_traits", "n_therapeutic_areas"]
    ]
    right = gene_class.load()[["ensembl_gene_id", "developmental_share", "haploinsufficiency"]]
    return left.merge(right, on="ensembl_gene_id", how="outer").drop_duplicates("ensembl_gene_id")


def augment(design: primary.Design, covariates: pd.DataFrame, columns: list[str]) -> primary.Design:
    """Add a covariate block and its score interactions to an existing design.

    Rows whose covariates are missing are dropped rather than imputed, and the
    count that survives is carried in the metadata, because a specification
    fitted on a different subset is not comparable with one fitted on the whole.
    """
    lookup = covariates.set_index("ensembl_gene_id")
    frame = lookup.reindex(design.gene)

    values = {}
    for column in columns:
        if column not in frame.columns:
            continue
        values[column] = primary.standardise(frame[column])

    if not values:
        return design

    present = np.all([np.isfinite(v) for v in values.values()], axis=0)
    conservation = design.x[:, design.names.index("z_conservation")]
    constraint = design.x[:, design.names.index("z_constraint")]

    extra_names: list[str] = []
    extra_columns: list[np.ndarray] = []
    for column, series in values.items():
        extra_names += [column, f"conservation_x_{column}", f"constraint_x_{column}"]
        extra_columns += [series, conservation * series, constraint * series]

    matrix = np.column_stack([design.x, *extra_columns])[present]
    names = [*design.names, *extra_names]

    return primary.Design(
        x=matrix,
        y=design.y[present],
        names=names,
        disease=design.disease[present],
        gene=design.gene[present],
        onset=design.onset[present],
        metadata={**design.metadata, "n_rows": int(present.sum()), "block_columns": columns},
    )


def restrict(
    design: primary.Design, covariates: pd.DataFrame, columns: list[str]
) -> primary.Design:
    """Drop rows whose covariates are missing for any of the named columns.

    The components are compared against each other, so all of them have to be
    fitted on the same rows. Fitting each on whatever rows its own covariate
    happens to cover would confound the covariate with its coverage.
    """
    import numpy as np

    lookup = covariates.set_index("ensembl_gene_id")
    frame = lookup.reindex(design.gene)
    present = np.all([np.isfinite(primary.standardise(frame[c])) for c in columns], axis=0)
    return primary.Design(
        x=design.x[present],
        y=design.y[present],
        names=design.names,
        disease=design.disease[present],
        gene=design.gene[present],
        onset=design.onset[present],
        metadata={**design.metadata, "n_rows": int(present.sum())},
    )


def covariate_gradient(
    design: primary.Design,
    covariates: pd.DataFrame,
    columns: list[str],
    seed: int,
    draws: int = 400,
) -> pd.DataFrame:
    """Whether the adjustment covariates vary with onset at all.

    A covariate can confound the score-by-onset interaction only if it tracks
    onset. One that is flat along the axis cannot attenuate the contrast whatever
    its relationship with the scores, so a null movement from it would be a test
    that could not have failed rather than a result. This is the check that says
    which of the two applies, reported as a rank correlation with onset over the
    design rows and resampled over diseases.
    """
    from scipy.stats import spearmanr

    lookup = covariates.set_index("ensembl_gene_id")
    frame = lookup.reindex(design.gene)
    diseases = np.unique(design.disease)
    positions = {name: np.flatnonzero(design.disease == name) for name in diseases}
    rng = np.random.default_rng(seed)
    resamples = [
        np.concatenate([positions[name] for name in rng.choice(diseases, len(diseases), True)])
        for _ in range(draws)
    ]

    # Reported over the positives as well as over every row. The design carries
    # ten matched controls per positive, drawn without regard to these
    # covariates, so a gradient present among disease genes is diluted about
    # elevenfold across the whole design. That account is a claim about
    # disease genes, so the positives are the stratum it is made on.
    strata = {"all rows": np.ones_like(design.y, dtype=bool), "positives": design.y > 0}

    rows = []
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        finite = np.isfinite(values) & np.isfinite(design.onset)
        for stratum, mask in strata.items():
            keep = finite & mask
            rho = float(spearmanr(values[keep], design.onset[keep]).statistic)

            collected = []
            for index in resamples:
                taken = index[keep[index]]
                if taken.size > 100:
                    collected.append(float(spearmanr(values[taken], design.onset[taken]).statistic))
            array = np.array(collected)
            low, high = np.percentile(array, [2.5, 97.5]) if array.size > 20 else (np.nan, np.nan)
            rows.append(
                {
                    "covariate": column,
                    "stratum": stratum,
                    "n_rows": int(keep.sum()),
                    "spearman_with_onset": rho,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "varies_with_onset": bool(array.size > 20 and low * high > 0),
                }
            )
    return pd.DataFrame(rows)


def estimate(design: primary.Design, label: str) -> dict:
    """Gamma under one specification."""
    try:
        result = estimator.fit(
            design.x,
            design.y,
            design.names,
            clusters={"disease": design.disease, "gene": design.gene},
        )
    except ValueError as error:
        return {"specification": label, "gamma": float("nan"), "note": str(error)}

    gamma, standard_error = result.contrast(primary.GAMMA_CONTRAST)
    alpha = config.analysis()["estimands"]["gamma"]["alpha"]
    critical = norm.ppf(1 - alpha / 2)
    return {
        "specification": label,
        "n_rows": int(design.x.shape[0]),
        "n_positives": int(design.y.sum()),
        "n_terms": len(design.names),
        "gamma": gamma,
        "standard_error": standard_error,
        "ci_low": gamma - critical * standard_error,
        "ci_high": gamma + critical * standard_error,
        "conservation_x_onset": float(result.coefficients[result.index("conservation_x_onset")]),
        "constraint_x_onset": float(result.coefficients[result.index("constraint_x_onset")]),
        "converged": result.converged,
    }


def decomposition(rows: list[dict]) -> pd.DataFrame:
    """Attenuation of Gamma relative to the base specification."""
    table = pd.DataFrame(rows)
    base = table.loc[table["specification"] == "base", "gamma"]
    reference = float(base.iloc[0]) if len(base) and np.isfinite(base.iloc[0]) else np.nan

    if np.isfinite(reference) and reference != 0:
        table["attenuation"] = 1.0 - table["gamma"] / reference
    else:
        table["attenuation"] = np.nan

    table["excludes_zero"] = table["ci_low"] * table["ci_high"] > 0
    return table


def run(evidence_source: str = "a_mendelian") -> pd.DataFrame:
    settings = config.analysis()
    seed = settings["seed"]
    k = settings["matching"]["k"]

    from src.features import evolutionary

    positives = pd.read_parquet(config.positives_path())
    subset = positives[positives["evidence_source"] == evidence_source]
    features = evolutionary.load()

    base = primary.build_design(
        subset,
        features,
        evolutionary.CONSERVATION_PRIMARY,
        evolutionary.CONSTRAINT_PRIMARY,
        seed,
        k,
        matched=True,
    )
    covariates = load_blocks()

    specifications = {
        "base": [],
        "pleiotropy": list(BLOCKS["pleiotropy"]),
        "gene_class": list(BLOCKS["gene_class"]),
        "both": list(BLOCKS["pleiotropy"]) + list(BLOCKS["gene_class"]),
    }

    # The covariates do not cover every gene, so a specification that enters them
    # is fitted on fewer rows than one that does not. Comparing across that gap
    # charges the change of sample to the conditioning, and the rows it drops are
    # not a random subset: a gene with no biological-process annotation is an
    # understudied gene, and understudied genes are rarely disease genes. Every
    # specification is therefore fitted on the rows they all share, and the
    # contrast on all rows is reported separately so the size of the gap is
    # visible rather than absorbed.
    columns_used = sorted({column for columns in specifications.values() for column in columns})
    shared = restrict(base, covariates, columns_used)

    rows = [estimate(base, "base_all_rows")]
    for label, columns in specifications.items():
        design = augment(shared, covariates, columns) if columns else shared
        rows.append(estimate(design, label))

    table = decomposition(rows)

    # The withdrawn covariate is included here. Whether it tracks onset is part
    # of the evidence for withdrawing it, so it is measured even though it is
    # never fitted.
    every = sorted(
        {column for columns in specifications.values() for column in columns} | set(WITHDRAWN)
    )
    gradient = covariate_gradient(base, covariates, every, seed)

    out_dir = config.derived_dir() / "mediation"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"mediation_{evidence_source}.parquet", index=False)
    gradient.to_parquet(out_dir / f"covariate_gradient_{evidence_source}.parquet", index=False)
    with (out_dir / f"mediation_{evidence_source}.json").open("w", encoding="utf-8") as handle:
        json.dump(table.to_dict(orient="records"), handle, indent=2, default=float)
        handle.write("\n")

    print(f"Mediation over the {evidence_source} arm")
    print(table.to_string(index=False))
    print()
    print("Do the adjustment covariates track onset?")
    print(gradient.round(4).to_string(index=False))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the mediation specifications")
    parser.add_argument("--source", default="a_mendelian")
    args = parser.parse_args()
    config.ensure_dirs()
    run(evidence_source=args.source)


if __name__ == "__main__":
    main()
