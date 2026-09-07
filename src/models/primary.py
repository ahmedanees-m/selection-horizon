"""The primary model and the estimand Gamma.

One stage, not two. For gene g and disease d, y is 1 if g is a positive for d:

    logit P(y = 1) = alpha
        + sum_s ( beta_s z[s,g] + gamma_s z[s,g] x onset[d] )
        + score by inheritance interactions
        + theta' x[g]
        + evidence source
        + (1 | disease) + (1 | gene)

and

    Gamma = decay_conservation - decay_constraint

where the decay of a score is how much discriminative power it *loses* across
the onset range, so it is the negative of that score's interaction coefficient:

    decay_s = -gamma_s          gamma_s is the coefficient on z[s,g] x onset

which makes

    Gamma = constraint_x_onset - conservation_x_onset

Gamma above zero means deep-time conservation loses discriminative power faster
than recent human constraint as onset rises. Writing the contrast directly on
the interaction coefficients without that negation reverses its sign, which a
regression test checks.

Three things about the specification are decisions rather than defaults.

The clustering is handled by a two-way cluster-robust sandwich over disease and
gene rather than by fitting cross-classified random effects. That is what the
power simulation was calibrated on, so the minimum detectable effect of 0.035
applies to this fit as stated rather than approximately.

Inheritance mode enters as a score interaction rather than as a sample split.
Splitting to autosomal dominant, where s_het is interpretable, costs two thirds
of the sample and pushes the minimum detectable effect from 0.035 to 0.052. The
interaction keeps every observation and identifies the onset effect within mode.

Evidence sources are never pooled. Gamma is estimated inside each source and
reported per source, because label quality and pleiotropy both differ across
them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config
from src.features import evolutionary, matching
from src.onset import inheritance, tiers

# Gamma is a contrast of decays, and a decay is the negative of an interaction
# coefficient. See the module docstring; the sign matters and has been wrong once.
GAMMA_CONTRAST = {"constraint_x_onset": 1.0, "conservation_x_onset": -1.0}


@dataclass
class Design:
    """A model-ready design matrix with its clustering keys."""

    x: np.ndarray
    y: np.ndarray
    names: list[str]
    disease: np.ndarray
    gene: np.ndarray
    onset: np.ndarray
    metadata: dict = field(default_factory=dict)


def standardise(values: pd.Series) -> np.ndarray:
    """Z-score within the analysis set, which is where the plan defines it."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    centre = np.nanmean(numeric)
    spread = np.nanstd(numeric)
    if not np.isfinite(spread) or spread == 0:
        return np.zeros_like(numeric)
    return (numeric - centre) / spread


# Which column carries the onset axis, by evidence source. The Mendelian arm is
# on Orphanet intervals and the three Platform arms are on a median age in
# years binned by decade, and the two are not the same column. Reading the
# Mendelian column for a Platform arm silently drops four fifths of it and puts
# what remains on an axis it was not measured against, so the choice is made
# from the evidence source rather than defaulted.
ONSET_COLUMN = {
    "a_mendelian": "onset_bin",
    "b_gwas": "onset_decade",
    "c_clinvar": "onset_decade",
    "d_gene_burden": "onset_decade",
}


# The crosstab records how many diseases each arm has an onset
# for. A design that keeps far fewer than that has lost them somewhere, and the
# most recent way it happened was reading the wrong onset column, which produced
# a design with a fifth of the complex arm and no warning at all. The floor is
# generous because a design legitimately loses diseases whose genes carry no
# score; it is set to catch a collapse, not a trim.
DESIGN_RETENTION_FLOOR = 0.5

# Numeric onset carried per disease by the span-value encoding.
ONSET_VALUE_COLUMN = "onset_span_value"

CROSSTAB_FOR_COLUMN = {
    "onset_bin": "crosstab.parquet",
    "onset_decade": "crosstab_complex.parquet",
}


def expected_diseases(evidence_source: str, onset_column: str) -> int | None:
    """How many diseases the crosstab records this arm as having an onset for."""
    name = CROSSTAB_FOR_COLUMN.get(onset_column)
    if name is None:
        return None
    path = config.derived_dir() / "analysis_set" / name
    if not path.exists():
        return None

    table = pd.read_parquet(path)
    rows = table[
        (table["evidence_source"] == evidence_source) & (table["onset_bin"] != "unassigned")
    ]
    return int(rows["n_diseases"].sum()) if not rows.empty else None


def check_design_against_crosstab(positives: pd.DataFrame, onset_column: str, kept: int) -> None:
    """Refuse a design that has lost most of the arm it was asked to build.

    Checked against an independent record rather than against itself. Both onset
    columns hold valid labels, so reading the wrong one produces a smaller design
    that is correct in every local respect and wrong about which diseases it
    describes. Nothing internal to the design can notice that; the crosstab can.
    """
    sources = {str(value) for value in positives["evidence_source"].dropna().unique()}
    if len(sources) != 1:
        return
    expected = expected_diseases(sources.pop(), onset_column)
    if not expected:
        return

    if kept < DESIGN_RETENTION_FLOOR * expected:
        raise ValueError(
            f"the design kept {kept} diseases against {expected} in the crosstab, "
            f"below the {DESIGN_RETENTION_FLOOR:.0%} floor. The most likely cause is the "
            f"onset column: this design read {onset_column!r}."
        )


def onset_column_for(positives: pd.DataFrame) -> str:
    """The onset column for a set of positives, from the arm they belong to."""
    if "evidence_source" not in positives.columns:
        return "onset_bin"
    sources = {str(value) for value in positives["evidence_source"].dropna().unique()}
    columns = {ONSET_COLUMN.get(source, "onset_bin") for source in sources}
    if len(columns) > 1:
        raise ValueError(
            f"these positives mix onset axes across arms {sorted(sources)}; "
            "each arm is fitted separately and never pooled"
        )
    return columns.pop() if columns else "onset_bin"


def onset_position(bins: pd.Series) -> np.ndarray:
    """Onset scaled to [0, 1] so that Gamma is per full onset range.

    Both vocabularies are handled. The Mendelian tiers are named intervals with
    an order the configuration fixes, and the complex tiers are decade ranges
    whose lower bound orders them. Scaling both onto the same unit interval is
    what makes a Gamma from one arm the same quantity as a Gamma from the other.
    """
    order = tiers.collapsed_order()
    named = {name: index / max(len(order) - 1, 1) for index, name in enumerate(order)}

    values = [str(value) for value in bins]
    lower_bounds = sorted(
        {int(text.split("-")[0]) for text in values if text.split("-")[0].isdigit()}
    )
    spread = max(len(lower_bounds) - 1, 1)
    by_decade = {bound: index / spread for index, bound in enumerate(lower_bounds)}

    def position(text: str) -> float:
        if text in named:
            return named[text]
        head = text.split("-")[0]
        return by_decade.get(int(head), float("nan")) if head.isdigit() else float("nan")

    return np.array([position(text) for text in values], dtype=float)


def scale_to_unit(values: np.ndarray) -> np.ndarray:
    """Put an onset encoding on [0, 1] so Gamma is per full onset range.

    Every encoding passes through here. An encoding measured in years and left
    unscaled would return a Gamma smaller by the width of the axis, which reads
    as a precise null rather than as a different unit.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    low, high = finite.min(), finite.max()
    return (values - low) / (high - low) if high > low else np.zeros_like(values)


def onset_midpoint(bins: pd.Series) -> np.ndarray:
    """Onset as an age in years, scaled to [0, 1].

    The level encoding spaces the onset tiers evenly, which treats the step from
    perinatal to infancy as the same distance as the step from adolescent to
    adult. In years those are half a year and thirty. This encoding uses the
    midpoint age of each tier instead, so a robustness arm can ask whether the
    contrast is a property of the ordering or of the spacing.
    """
    midpoints = dict(tiers.collapse_spec()["midpoints"])

    def age(value) -> float:
        text = str(value)
        if text in midpoints:
            return float(midpoints[text])
        parts = text.split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return (float(parts[0]) + float(parts[1])) / 2.0
        return float("nan")

    return scale_to_unit(np.array([age(value) for value in bins], dtype=float))


def build_design(
    positives: pd.DataFrame,
    features: pd.DataFrame,
    conservation_score: str,
    constraint_score: str,
    seed: int,
    k: int,
    matched: bool = True,
    adjust_genes_per_disease: bool = True,
    three_way_inheritance: bool = False,
    onset_encoding: str = "level",
) -> Design:
    """Assemble one evidence source into a design matrix.

    Positives are the observed gene-disease pairs. Controls are either matched
    on the confounders within disease, or the full complement of scored genes
    when the unmatched arm is requested. Both are primary outputs; they bound
    the answer from opposite directions.
    """
    genes = features.rename(columns={"ensembl_gene_id": "gene_id"})
    scored = genes.dropna(subset=[conservation_score, constraint_score])

    onset_column = onset_column_for(positives)
    # The span-value encoding carries a numeric onset for every disease, which is
    # what lets the registered arm re-admit diseases that have no ordinal level.
    value_column = ONSET_VALUE_COLUMN if onset_encoding == "span_value" else onset_column
    if value_column not in positives.columns:
        raise ValueError(f"{value_column} is not on the positives frame")
    positives = positives[positives["gene_id"].isin(set(scored["gene_id"]))]
    positives = positives.dropna(subset=[value_column])
    if positives.empty:
        raise ValueError("no positives survive the score and onset requirements")

    draw = matching.draw_controls if matched else matching.draw_random_controls
    drawn = draw(positives, features, k=k, seed=seed)
    carried = ["disease_id", onset_column, "inheritance_mode"]
    if value_column not in carried:
        carried.append(value_column)
    frame = drawn.merge(
        positives[carried].drop_duplicates("disease_id"),
        on="disease_id",
        how="left",
    )

    frame = frame.merge(scored, on="gene_id", how="inner").dropna(subset=[value_column])
    if frame.empty:
        raise ValueError("the design is empty after joining features")

    if onset_encoding == "span_midpoint":
        onset = onset_midpoint(frame[onset_column])
    elif onset_encoding == "span_value":
        onset = scale_to_unit(frame[value_column].to_numpy(dtype=float))
    elif onset_encoding == "level":
        onset = onset_position(frame[onset_column])
    else:
        raise ValueError(f"unknown onset encoding {onset_encoding!r}")
    keep = np.isfinite(onset)
    frame, onset = frame.loc[keep].reset_index(drop=True), onset[keep]
    centred_onset = onset - onset.mean()

    z_conservation = standardise(frame[conservation_score])
    z_constraint = standardise(frame[constraint_score])

    # How many genes a disease implicates rises with onset across the complex
    # arm: a median of 3 in the first decade against 23.5 past seventy, because
    # later-onset traits are more polygenic and their credible sets are larger.
    # Left out, that difference in labelling density loads onto the onset term.
    genes_per_disease = frame.groupby("disease_id")["gene_id"].transform("nunique")

    columns = {
        "intercept": np.ones(len(frame)),
        "z_conservation": z_conservation,
        "z_constraint": z_constraint,
        "onset": centred_onset,
        "conservation_x_onset": z_conservation * centred_onset,
        "constraint_x_onset": z_constraint * centred_onset,
        "cds_length_decile": standardise(frame["cds_length_decile"]),
        "expected_lof_decile": standardise(frame["expected_lof_decile"]),
        "gc_content": standardise(frame["gc_content"]),
    }

    # The adjustment for labelling density is reported as a pair rather than
    # applied silently. It removes little of the onset signal, since the
    # disease-level correlation between the two is 0.13 on the log scale, but
    # the unadjusted estimate is reported next to the adjusted one so that a
    # reader can see that rather than take it on trust.
    if adjust_genes_per_disease:
        columns["log_genes_per_disease"] = standardise(np.log(genes_per_disease.clip(lower=1)))

    # Inheritance mode is a confounder of the onset contrast and enters as a
    # stratifying interaction with each score. Three things about it are fixed
    # here rather than left to be read off the code.
    #
    # It is a property of the disease, not of the gene. Orphanet reports it per
    # disease and a gene that causes several diseases inherits whichever label
    # each of those diseases carries, so the same gene can appear under more
    # than one mode across rows. That is a consequence of the assignment being
    # at disease level and not a gene carrying two modes at once.
    #
    # A disease reported under several modes is labelled mixed rather than
    # forced into one, and mixed is not treated as either.
    #
    # Dominant and recessive each get their own indicator, with mixed and
    # unknown together as the reference. Folding recessive in with unknown, as
    # a single dominant indicator would, dilutes exactly the distinction that
    # motivates the stratifier: s_het is a coefficient on heterozygotes and is
    # close to blind to recessive genes by construction, while a disease of
    # unknown mode says nothing either way.
    #
    # The terms enter only where they vary. The Platform arms carry no
    # inheritance label, and an all-zero column would make the design rank
    # deficient rather than merely uninformative.
    indicators = {
        "dominant": (frame["inheritance_mode"] == inheritance.DOMINANT).to_numpy(dtype=float),
        "recessive": (frame["inheritance_mode"] == inheritance.RECESSIVE).to_numpy(dtype=float),
    }
    modes_present = [name for name, values in indicators.items() if 0 < values.sum() < len(values)]

    # One category has to be held out or the indicators sum to one and collide
    # with the intercept. Where rows of mixed or unknown mode exist they are the
    # reference, which is what makes the dominant and recessive coefficients
    # each a contrast against an uninformative baseline. Where every row is one
    # or the other, there is no such baseline and recessive becomes the
    # reference, so the dominant coefficient is dominant against recessive. The
    # design records which it was, because the two are not the same contrast.
    unclassified = 1.0 - sum(indicators[name] for name in modes_present)
    reference = "mixed or unknown"
    if len(modes_present) > 1 and not (unclassified > 0).any():
        reference = modes_present.pop()
    for name in modes_present:
        values = indicators[name]
        columns[name] = values
        columns[f"conservation_x_{name}"] = z_conservation * values
        columns[f"constraint_x_{name}"] = z_constraint * values

    # The three-way interaction lets the onset slope itself differ by mode. The
    # two-way above assumes a common slope within mode and returns a weighted
    # average if that assumption is wrong, so the three-way is fitted as a
    # robustness arm rather than as the primary specification.
    if three_way_inheritance:
        for name in modes_present:
            values = indicators[name]
            columns[f"onset_x_{name}"] = centred_onset * values
            columns[f"conservation_x_onset_x_{name}"] = z_conservation * centred_onset * values
            columns[f"constraint_x_onset_x_{name}"] = z_constraint * centred_onset * values

    has_inheritance_variation = bool(modes_present)

    names = list(columns)
    matrix = np.column_stack([columns[name] for name in names])
    finite = np.all(np.isfinite(matrix), axis=1)

    check_design_against_crosstab(
        positives, onset_column, int(frame.loc[finite, "disease_id"].nunique())
    )

    return Design(
        x=matrix[finite],
        y=frame.loc[finite, "y"].to_numpy(dtype=float),
        names=names,
        disease=frame.loc[finite, "disease_id"].to_numpy(),
        gene=frame.loc[finite, "gene_id"].to_numpy(),
        onset=onset[finite],
        metadata={
            "onset_column": onset_column,
            "onset_encoding": onset_encoding,
            "adjust_genes_per_disease": adjust_genes_per_disease,
            "three_way_inheritance": three_way_inheritance,
            "inheritance_modes": modes_present,
            "inheritance_reference": reference if modes_present else None,
            "conservation_score": conservation_score,
            "constraint_score": constraint_score,
            "matched": matched,
            "k": k if matched else None,
            "n_positives": int(frame.loc[finite, "y"].sum()),
            "n_rows": int(finite.sum()),
            "n_diseases": int(frame.loc[finite, "disease_id"].nunique()),
            "n_genes": int(frame.loc[finite, "gene_id"].nunique()),
            "inheritance_terms": bool(has_inheritance_variation),
        },
    )


def estimate_gamma(design: Design) -> dict:
    """Fit the model and return Gamma with its cluster-robust interval."""
    from src.models import estimator

    result = estimator.fit(
        design.x,
        design.y,
        design.names,
        clusters={"disease": design.disease, "gene": design.gene},
    )
    gamma, standard_error = result.contrast(GAMMA_CONTRAST)

    alpha = config.analysis()["estimands"]["gamma"]["alpha"]
    critical = norm.ppf(1 - alpha / 2)
    return {
        **design.metadata,
        "gamma": gamma,
        "standard_error": standard_error,
        "ci_low": gamma - critical * standard_error,
        "ci_high": gamma + critical * standard_error,
        "alpha": alpha,
        "converged": result.converged,
        "n_clusters": result.n_clusters,
        "conservation_main": float(result.coefficients[result.index("z_conservation")]),
        "constraint_main": float(result.coefficients[result.index("z_constraint")]),
        "conservation_x_onset": float(result.coefficients[result.index("conservation_x_onset")]),
        "constraint_x_onset": float(result.coefficients[result.index("constraint_x_onset")]),
    }


def permutation_null(design: Design, draws: int, seed: int) -> np.ndarray:
    """Gamma under onset permuted across diseases within the design.

    Onset is a disease-level attribute, so it is permuted over diseases and
    broadcast back to their rows. Permuting row-wise would break the clustering
    and produce a null that is too narrow.
    """
    from src.models import estimator

    rng = np.random.default_rng(seed)
    disease_onset = (
        pd.DataFrame({"disease": design.disease, "onset": design.onset})
        .drop_duplicates("disease")
        .set_index("disease")["onset"]
    )

    conservation = design.x[:, design.names.index("z_conservation")]
    constraint = design.x[:, design.names.index("z_constraint")]
    onset_column = design.names.index("onset")
    conservation_column = design.names.index("conservation_x_onset")
    constraint_column = design.names.index("constraint_x_onset")

    values = np.empty(draws)
    for draw in range(draws):
        shuffled = pd.Series(rng.permutation(disease_onset.to_numpy()), index=disease_onset.index)
        onset = pd.Series(design.disease).map(shuffled).to_numpy(dtype=float)
        centred = onset - onset.mean()

        matrix = design.x.copy()
        matrix[:, onset_column] = centred
        matrix[:, conservation_column] = conservation * centred
        matrix[:, constraint_column] = constraint * centred

        result = estimator.fit(
            matrix,
            design.y,
            design.names,
            clusters={"disease": design.disease, "gene": design.gene},
        )
        values[draw], _ = result.contrast(GAMMA_CONTRAST)
    return values


def run(draws: int = 0) -> pd.DataFrame:
    settings = config.analysis()
    seed = settings["seed"]
    k = settings["matching"]["k"]

    out_dir = config.derived_dir() / "estimands"
    out_dir.mkdir(parents=True, exist_ok=True)

    positives = pd.read_parquet(config.positives_path())
    features = evolutionary.load()

    conservation_score = evolutionary.CONSERVATION_PRIMARY
    constraint_score = evolutionary.CONSTRAINT_PRIMARY

    rows = []
    for source in sorted(positives["evidence_source"].unique()):
        subset = positives[positives["evidence_source"] == source]
        # Four primary outputs per arm, not one. Matched against unmatched bounds
        # the confounding from opposite directions, and adjusted against
        # unadjusted shows what the labelling-density adjustment does rather
        # than asserting that it does little. All four are reported.
        for matched in (True, False):
            for adjusted in (True, False):
                try:
                    design = build_design(
                        subset,
                        features,
                        conservation_score,
                        constraint_score,
                        seed,
                        k,
                        matched,
                        adjust_genes_per_disease=adjusted,
                    )
                except ValueError as error:
                    print(f"{source} ({'matched' if matched else 'unmatched'}): {error}")
                    continue
                result = estimate_gamma(design)
                result["evidence_source"] = source
                rows.append(result)
                print(
                    f"{source:<16} {'matched' if matched else 'unmatched':<10} "
                    f"{'adjusted' if adjusted else 'unadjusted':<11} "
                    f"rows={result['n_rows']:>8,} positives={result['n_positives']:>6,} "
                    f"gamma={result['gamma']:+.4f} "
                    f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]"
                )

            if draws and matched and source == "a_mendelian":
                design = build_design(
                    subset, features, conservation_score, constraint_score, seed, k, True
                )
                null = permutation_null(design, draws, seed)
                np.save(out_dir / f"permutation_null_{source}.npy", null)
                print(
                    f"   permutation null over {draws} draws: "
                    f"mean {null.mean():+.4f}, sd {null.std():.4f}, "
                    f"two-sided p = {float(np.mean(np.abs(null) >= abs(result['gamma']))):.4f}"
                )

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "gamma_estimates.parquet", index=False)
    with (out_dir / "gamma_estimates.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, default=float)
        handle.write("\n")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the primary model and estimate Gamma")
    parser.add_argument(
        "--permutations", type=int, default=0, help="permutation draws for the placebo null"
    )
    args = parser.parse_args()
    config.ensure_dirs()
    run(draws=args.permutations)


if __name__ == "__main__":
    main()
