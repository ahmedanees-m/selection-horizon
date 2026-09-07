"""Control analyses on the onset axis.

Four controls, each fitted on the control score alone with the evolutionary
scores dropped from the design.

    gene length                 negative    expects a flat onset slope
    tissue specificity          negative    expects no decay
    developmental annotation    positive    expects a steep decay
    permuted onset              placebo     expects a null centred on zero

A gene property unrelated to selection should not decline with disease onset,
which is what the two negative controls measure. Genes annotated to the
developmental branch of the Gene Ontology are expected to lose their association
with disease as onset rises, which is what the positive control measures.
Developmentally annotated genes are also pleiotropic and haploinsufficient, so
the positive control speaks to whether the onset axis resolves gene class rather
than fitness cost; that separation is the subject of the mediation.

The placebo permutes onset over diseases rather than over rows, and within
therapeutic area, so the permutation preserves the association between disease
class and onset and removes only the association between onset and the scores.

Each control is estimated as a decay, the loss of a score's discrimination per
unit of onset, which is the negative of that score's interaction with onset. A
positive value means the score loses discriminative power as disease appears
later.

The evolutionary scores are dropped rather than adjusted for, because a control
conditioned on the scores it is meant to be independent of would have its
gradient partialled away.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config
from src.features import evolutionary
from src.models import primary

# A decay is the negative of an interaction coefficient, so that a positive
# number means the score loses discrimination as onset rises. This is the same
# convention primary.GAMMA_CONTRAST uses.
DECAY_CONTRAST = {"control_x_onset": -1.0}

# A placebo null whose centre sits within a tenth of its own spread of zero is
# centred for the purpose of the threshold. See summarise_null.
CENTRED_TOLERANCE = 0.1


@dataclass
class Control:
    """One control: which score stands in, and what has to be true of it."""

    name: str
    kind: str
    score: str
    expectation: str
    rationale: str
    # A covariate the control score would be collinear with. Gene length is
    # already an adjustment covariate as a decile, and leaving it in while
    # testing gene length itself would make the design rank deficient.
    collinear_with: tuple[str, ...] = ()


def controls() -> list[Control]:
    return [
        Control(
            "gene_length",
            "negative",
            "cds_length",
            "flat",
            "Coding length is adjusted for as a decile and should carry no onset "
            "gradient of its own. A gradient here means the adjustment is not working",
            collinear_with=("cds_length_decile",),
        ),
        Control(
            "tissue_specificity",
            "negative",
            "expression_breadth_tau",
            "flat",
            "How narrowly a gene is expressed is a real property measured "
            "independently of any evolutionary score, and nothing should make it "
            "decline as disease appears later",
        ),
        Control(
            "developmental_annotation",
            "positive",
            "developmental_share",
            "steep decay",
            "Genes whose annotated function is building the organism cause disease "
            "early. If that association does not fall with onset, the onset axis "
            "has not resolved and nothing estimated along it can be read. Passing "
            "establishes that the axis resolves gene class, which is the confound "
            "the mediation conditions away, and not that it resolves fitness cost",
        ),
    ]


NOT_RUN = {
    "affected_tissue_conditioning": (
        "the plan names the negative control as tissue-specific expression in the "
        "affected tissue, which needs a disease to tissue mapping that none of the "
        "sources carry; the control runs on tissue specificity without conditioning "
        "on tissue and is weaker for it"
    ),
    "developmental_expression": (
        "the positive control is specified on fetal expression. The developmental "
        "series is deposited as alignments rather than as a joinable matrix, so the "
        "developmental branch of the Gene Ontology stands in; see src/features/gene_class"
    ),
}


def control_features(features: pd.DataFrame | None = None) -> pd.DataFrame:
    """The evolutionary matrix widened with the columns the controls need."""
    from src.features import baseline, gene_class

    table = evolutionary.load() if features is None else features
    for source, columns in (
        (baseline.load(), ["ensembl_gene_id", "expression_breadth_tau"]),
        (gene_class.load(), ["ensembl_gene_id", "developmental_share"]),
    ):
        available = [column for column in columns if column in source.columns]
        if len(available) > 1:
            table = table.merge(source[available], on="ensembl_gene_id", how="left")
    return table


def control_design(design: primary.Design, values: np.ndarray, drop: tuple[str, ...]) -> dict:
    """Recast a fitted design around one control score.

    The rows, the outcome, the clustering and the confounders are the design the
    primary estimand uses, so the control is answered on the same analysis set.
    What changes is the score: the two evolutionary columns and their onset
    interactions come out, and the control score and its interaction go in.
    """
    keep = [
        name
        for name in design.names
        if name
        not in {
            "z_conservation",
            "z_constraint",
            "conservation_x_onset",
            "constraint_x_onset",
            "conservation_x_dominant",
            "constraint_x_dominant",
            *drop,
        }
    ]
    matrix = design.x[:, [design.names.index(name) for name in keep]]

    centred = design.onset - design.onset.mean()
    z_control = primary.standardise(pd.Series(values))
    matrix = np.column_stack([matrix, z_control, z_control * centred])
    names = [*keep, "z_control", "control_x_onset"]

    finite = np.all(np.isfinite(matrix), axis=1)
    return {
        "x": matrix[finite],
        "y": design.y[finite],
        "names": names,
        "disease": design.disease[finite],
        "gene": design.gene[finite],
        "n_rows": int(finite.sum()),
        "n_dropped": int((~finite).sum()),
    }


def run_control(
    positives: pd.DataFrame,
    features: pd.DataFrame,
    control: Control,
    seed: int,
    k: int,
    matched: bool = True,
) -> dict:
    from src.models import estimator

    if control.score not in features.columns:
        return {
            "control": control.name,
            "kind": control.kind,
            "expectation": control.expectation,
            "decay": float("nan"),
            "note": f"{control.score} is not in the feature matrix",
        }

    design = primary.build_design(
        positives,
        features,
        evolutionary.CONSERVATION_PRIMARY,
        evolutionary.CONSTRAINT_PRIMARY,
        seed,
        k,
        matched,
    )

    lookup = features.drop_duplicates("ensembl_gene_id").set_index("ensembl_gene_id")[control.score]
    values = pd.Series(design.gene).map(lookup).to_numpy(dtype=float)

    recast = control_design(design, values, control.collinear_with)
    if recast["n_rows"] < 100:
        return {
            "control": control.name,
            "kind": control.kind,
            "expectation": control.expectation,
            "decay": float("nan"),
            "note": f"only {recast['n_rows']} rows carry {control.score}",
        }

    result = estimator.fit(
        recast["x"],
        recast["y"],
        recast["names"],
        clusters={"disease": recast["disease"], "gene": recast["gene"]},
    )
    decay, standard_error = result.contrast(DECAY_CONTRAST)

    alpha = config.analysis()["estimands"]["gamma"]["alpha"]
    critical = norm.ppf(1 - alpha / 2)
    low, high = decay - critical * standard_error, decay + critical * standard_error

    return {
        "control": control.name,
        "kind": control.kind,
        "score": control.score,
        "expectation": control.expectation,
        "rationale": control.rationale,
        "n_rows": recast["n_rows"],
        "n_dropped_for_missing_score": recast["n_dropped"],
        "main_effect": float(result.coefficients[result.index("z_control")]),
        "decay": float(decay),
        "standard_error": float(standard_error),
        "ci_low": float(low),
        "ci_high": float(high),
        "excludes_zero": bool(low * high > 0),
        "note": "",
    }


def as_expected(row: dict) -> bool:
    """Whether the control's decay matches the expectation stated for it."""
    if not np.isfinite(row.get("decay", float("nan"))):
        return False
    excludes_zero = bool(row.get("excludes_zero"))
    if row["kind"] == "negative":
        return not excludes_zero
    if row["kind"] == "positive":
        return excludes_zero and row["decay"] > 0
    return False


def therapy_areas() -> pd.Series:
    """One therapeutic area per disease, for the stratified permutation.

    A disease can sit in several areas. The first is taken, deterministically
    after sorting, because the stratum only has to be a stable disease class and
    a disease that belongs to two areas does not belong to a third one made of
    both.
    """
    from src.ingest import open_targets

    disease = open_targets.read("disease", columns=["id", "therapeuticAreas"])

    def first(value) -> str:
        if value is None:
            return "unassigned"
        items = sorted(str(item) for item in value if item is not None)
        return items[0] if items else "unassigned"

    return pd.Series(
        [first(value) for value in disease["therapeuticAreas"]],
        index=disease["id"].to_numpy(),
    )


def stratified_permutation_null(
    design: primary.Design, strata: pd.Series, draws: int, seed: int
) -> np.ndarray:
    """Gamma under onset permuted across diseases within therapeutic area.

    Permuting across all diseases would also break the association between
    disease class and onset, so a null built that way answers a question nobody
    asked. Permuting within area leaves that association intact and destroys
    only the one under test.
    """
    from src.models import estimator

    rng = np.random.default_rng(seed)
    by_disease = (
        pd.DataFrame({"disease": design.disease, "onset": design.onset})
        .drop_duplicates("disease")
        .set_index("disease")["onset"]
    )
    stratum = pd.Series(by_disease.index, index=by_disease.index).map(strata).fillna("unassigned")

    conservation = design.x[:, design.names.index("z_conservation")]
    constraint = design.x[:, design.names.index("z_constraint")]
    onset_column = design.names.index("onset")
    conservation_column = design.names.index("conservation_x_onset")
    constraint_column = design.names.index("constraint_x_onset")

    values = np.empty(draws)
    for draw in range(draws):
        shuffled = by_disease.copy()
        for _, index in stratum.groupby(stratum).groups.items():
            shuffled.loc[index] = rng.permutation(by_disease.loc[index].to_numpy())

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
        values[draw], _ = result.contrast(primary.GAMMA_CONTRAST)
    return values


def summarise_null(values: np.ndarray) -> dict:
    """Whether the placebo null is centred, which the threshold requires."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return {"draws": int(finite.size), "note": "too few draws to summarise"}

    centre = float(np.mean(finite))
    spread = float(np.std(finite, ddof=1))
    if spread <= 0:
        return {"draws": int(finite.size), "note": "the null has no spread"}

    # The offset is measured against the null's own spread rather than against
    # the standard error of its mean. Testing the mean against its standard
    # error would make the criterion tighter with every extra draw, so a null
    # displaced by a thousandth of the effect size would fall below the threshold at ten
    # thousand draws while passing it at a hundred. That is a property of the
    # draw count and not of the model. An offset below a tenth of the null's own
    # spread is negligible at any number of draws, and the significance of the
    # offset is reported separately so that a small but real displacement is
    # visible without stopping the project over it.
    error = spread / np.sqrt(finite.size)
    offset = abs(centre) / spread

    # Two conditions, and a null falls below the threshold only on both. The displacement
    # has to be material, which the tolerance decides, and it has to be
    # resolved, which the standard error decides. Requiring only the first would
    # reject on the sampling noise of a handful of draws, since the
    # mean of n draws wanders by a fraction of the spread that is large until n
    # is in the hundreds. Requiring only the second is the criterion this
    # replaced, which tightens without limit as draws accumulate.
    resolved = bool(abs(centre) > 1.96 * error)
    material = bool(offset > CENTRED_TOLERANCE)
    return {
        "draws": int(finite.size),
        "mean": centre,
        "standard_deviation": spread,
        "standard_error_of_mean": float(error),
        "offset_in_null_standard_deviations": float(offset),
        "mean_differs_from_zero": resolved,
        "offset_is_material": material,
        "quantile_2.5": float(np.quantile(finite, 0.025)),
        "quantile_97.5": float(np.quantile(finite, 0.975)),
        "centred": not (resolved and material),
    }


def run(
    evidence_source: str = "a_mendelian",
    draws: int = 0,
    positives: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    settings = config.analysis()
    seed = settings["seed"]
    k = settings["matching"]["k"]

    if positives is None:
        everything = pd.read_parquet(config.positives_path())
        positives = everything[everything["evidence_source"] == evidence_source]
    features = control_features(features)

    rows = [run_control(positives, features, control, seed, k) for control in controls()]
    for row in rows:
        row["as_expected"] = as_expected(row)
    table = pd.DataFrame(rows)

    print(f"Controls over the {evidence_source} arm")
    print(
        "   the positive control passing shows the onset axis resolves gene class, "
        "which is the confound the mediation conditions away, and not that it resolves "
        "fitness cost"
    )
    columns = [
        "control",
        "kind",
        "expectation",
        "decay",
        "ci_low",
        "ci_high",
        "as_expected",
        "note",
    ]
    print(table[[column for column in columns if column in table.columns]].to_string(index=False))

    placebo: dict = {"draws": draws}
    permutation_draws: pd.DataFrame | None = None
    if draws:
        design = primary.build_design(
            positives,
            features,
            evolutionary.CONSERVATION_PRIMARY,
            evolutionary.CONSTRAINT_PRIMARY,
            seed,
            k,
            True,
        )
        values = stratified_permutation_null(design, therapy_areas(), draws, seed)
        placebo = summarise_null(values)
        # The draws are kept, not only their summary. A null is judged by its
        # shape as much as by its centre, and a summary cannot be re-read.
        permutation_draws = pd.DataFrame({"gamma": values})
        print()
        print(f"Placebo: onset permuted within therapeutic area, {draws} draws")
        for name, value in placebo.items():
            print(f"   {name}: {value}")
        if not placebo.get("centred", True):
            print()
            print("   the placebo null is not centred on zero, which the threshold rejects")

    print()
    print("Controls not run as specified, and why")
    for name, reason in NOT_RUN.items():
        print(f"   {name}: {reason}")

    out_dir = config.derived_dir() / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"controls_{evidence_source}.parquet", index=False)
    if permutation_draws is not None:
        permutation_draws.to_parquet(
            out_dir / f"permutation_null_{evidence_source}.parquet", index=False
        )
    with (out_dir / f"controls_{evidence_source}.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "controls": table.to_dict(orient="records"),
                "placebo": placebo,
                "not_run": NOT_RUN,
            },
            handle,
            indent=2,
            default=float,
        )
        handle.write("\n")

    unexpected = table[~table["as_expected"]]
    if not unexpected.empty:
        print()
        print("Controls that did not match their expectation")
        for _, row in unexpected.iterrows():
            print(f"   {row['control']}: decay {row['decay']:+.4f}, expected {row['expectation']}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the control analyses")
    parser.add_argument("--source", default="a_mendelian")
    parser.add_argument("--draws", type=int, default=0)
    args = parser.parse_args()
    config.ensure_dirs()
    run(evidence_source=args.source, draws=args.draws)


if __name__ == "__main__":
    main()
