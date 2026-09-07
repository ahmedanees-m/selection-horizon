"""The robustness suite.

Every arm runs and every arm is tabulated. The arms fall into five kinds.

    score swaps         which metric stands for each family, and how
                        conservation is aggregated
    design swaps        matched against unmatched controls, and the number of
                        controls per positive
    restriction arms    dropping a gene class or a disease class that could
                        carry the result on its own
    encoding swaps      how onset is coded
    inclusion arms      diseases the primary design excludes

Two further arms are listed in NOT_RUN with the reason. The Evo 2 size
comparison depends on scores the validation did not release, and the
birth-cohort restriction needs individual birth years that the registry does not
publish.

Each arm returns Gamma with its interval on the same scale as the primary
estimate, and the table reports the difference from the primary and flags an arm
that changes it by more than half.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.features import evolutionary
from src.models import primary

MATERIAL_CHANGE = 0.5


@dataclass
class Arm:
    """One robustness arm: the specification it changes and the reason."""

    name: str
    kind: str
    rationale: str
    conservation: str | None = None
    constraint: str | None = None
    k: int | None = None
    matched: bool | None = None
    gene_filter: str | None = None
    disease_filter: str | None = None
    onset_encoding: str = "level"
    three_way_inheritance: bool = False
    adjust_genes_per_disease: bool = True
    include_all_ages: bool = False


def arms() -> list[Arm]:
    return [
        Arm("primary", "reference", "the pre-registered specification"),
        Arm(
            "constraint_loeuf",
            "score swap",
            "LOEUF for s_het. Roughly a third of genes are underpowered for LOEUF, "
            "which is why s_het is primary",
            constraint="loeuf",
        ),
        Arm(
            "constraint_missense_z",
            "score swap",
            "missense constraint rather than loss-of-function constraint",
            constraint="missense_z",
        ),
        Arm(
            "constraint_gnomad_v2",
            "score swap",
            "the previous gnomAD release, to show the answer is not release specific",
            constraint="loeuf_v2",
        ),
        Arm(
            "conservation_100way",
            "score swap",
            "the 100-way alignment for the 241-way Zoonomia one",
            conservation="phylop_100way_fraction_above",
        ),
        Arm(
            "conservation_phastcons",
            "score swap",
            "a different conservation statistic on the same alignment depth",
            conservation="phastcons_100way_fraction_above",
        ),
        Arm(
            "conservation_fourfold",
            "score swap",
            "four-fold degenerate sites rather than fraction above threshold. These "
            "correlate less with constraint, so this arm has less power by construction",
            conservation="phylop_241way_mean_fourfold",
        ),
        Arm("controls_k5", "design swap", "five controls per positive", k=5),
        Arm("controls_k20", "design swap", "twenty controls per positive", k=20),
        Arm(
            "unmatched",
            "design swap",
            "controls drawn at random rather than on the confounders. Reported as a "
            "primary output, not only here",
            matched=False,
        ),
        Arm(
            "autosomal_only",
            "restriction",
            "dropping the sex chromosomes, where constraint is estimated differently",
            gene_filter="autosomal",
        ),
        Arm(
            "no_hla_no_olfactory",
            "restriction",
            "dropping two gene families under selective regimes of their own",
            gene_filter="no_hla_no_olfactory",
        ),
        Arm(
            "oncology_excluded",
            "restriction",
            "dropping hereditary cancer predisposition, which is a sixth of the adult "
            "cell and whose genes are DNA repair genes constrained for reasons "
            "unrelated to when the cancer appears",
            disease_filter="no_oncology",
        ),
        Arm(
            "inheritance_three_way",
            "design swap",
            "the onset slope allowed to differ by mode of inheritance rather than "
            "assumed common within mode. The two-way primary returns a weighted "
            "average of two slopes if they genuinely differ, and this arm is what "
            "says whether they do",
            three_way_inheritance=True,
        ),
        Arm(
            "onset_span_midpoint",
            "encoding swap",
            "the midpoint of a disease's interval span rather than its earliest interval",
            onset_encoding="span_midpoint",
        ),
        Arm(
            "all_ages_included",
            "inclusion",
            "diseases annotated only as spanning every age, entered at the midpoint of that span",
            onset_encoding="span_value",
            include_all_ages=True,
        ),
    ]


NOT_RUN = {
    "evo2_size_comparison": "the Evo 2 arm released no scores to compare",
    "birth_cohort_restriction": (
        "needs individual birth years, which the registry does not publish for the " "restriction"
    ),
    "t1b_and_t3_onset_swaps": "T1b is a concordance figure only and T3 is not yet built",
}


# Columns a restriction arm needs, which do not travel with the score matrix.
# Attached by prepare_inputs before any arm runs.
REQUIRED_COLUMNS = {
    "autosomal": ("is_autosomal",),
    "no_hla_no_olfactory": ("is_hla", "is_olfactory"),
}


def prepare_inputs(
    positives: pd.DataFrame, features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the columns the restriction and score-swap arms need.

    The score matrix carries scores. Gene-family membership lives in the gene
    table and disease names live in the Platform index, so an arm that filters on
    either needs them joined first.
    """
    from src.features import gene_table

    try:
        genes = gene_table.load()
    except FileNotFoundError:
        return positives, features

    wanted = [
        column
        for column in ("is_autosomal", "is_hla", "is_olfactory", "loeuf_v2")
        if column in genes.columns and column not in features.columns
    ]
    if wanted:
        features = features.merge(
            genes[["ensembl_gene_id", *wanted]], on="ensembl_gene_id", how="left"
        )

    if "loeuf_v2" not in features.columns:
        try:
            v2 = gene_table.load_constraint_v2()
        except FileNotFoundError:
            v2 = pd.DataFrame()
        if "loeuf_v2" in getattr(v2, "columns", []):
            features = features.merge(
                v2[["ensembl_gene_id", "loeuf_v2"]], on="ensembl_gene_id", how="left"
            )

    if "disease_name" not in positives.columns:
        from src.ingest import open_targets

        try:
            index = open_targets.read("disease", columns=["id", "name"])
        except (FileNotFoundError, KeyError):
            return positives, features
        names = index.drop_duplicates("id").set_index("id")["name"]
        positives = positives.assign(disease_name=positives["disease_id"].map(names))

    return positives, features


def apply_filters(
    positives: pd.DataFrame, features: pd.DataFrame, arm: Arm
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict the gene or disease set for a restriction arm.

    A filter that cannot find the column it needs raises. Skipping it instead
    returns the reference estimate under a restriction arm's name, which reads as
    a robustness arm that passed rather than one that never ran.
    """
    if arm.gene_filter:
        required = REQUIRED_COLUMNS.get(arm.gene_filter, ())
        missing = [column for column in required if column not in features.columns]
        if missing:
            raise KeyError(f"{arm.gene_filter} needs {missing}, which the feature matrix lacks")

        if arm.gene_filter == "autosomal":
            features = features[features["is_autosomal"].fillna(True).astype(bool)]
        elif arm.gene_filter == "no_hla_no_olfactory":
            excluded = features["is_hla"].fillna(False).astype(bool) | features[
                "is_olfactory"
            ].fillna(False).astype(bool)
            features = features[~excluded]

    if arm.disease_filter == "no_oncology":
        from src.diagnostics.analysis_set import classify_disease

        if "disease_name" not in positives.columns:
            raise KeyError("no_oncology needs disease_name, which the positives lack")
        categories = positives["disease_name"].map(classify_disease)
        positives = positives[categories != "hereditary cancer predisposition"]

    return positives, features


def with_all_ages(positives: pd.DataFrame) -> pd.DataFrame:
    """Re-admit diseases annotated only as spanning every age interval.

    The plan excludes them from the continuous axis because they carry no
    ordinal position, and names their inclusion at the span midpoint as a
    robustness arm. That arm needs a numeric onset for every disease rather than
    a level: each annotated disease takes the midpoint of its own interval span,
    and an all-ages disease takes the midpoint of the full span from the earliest
    interval to the latest.
    """
    from src.onset import tiers

    if primary.onset_column_for(positives) != "onset_bin":
        raise ValueError("the all-ages arm is defined on the Orphanet onset vocabulary only")

    long = pd.read_parquet(config.interim_dir() / "orphanet" / "orphanet_onset.parquet")
    spans = tiers.build_t1a(long).set_index("orpha_code")["onset_span_midpoint"]
    intervals = tiers.interval_table().dropna(subset=["rank"])
    full_span = float((intervals["midpoint"].min() + intervals["midpoint"].max()) / 2.0)

    every = long.groupby("orpha_code")["onset_interval"].apply(
        lambda values: set(values.dropna()) == {tiers.ALL_AGES}
    )
    all_ages = set(every[every].index)

    codes = pd.to_numeric(
        positives["disease_id"].astype(str).str.replace("Orphanet_", "", regex=False),
        errors="coerce",
    )
    values = codes.map(spans).where(~codes.isin(all_ages), full_span)

    widened = positives.assign(**{primary.ONSET_VALUE_COLUMN: values.to_numpy()})
    return widened.dropna(subset=[primary.ONSET_VALUE_COLUMN])


def run_arm(
    positives: pd.DataFrame,
    features: pd.DataFrame,
    arm: Arm,
    seed: int,
    default_k: int,
) -> dict:
    conservation = arm.conservation or evolutionary.CONSERVATION_PRIMARY
    constraint = arm.constraint or evolutionary.CONSTRAINT_PRIMARY
    k = arm.k or default_k
    matched = True if arm.matched is None else arm.matched

    if conservation not in features.columns or constraint not in features.columns:
        return {
            "arm": arm.name,
            "kind": arm.kind,
            "rationale": arm.rationale,
            "gamma": float("nan"),
            "note": f"{conservation} or {constraint} is not in the feature matrix",
        }

    try:
        if arm.include_all_ages:
            positives = with_all_ages(positives)
        subset, restricted = apply_filters(positives, features, arm)
        design = primary.build_design(
            subset,
            restricted,
            conservation,
            constraint,
            seed,
            k,
            matched,
            adjust_genes_per_disease=arm.adjust_genes_per_disease,
            three_way_inheritance=arm.three_way_inheritance,
            onset_encoding=arm.onset_encoding,
        )
        result = primary.estimate_gamma(design)
    except (ValueError, KeyError, FileNotFoundError) as error:
        return {
            "arm": arm.name,
            "kind": arm.kind,
            "rationale": arm.rationale,
            "gamma": float("nan"),
            "note": str(error),
        }

    return {
        "arm": arm.name,
        "kind": arm.kind,
        "rationale": arm.rationale,
        "conservation": conservation,
        "constraint": constraint,
        "k": k,
        "matched": matched,
        "n_rows": result["n_rows"],
        "n_positives": result["n_positives"],
        "n_diseases": result["n_diseases"],
        "gamma": result["gamma"],
        "standard_error": result["standard_error"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "note": "",
    }


def summarise(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    reference = table.loc[table["arm"] == "primary", "gamma"]
    baseline = float(reference.iloc[0]) if len(reference) else np.nan

    table["difference_from_primary"] = table["gamma"] - baseline
    table["sign_agrees"] = np.sign(table["gamma"]) == np.sign(baseline)
    table["excludes_zero"] = table["ci_low"] * table["ci_high"] > 0

    # An arm that flips the sign, or that moves Gamma by more than half the
    # primary estimate, is flagged separately
    # rather than in a supplementary table.
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.abs(table["difference_from_primary"] / baseline)
    # An arm that did not fit has no estimate to compare, so it is listed with its
    # note rather than escalated as a change to the conclusion.
    table["large_change"] = ((~table["sign_agrees"]) | (relative > MATERIAL_CHANGE)) & table[
        "gamma"
    ].notna()
    table.loc[table["arm"] == "primary", "large_change"] = False
    return table


def run(evidence_source: str = "a_mendelian", positives=None, features=None) -> pd.DataFrame:
    settings = config.analysis()
    seed = settings["seed"]
    default_k = settings["matching"]["k"]

    if positives is None:
        all_positives = pd.read_parquet(
            config.derived_dir() / "analysis_set" / "positives_collapsed.parquet"
        )
        positives = all_positives[all_positives["evidence_source"] == evidence_source]
    if features is None:
        features = evolutionary.load()

    positives, features = prepare_inputs(positives, features)

    rows = [run_arm(positives, features, arm, seed, default_k) for arm in arms()]
    table = summarise(rows)

    out_dir = config.derived_dir() / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"robustness_{evidence_source}.parquet", index=False)
    with (out_dir / f"robustness_{evidence_source}.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "arms": table.to_dict(orient="records"),
                "not_run": NOT_RUN,
            },
            handle,
            indent=2,
            default=float,
        )
        handle.write("\n")

    columns = ["arm", "kind", "n_positives", "gamma", "ci_low", "ci_high", "large_change", "note"]
    print(f"Robustness suite over the {evidence_source} arm")
    print(table[[c for c in columns if c in table.columns]].to_string(index=False))
    print()
    print("Arms not run, and why")
    for name, reason in NOT_RUN.items():
        print(f"   {name}: {reason}")

    escalate = table[table["large_change"]]
    if not escalate.empty:
        print()
        print("Arms that change the primary estimate by more than half")
        for _, row in escalate.iterrows():
            print(f"   {row['arm']}: gamma {row['gamma']:+.4f} against {row.get('note', '')}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every robustness arm")
    parser.add_argument("--source", default="a_mendelian")
    args = parser.parse_args()
    config.ensure_dirs()
    run(evidence_source=args.source)


if __name__ == "__main__":
    main()
