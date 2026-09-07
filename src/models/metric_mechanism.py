"""The onset contrast fitted with each constraint metric in turn.

Three constraint metrics are fitted against the same conservation metric on a
common gene set, so the only quantity varying between designs is which score
enters the contrast. The gene set is the genes carrying all three constraint
metrics and the conservation metric, because the metrics differ in coverage and
designs built on their own gene sets would differ in composition as well as in
score.

Each metric is reported with the contrast, the two decays it is composed of, the
movement under each conditioning block, and the contrast under a specification
that lets the onset slope vary by mode of inheritance.

Movements under conditioning are taken against a base fitted on the rows every
specification shares, since the conditioning covariates do not cover every gene.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src import config
from src.models import mediation, primary

CONSTRAINT_METRICS = {
    "s_het": "s_het_log",
    "loeuf": "loeuf",
    "missense_z": "missense_z",
}

FEATURE_INFORMED = ("s_het",)


# The covariates conditioned on in the mediation do not stand in the same relation to the
# outcome. Conditioning on a common cause of onset and constraint removes
# confounding. Conditioning on a variable that is partly a descendant of
# disease-gene status opens a path instead, and can create association where
# there was none. Both produce a contrast that moves away from zero, so the
# direction of movement does not distinguish them and the block has to be split.
#
# The classification is by how each covariate is derived, not by judgement:
COMPONENT_ORIGIN = {
    "haploinsufficiency": "gnomAD allele counts, no disease annotation involved",
    "developmental_share": "Gene Ontology curation, which follows what genes are studied for",
    "pleiotropy_effective_traits": "UK Biobank effect estimates across traits",
    "n_therapeutic_areas": "Open Targets disease associations, counted per gene",
}

# Disease breadth is classified here and excluded from the fitted blocks; see
# mediation.EXCLUDED for why. It stays in this dict because the split reports
# what it does on its own.


def common_gene_set(features: pd.DataFrame) -> pd.DataFrame:
    """Genes carrying every metric under comparison, so only the score varies."""
    from src.features import evolutionary

    needed = [evolutionary.CONSERVATION_PRIMARY, *CONSTRAINT_METRICS.values()]
    missing = [column for column in needed if column not in features.columns]
    if missing:
        raise KeyError(f"the feature matrix lacks {missing}")
    return features.dropna(subset=needed).reset_index(drop=True)


def movement_for(
    metric_column: str,
    positives: pd.DataFrame,
    features: pd.DataFrame,
    covariates: pd.DataFrame,
    seed: int,
    k: int,
) -> dict:
    """Contrast under each mediation specification, for one constraint metric."""
    from src.features import evolutionary

    base = primary.build_design(
        positives,
        features,
        evolutionary.CONSERVATION_PRIMARY,
        metric_column,
        seed,
        k,
        matched=True,
    )

    blocks = {
        "base": [],
        "gene_class": list(mediation.BLOCKS["gene_class"]),
        "both": list(mediation.BLOCKS["pleiotropy"]) + list(mediation.BLOCKS["gene_class"]),
    }

    # The covariates do not cover every gene, so a specification that enters them
    # is fitted on fewer rows than one that does not. Reading a movement off that
    # comparison charges the change of sample to the conditioning. The movements
    # are therefore taken against a base fitted on the rows every specification
    # shares, and the contrast on all rows is kept separately because that is the
    # quantity the metric comparison reports.
    shared = mediation.restrict(base, covariates, blocks["both"])

    result = {
        "n_positives": base.metadata["n_positives"],
        "n_genes": base.metadata["n_genes"],
        "n_rows": int(base.x.shape[0]),
        "n_rows_shared": int(shared.x.shape[0]),
    }
    fitted = mediation.estimate(base, "base")
    result["base"] = float(fitted["gamma"])
    # The interval on the two-way fit is kept alongside the three-way one below,
    # so the two specifications can be compared on this row set without pairing
    # an estimate from one sample against an interval from another.
    result["base_ci_low"] = float(fitted.get("ci_low", float("nan")))
    result["base_ci_high"] = float(fitted.get("ci_high", float("nan")))
    # Gamma is a contrast of two decays and conservation is held at the same
    # metric in every row, so reporting both components says whether the metrics
    # differ in their own onset slope or whether the joint fit is moving.
    result["conservation_decay"] = -float(fitted["conservation_x_onset"])
    result["constraint_decay"] = -float(fitted["constraint_x_onset"])

    result["shared_base"] = float(mediation.estimate(shared, "shared_base")["gamma"])
    for label in ("gene_class", "both"):
        widened = mediation.augment(shared, covariates, blocks[label])
        result[label] = float(mediation.estimate(widened, label)["gamma"])
    result["moved_by_sample"] = result["shared_base"] - result["base"]
    result["moved_by_gene_class"] = result["gene_class"] - result["shared_base"]
    result["moved_by_both"] = result["both"] - result["shared_base"]

    # The arm that lets the onset slope differ by mode of inheritance. The
    # two-way primary assumes a common slope within mode and
    # returns a weighted average if that is wrong, so this is what says whether
    # the composition shift along the onset axis is carrying the contrast.
    three_way = primary.build_design(
        positives,
        features,
        evolutionary.CONSERVATION_PRIMARY,
        metric_column,
        seed,
        k,
        matched=True,
        three_way_inheritance=True,
    )
    fitted = mediation.estimate(three_way, "three_way")
    result["three_way"] = float(fitted["gamma"])
    result["three_way_ci_low"] = float(fitted.get("ci_low", float("nan")))
    result["three_way_ci_high"] = float(fitted.get("ci_high", float("nan")))
    return result


def paired_difference(
    positives: pd.DataFrame,
    features: pd.DataFrame,
    seed: int,
    k: int,
    left: str = "loeuf",
    right: str = "s_het",
    draws: int = 400,
) -> dict:
    """Bootstrap the difference between two metrics' contrasts, paired.

    Comparing two marginal intervals is the wrong test here. The two estimates
    are fitted on the same genes, the same positives and the same design, so
    their sampling errors are correlated, and marginal intervals that overlap
    can be compatible with a difference that is well determined. Only a paired
    comparison can say which. The correlation between the two contrasts across
    the draws is returned with the difference.

    Diseases are the resampling unit, because the design clusters on disease and
    a row-wise resample would break that. Each drawn disease is relabelled so
    that a disease drawn twice enters as two clusters rather than one.
    """
    import numpy as np

    from src.features import evolutionary

    columns = CONSTRAINT_METRICS
    diseases = positives["disease_id"].unique()
    rng = np.random.default_rng(seed)

    def contrast(sample: pd.DataFrame, metric: str) -> float:
        design = primary.build_design(
            sample,
            features,
            evolutionary.CONSERVATION_PRIMARY,
            columns[metric],
            seed,
            k,
            matched=True,
        )
        return float(primary.estimate_gamma(design)["gamma"])

    observed = {name: contrast(positives, name) for name in (left, right)}

    pairs = []
    for _ in range(draws):
        drawn = rng.choice(diseases, size=len(diseases), replace=True)
        frame = pd.DataFrame({"disease_id": drawn})
        frame["cluster"] = np.arange(len(frame))
        sample = frame.merge(positives, on="disease_id", how="left")
        sample = sample.assign(disease_id=sample["cluster"].astype(str)).drop(columns="cluster")
        try:
            pairs.append((contrast(sample, left), contrast(sample, right)))
        except (ValueError, KeyError):
            continue

    drawn_pairs = np.array(pairs).reshape(-1, 2)
    values = drawn_pairs[:, 0] - drawn_pairs[:, 1]
    low, high = np.percentile(values, [2.5, 97.5]) if values.size > 20 else (np.nan, np.nan)
    correlation = (
        float(np.corrcoef(drawn_pairs[:, 0], drawn_pairs[:, 1])[0, 1])
        if values.size > 20
        else float("nan")
    )
    return {
        "left": left,
        "right": right,
        f"observed_{left}": observed[left],
        f"observed_{right}": observed[right],
        "observed_difference": observed[left] - observed[right],
        "draws": int(values.size),
        "bootstrap_mean": float(values.mean()) if values.size else float("nan"),
        "ci_low": float(low),
        "ci_high": float(high),
        "excludes_zero": bool(values.size > 20 and low * high > 0),
        "correlation": correlation,
        f"sd_{left}": float(drawn_pairs[:, 0].std(ddof=1)) if values.size > 1 else float("nan"),
        f"sd_{right}": float(drawn_pairs[:, 1].std(ddof=1)) if values.size > 1 else float("nan"),
    }


def attenuation_bound(
    positives: pd.DataFrame,
    features: pd.DataFrame,
    covariates: pd.DataFrame,
    seed: int,
    k: int,
    draws: int = 400,
) -> dict:
    """Movement of the contrast under all four covariates, across the change of rows.

    The covariates do not cover every gene, so entering them drops rows, and the
    movement measured against the contrast on all rows combines that change of
    sample with the effect of conditioning. component_bound separates the two by
    fitting every specification on the rows they share. The change is a paired
    quantity on a single design, so it is bootstrapped as one rather than read
    off two marginal intervals.
    """
    import numpy as np

    from src.features import evolutionary

    blocks = list(COMPONENT_ORIGIN)
    diseases = positives["disease_id"].unique()
    rng = np.random.default_rng(seed)

    def change(sample: pd.DataFrame, metric: str) -> float:
        design = primary.build_design(
            sample,
            features,
            evolutionary.CONSERVATION_PRIMARY,
            CONSTRAINT_METRICS[metric],
            seed,
            k,
            matched=True,
        )
        base = mediation.estimate(design, "base")["gamma"]
        widened = mediation.augment(design, covariates, blocks)
        adjusted = mediation.estimate(widened, "both")["gamma"]
        return float(adjusted) - float(base)

    observed = {name: change(positives, name) for name in CONSTRAINT_METRICS}

    collected: dict[str, list[float]] = {name: [] for name in CONSTRAINT_METRICS}
    for _ in range(draws):
        drawn = rng.choice(diseases, size=len(diseases), replace=True)
        frame = pd.DataFrame({"disease_id": drawn})
        frame["cluster"] = np.arange(len(frame))
        sample = frame.merge(positives, on="disease_id", how="left")
        sample = sample.assign(disease_id=sample["cluster"].astype(str)).drop(columns="cluster")
        for name in CONSTRAINT_METRICS:
            try:
                collected[name].append(change(sample, name))
            except (ValueError, KeyError):
                continue

    results = {}
    for name, values in collected.items():
        array = np.array(values)
        low, high = np.percentile(array, [2.5, 97.5]) if array.size > 20 else (np.nan, np.nan)
        results[name] = {
            "observed_change": observed[name],
            "draws": int(array.size),
            "ci_low": float(low),
            "ci_high": float(high),
            "excludes_zero": bool(array.size > 20 and low * high > 0),
        }
    return results


def component_bound(
    positives: pd.DataFrame,
    features: pd.DataFrame,
    covariates: pd.DataFrame,
    seed: int,
    k: int,
    draws: int = 400,
) -> dict:
    """Attribute the movement under conditioning to individual covariates.

    The whole block moves the contrast away from zero. That is what a suppressed
    confounder looks like and also what a collider looks like, so the block is
    entered one covariate at a time. Whether the movement is carried by the
    covariate that cannot be downstream of the outcome decides which reading the
    design supports.
    """
    import numpy as np

    from src.features import evolutionary

    components = list(COMPONENT_ORIGIN)
    # Every covariate is entered on its own, the excluded one included, because
    # it is reported on its own. Only the retained set is entered jointly as a
    # specification.
    retained = [name for name in components if name not in mediation.EXCLUDED]
    diseases = positives["disease_id"].unique()
    rng = np.random.default_rng(seed)

    def changes(sample: pd.DataFrame, metric: str) -> dict[str, float]:
        full = primary.build_design(
            sample,
            features,
            evolutionary.CONSERVATION_PRIMARY,
            CONSTRAINT_METRICS[metric],
            seed,
            k,
            matched=True,
        )
        design = mediation.restrict(full, covariates, components)
        base = float(mediation.estimate(design, "base")["gamma"])
        out = {"shared_rows_contrast": base}
        for name in components:
            widened = mediation.augment(design, covariates, [name])
            out[name] = float(mediation.estimate(widened, name)["gamma"]) - base
        for label, columns in (("retained_together", retained), ("all_four", components)):
            widened = mediation.augment(design, covariates, columns)
            out[label] = float(mediation.estimate(widened, label)["gamma"]) - base
        out["n_rows"] = float(design.x.shape[0])
        return out

    keys = [*components, "retained_together", "all_four", "shared_rows_contrast"]
    observed = {name: changes(positives, name) for name in CONSTRAINT_METRICS}

    collected: dict[tuple[str, str], list[float]] = {
        (metric, key): [] for metric in CONSTRAINT_METRICS for key in keys
    }
    for _ in range(draws):
        drawn = rng.choice(diseases, size=len(diseases), replace=True)
        frame = pd.DataFrame({"disease_id": drawn})
        frame["cluster"] = np.arange(len(frame))
        sample = frame.merge(positives, on="disease_id", how="left")
        sample = sample.assign(disease_id=sample["cluster"].astype(str)).drop(columns="cluster")
        for metric in CONSTRAINT_METRICS:
            try:
                values = changes(sample, metric)
            except (ValueError, KeyError):
                continue
            for key in keys:
                collected[(metric, key)].append(values[key])

    results: dict[str, dict] = {}
    for metric in CONSTRAINT_METRICS:
        entry: dict[str, object] = {"n_rows": int(observed[metric]["n_rows"])}
        for key in keys:
            array = np.array(collected[(metric, key)])
            low, high = np.percentile(array, [2.5, 97.5]) if array.size > 20 else (np.nan, np.nan)
            entry[key] = {
                "origin": COMPONENT_ORIGIN.get(
                    key,
                    (
                        "the contrast on the shared rows"
                        if key == "shared_rows_contrast"
                        else "a joint specification"
                    ),
                ),
                "excluded": key in mediation.EXCLUDED,
                "observed_change": observed[metric][key],
                "draws": int(array.size),
                "ci_low": float(low),
                "ci_high": float(high),
                "excludes_zero": bool(array.size > 20 and low * high > 0),
            }
        results[metric] = entry
    return results


def annotation_depth() -> pd.DataFrame:
    """Is the developmental covariate a measure of function or of study effort?

    Gene Ontology annotation accumulates on genes that have been studied, and
    genes are studied because they cause disease. A covariate that tracks
    annotation count is therefore partly a record of the outcome. The care-burden
    component of the severity index correlates 0.790 with annotation count, which
    is why that component is rarefied.
    """
    from scipy.stats import spearmanr

    from src.features import gene_class

    table = gene_class.load()
    depth = table["n_process_terms"]
    rows = []
    for column in ("n_developmental_terms", "developmental_share", "is_developmental"):
        values = table[column].astype(float)
        keep = values.notna() & depth.notna() & (depth > 0)
        rho, _ = spearmanr(values[keep], depth[keep])
        rows.append(
            {
                "covariate": column,
                "n_genes": int(keep.sum()),
                "spearman_with_annotation_count": float(rho),
            }
        )
    return pd.DataFrame(rows)


def score_correlations(features: pd.DataFrame, positives: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation among the constraint metrics, overall and by onset level.

    If the three agree uniformly, they differ in how they relate to disease-gene
    status rather than in what they measure. If they diverge at one end of the
    axis, the disagreement is localised there.
    """
    columns = list(CONSTRAINT_METRICS.values())
    onset = positives.drop_duplicates("disease_id").set_index("disease_id")
    genes = positives[["disease_id", "gene_id"]].merge(
        onset[["onset_bin"]], left_on="disease_id", right_index=True, how="left"
    )
    genes = genes.merge(
        features[["ensembl_gene_id", *columns]],
        left_on="gene_id",
        right_on="ensembl_gene_id",
        how="inner",
    ).dropna(subset=columns)

    rows = []
    pairs = [(a, b) for i, a in enumerate(columns) for b in columns[i + 1 :]]
    for label, block in [("all genes", genes)] + [
        (str(name), part) for name, part in genes.groupby("onset_bin", observed=True)
    ]:
        if len(block) < 50:
            continue
        row = {"stratum": label, "n_genes": int(block["gene_id"].nunique())}
        for left, right in pairs:
            row[f"{left} vs {right}"] = round(
                float(block[left].corr(block[right], method="spearman")), 3
            )
        rows.append(row)
    return pd.DataFrame(rows)


def inputs(evidence_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Positives and the common-gene-set features for one evidence source."""
    from src.features import evolutionary
    from src.models import robustness

    positives = pd.read_parquet(config.positives_path())
    positives = positives[positives["evidence_source"] == evidence_source]
    features = evolutionary.load()
    positives, features = robustness.prepare_inputs(positives, features)
    return positives, common_gene_set(features)


def bootstrap(evidence_source: str = "a_mendelian", draws: int = 400) -> dict:
    """The disease bootstraps: paired metric differences and conditioning bounds."""
    settings = config.analysis()
    seed, k = settings["seed"], settings["matching"]["k"]
    positives, features = inputs(evidence_source)
    covariates = mediation.load_blocks()

    records = {
        "metric_paired_difference": {
            left: paired_difference(positives, features, seed, k, left, "s_het", draws)
            for left in ("loeuf", "missense_z")
        },
        "component_bound": component_bound(positives, features, covariates, seed, k, draws),
        "attenuation_bound": attenuation_bound(positives, features, covariates, seed, k, draws),
    }
    out_dir = config.derived_dir() / "mediation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, record in records.items():
        with (out_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, default=float)
            handle.write("\n")

    for left, entry in records["metric_paired_difference"].items():
        print(
            f"   {left} minus s_het: {entry['observed_difference']:+.4f} "
            f"[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}], "
            f"correlation between the contrasts {entry['correlation']:.3f}"
        )
    return records


def run(evidence_source: str = "a_mendelian") -> pd.DataFrame:
    settings = config.analysis()
    seed, k = settings["seed"], settings["matching"]["k"]

    positives, features = inputs(evidence_source)
    covariates = mediation.load_blocks()

    rows = []
    for name, column in CONSTRAINT_METRICS.items():
        row = movement_for(column, positives, features, covariates, seed, k)
        row["metric"] = name
        row["feature_informed"] = name in FEATURE_INFORMED
        rows.append(row)

    table = pd.DataFrame(rows)[
        [
            "metric",
            "feature_informed",
            "n_positives",
            "n_genes",
            "n_rows",
            "n_rows_shared",
            "conservation_decay",
            "constraint_decay",
            "base",
            "base_ci_low",
            "base_ci_high",
            "shared_base",
            "gene_class",
            "both",
            "moved_by_sample",
            "moved_by_gene_class",
            "moved_by_both",
            "three_way",
            "three_way_ci_low",
            "three_way_ci_high",
        ]
    ]

    print(f"Constraint metrics under the mediation specifications, {evidence_source} arm")
    print("Common gene set, so only the constraint score differs between rows.")
    print()
    print(table.to_string(index=False))

    informed = table[table["feature_informed"]]
    direct = table[~table["feature_informed"]]
    correlations = score_correlations(features, positives)
    if not correlations.empty:
        print()
        print("Rank correlation among the constraint metrics")
        print(correlations.to_string(index=False))

    print()
    if not informed.empty and not direct.empty:
        for column in ("moved_by_gene_class", "moved_by_both"):
            largest_direct = direct[column].max()
            observed = float(informed[column].iloc[0])
            holds = observed > largest_direct
            print(
                f"   {column}: s_het {observed:+.4f} against a largest direct measure of "
                f"{largest_direct:+.4f}. Prediction holds: {holds}"
            )

    out_dir = config.derived_dir() / "mediation"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"metric_mechanism_{evidence_source}.parquet", index=False)
    if not correlations.empty:
        correlations.to_parquet(
            out_dir / f"score_correlations_{evidence_source}.parquet", index=False
        )
    with (out_dir / f"metric_mechanism_{evidence_source}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(table.to_dict(orient="records"), handle, indent=2, default=float)
        handle.write("\n")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Test why the constraint metrics disagree")
    parser.add_argument("--source", default="a_mendelian")
    parser.add_argument(
        "--draws", type=int, default=400, help="disease resamples for the bootstraps, 0 to skip"
    )
    args = parser.parse_args()
    config.ensure_dirs()
    run(evidence_source=args.source)
    if args.draws:
        bootstrap(evidence_source=args.source, draws=args.draws)


if __name__ == "__main__":
    main()
