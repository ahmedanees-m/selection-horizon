"""Descriptive measurements of the analysis set.

Three quantities, none of which is a component of either estimand.

Per-disease discrimination of the non-evolutionary model, arm by arm, which is
the denominator of the incremental-value ratio. The differential character of
the Evo 2 exclusion, measured on constraint, coding length, base composition and
disease-gene status. And the association between label confidence and onset,
which bears on how much of an onset gradient is attributable to label quality.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from src import config

ARMS = ("a_mendelian", "b_gwas")

# Estimand 2 excludes diseases whose non-evolutionary model barely
# discriminates, because the ratio it forms has that lift in the denominator.
CANDIDATE_FLOORS = (0.03, 0.05, 0.10)


def _positives() -> pd.DataFrame:
    return pd.read_parquet(config.positives_path())


def genes_per_disease_against_onset() -> dict:
    """How far the labelling density and the exposure move together.

    Reported per arm as a rank correlation over diseases, and as the median
    gene count per onset level, because a correlation alone does not show
    whether the relationship is a gradient or one extreme bin.
    """
    from src.models import primary

    positives = _positives()
    results: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        arm_rows = positives[positives["evidence_source"] == arm]
        column = primary.onset_column_for(arm_rows)
        subset = arm_rows[arm_rows[column].notna()]
        if subset.empty:
            results[arm] = {"note": "no positives with an onset"}
            continue

        per_disease = (
            subset.groupby("disease_id")
            .agg(n_genes=("gene_id", "nunique"), onset_bin=(column, "first"))
            .reset_index()
        )

        position = primary.onset_position(per_disease["onset_bin"])

        usable = np.isfinite(position)
        log_genes = np.log(per_disease["n_genes"].clip(lower=1).to_numpy(float))
        spearman = stats.spearmanr(position[usable], log_genes[usable])
        pearson = stats.pearsonr(position[usable], log_genes[usable])

        results[arm] = {
            "n_diseases": int(usable.sum()),
            "spearman": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "pearson_on_log": float(pearson.statistic),
            "median_genes_by_onset": (
                per_disease.groupby("onset_bin")["n_genes"].median().to_dict()
            ),
            "onset_column": column,
            "bins_not_recognised": int((~np.isfinite(position)).sum()),
        }
    return results


def baseline_discrimination() -> dict:
    """Per-disease discrimination of the non-evolutionary model, arm by arm.

    Only M0 is fitted. The evolutionary scores are not put into any model here,
    so nothing computed in this function is a component of either estimand.
    """
    from src.features import baseline, evolutionary
    from src.models import log_r, primary

    settings = config.analysis()
    seed, k = settings["seed"], settings["matching"]["k"]
    features = evolutionary.load().merge(baseline.load(), on="ensembl_gene_id", how="left")
    positives = _positives()

    results: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        subset = positives[positives["evidence_source"] == arm]
        try:
            design = primary.build_design(
                subset,
                features,
                evolutionary.CONSERVATION_PRIMARY,
                evolutionary.CONSTRAINT_PRIMARY,
                seed,
                k,
                matched=False,
            )
        except (ValueError, KeyError) as error:
            results[arm] = {"note": str(error)[:160]}
            continue

        lookup = features.drop_duplicates("ensembl_gene_id").set_index("ensembl_gene_id")
        columns = [c for c in log_r.BASELINE_COLUMNS if c in lookup.columns]
        values = np.column_stack(
            [pd.Series(design.gene).map(lookup[c]).to_numpy(dtype=float) for c in columns]
        )

        rows = []
        for disease in pd.unique(design.disease):
            mask = design.disease == disease
            y = design.y[mask]
            if int(y.sum()) < log_r.MIN_POSITIVES:
                continue
            auc = log_r.cross_validated_auroc(log_r._standardise(values[mask]), y, seed)
            rows.append({"disease_id": disease, "n_positives": int(y.sum()), "auc_m0": auc})

        table = pd.DataFrame(rows)
        if table.empty:
            results[arm] = {"note": "no disease carries enough positives to score"}
            continue

        lift = table["auc_m0"] - 0.5
        results[arm] = {
            "n_diseases_scored": int(len(table)),
            "auc_m0_median": float(table["auc_m0"].median()),
            "auc_m0_quartiles": [float(table["auc_m0"].quantile(q)) for q in (0.25, 0.5, 0.75)],
            "diseases_above_floor": {
                str(floor): int((lift > floor).sum()) for floor in CANDIDATE_FLOORS
            },
        }
        table.to_parquet(
            config.derived_dir() / "descriptives" / f"baseline_discrimination_{arm}.parquet",
            index=False,
        )
    return results


def evo2_exclusion_is_differential() -> dict:
    """Whether the genes the stop rule cannot reach are an unrepresentative set."""
    from src.evo2 import score as scoring
    from src.features import evolutionary

    starts = scoring.canonical_coding_starts()
    eligible = set(scoring.eligible(starts)["ensembl_gene_id"])
    starts["eligible"] = starts["ensembl_gene_id"].isin(eligible)

    features = evolutionary.load()[["ensembl_gene_id", "s_het_log", "cds_length", "gc_content"]]
    table = starts.merge(features, on="ensembl_gene_id", how="left")

    positives = _positives()
    disease_genes = set(positives.loc[positives["evidence_source"].isin(ARMS), "gene_id"])
    table["is_disease_gene"] = table["ensembl_gene_id"].isin(disease_genes)

    results: dict[str, object] = {
        "n_genes": int(len(table)),
        "n_eligible": int(table["eligible"].sum()),
        "n_excluded": int((~table["eligible"]).sum()),
        "excluded_fraction": float((~table["eligible"]).mean()),
    }

    for column in ("s_het_log", "cds_length", "gc_content"):
        kept = table.loc[table["eligible"], column].dropna()
        dropped = table.loc[~table["eligible"], column].dropna()
        if kept.empty or dropped.empty:
            continue
        test = stats.mannwhitneyu(kept, dropped, alternative="two-sided")
        # A rank-biserial correlation reads as an effect size on a scale that
        # does not change with sample size, which matters because a difference
        # of no consequence is significant at twenty thousand genes.
        effect = 2.0 * test.statistic / (len(kept) * len(dropped)) - 1.0
        results[column] = {
            "median_eligible": float(kept.median()),
            "median_excluded": float(dropped.median()),
            "rank_biserial": float(effect),
            "p": float(test.pvalue),
        }

    share_kept = float(table.loc[table["eligible"], "is_disease_gene"].mean())
    share_dropped = float(table.loc[~table["eligible"], "is_disease_gene"].mean())
    counts = pd.crosstab(table["eligible"], table["is_disease_gene"])
    chi = stats.chi2_contingency(counts.to_numpy())
    results["disease_gene_status"] = {
        "share_among_eligible": share_kept,
        "share_among_excluded": share_dropped,
        "difference": share_kept - share_dropped,
        "p": float(chi.pvalue),
    }
    return results


def label_confidence_against_onset() -> dict:
    """Whether the locus-to-gene score of the GWAS arm varies with onset.

    The score is the label quality of that arm. If it falls as onset rises then
    the labels are getting weaker along the exposure, which attenuates any
    discrimination measured against them and does so differentially.
    """
    from src.ingest import open_targets

    evidence = open_targets.read(
        "evidence_gwas_credible_sets", columns=["diseaseId", "targetId", "score"]
    )
    evidence = evidence.dropna(subset=["score"])

    from src.models import primary

    positives = _positives()
    gwas = positives[positives["evidence_source"] == "b_gwas"]
    column = primary.onset_column_for(gwas)
    onset = gwas[gwas[column].notna()].drop_duplicates("disease_id").set_index("disease_id")[column]
    evidence["onset_bin"] = evidence["diseaseId"].map(onset)
    joined = evidence.dropna(subset=["onset_bin"])
    if joined.empty:
        return {"note": "no GWAS evidence rows carry an onset"}

    position = primary.onset_position(joined["onset_bin"])
    usable = np.isfinite(position)
    joined = joined.loc[usable]
    spearman = stats.spearmanr(position[usable], joined["score"].to_numpy(float))

    strong = joined.assign(strong=joined["score"] >= 0.5)
    return {
        "onset_column": column,
        "n_rows": int(len(joined)),
        "spearman_score_against_onset": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "median_score_by_onset": joined.groupby("onset_bin")["score"].median().to_dict(),
        "share_above_half_by_onset": strong.groupby("onset_bin")["strong"].mean().to_dict(),
    }


def run() -> dict:
    out_dir = config.derived_dir() / "descriptives"
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = {
        "genes_per_disease_against_onset": genes_per_disease_against_onset,
        "baseline_discrimination": baseline_discrimination,
        "evo2_exclusion_is_differential": evo2_exclusion_is_differential,
        "label_confidence_against_onset": label_confidence_against_onset,
    }

    results: dict[str, dict[str, object]] = {}
    for name, function in checks.items():
        print(f"=== {name} ===", flush=True)
        try:
            results[name] = function()
        except (ValueError, KeyError, FileNotFoundError) as error:
            results[name] = {"note": f"{type(error).__name__}: {error}"}
        print(json.dumps(results[name], indent=2, default=float))
        print(flush=True)

    with (out_dir / "descriptives.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=float)
        handle.write("\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Measurements the pre-registration quotes")
    parser.parse_args()
    config.ensure_dirs()
    run()


if __name__ == "__main__":
    main()
