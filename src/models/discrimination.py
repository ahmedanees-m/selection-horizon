"""Discrimination of each score family, by onset level and by inheritance stratum.

Two descriptive quantities, each read straight off a fitted design or a scored
gene table. Neither is an estimand; both are the stratified figures the analysis
reports.

**By onset level.** Gamma is a contrast of two onset slopes, which says how
discrimination changes but not what it is at any point on the axis. The
model-implied value at each level converts the fitted score effect at that level
onto the AUROC scale the power simulation calibrated, and the empirical value at
each level is the same quantity measured directly on the rows in that level.
Reporting both says whether the fitted slope describes the data it was fitted on.

**By inheritance stratum.** s_het is a selection coefficient on heterozygotes and
recessive disease genes have unaffected heterozygotes, so the constraint metrics
are not interchangeable on recessive disease. The stratum figures are measured
twice. Against the pool of scored genes, which is how a prioritisation using one
metric would rank those genes, and against controls matched on the covariates the
primary design matches on, which asks whether the metrics differ once coding
length, expected loss-of-function count and base composition are held fixed. The
two answer different questions and both are reported: LOEUF is a function of the
expected count, so the unmatched comparison carries that covariate inside it.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.diagnostics.power import rank_auroc
from src.models import primary

# Measured on the synthetic generator: the fitted contrast is linear in the
# generated effect through the origin at about nine and a third logit units per
# AUROC-equivalent unit. The same constant converts a score effect at one onset
# level onto the AUROC scale.
LOGIT_PER_AUROC = 9.33

STRATA = ("recessive", "dominant")
DRAWS = 1000

# The covariates the primary design matches positives and controls on.
MATCHING_COVARIATES = ["cds_length", "expected_lof", "gc_content"]

# LOEUF runs the other way: a low value is a constrained gene. It is negated so
# that every metric here points the same way and an AUROC above a half means the
# metric ranks disease genes as more constrained.
INVERTED = ("loeuf", "loeuf_v2")


def implied_auroc(effect: float) -> float:
    return 0.5 + float(effect) / LOGIT_PER_AUROC


def by_onset(design: primary.Design, fitted: dict, seed: int) -> pd.DataFrame:
    """Discrimination at each onset level, model-implied and measured."""
    levels = np.unique(design.onset)
    rng = np.random.default_rng(seed)
    conservation = design.x[:, design.names.index("z_conservation")]
    constraint = design.x[:, design.names.index("z_constraint")]

    rows = []
    for family, values, main, interaction in (
        ("conservation", conservation, "conservation_main", "conservation_x_onset"),
        ("constraint", constraint, "constraint_main", "constraint_x_onset"),
    ):
        for level in levels:
            mask = design.onset == level
            y = design.y[mask]
            if y.sum() < 5 or (1 - y).sum() < 5:
                continue
            observed = rank_auroc(values[mask], y)
            draws = []
            index = np.flatnonzero(mask)
            for _ in range(200):
                taken = rng.choice(index, size=index.size, replace=True)
                if design.y[taken].sum() < 5:
                    continue
                draws.append(rank_auroc(values[taken], design.y[taken]))
            array = np.array(draws)
            low, high = np.percentile(array, [2.5, 97.5]) if array.size > 20 else (np.nan, np.nan)
            centred = float(level) - float(design.onset.mean())
            rows.append(
                {
                    "family": family,
                    "onset": float(level),
                    "n_positives": int(y.sum()),
                    "n_rows": int(mask.sum()),
                    "model_implied_auroc": implied_auroc(
                        fitted[main] + fitted[interaction] * centred
                    ),
                    "measured_auroc": float(observed),
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
    return pd.DataFrame(rows)


def per_disease_auroc(design: primary.Design, minimum: int) -> pd.DataFrame:
    """Observed discrimination of each score family within one disease."""
    conservation = design.x[:, design.names.index("z_conservation")]
    constraint = design.x[:, design.names.index("z_constraint")]

    rows = []
    for disease in pd.unique(design.disease):
        mask = design.disease == disease
        y = design.y[mask]
        if int(y.sum()) < minimum or int((1 - y).sum()) < minimum:
            continue
        rows.append(
            {
                "disease_id": disease,
                "onset": float(np.unique(design.onset[mask])[0]),
                "n_positives": int(y.sum()),
                "conservation": rank_auroc(conservation[mask], y),
                "constraint": rank_auroc(constraint[mask], y),
            }
        )
    return pd.DataFrame(rows)


def scored_pool(features: pd.DataFrame, metrics: dict[str, str]) -> pd.DataFrame:
    """Genes carrying every metric and every matching covariate."""
    scored = features.drop_duplicates("ensembl_gene_id").dropna(subset=list(metrics.values()))
    return scored.dropna(subset=MATCHING_COVARIATES).reset_index(drop=True)


def matched_blocks(
    stratum_genes: set[str],
    pool: pd.DataFrame,
    disease_genes: set[str],
    k: int,
    seed: int,
    bins: int = 10,
) -> tuple[list[tuple[int, list[int]]], int]:
    """Each disease gene of a stratum with k control genes matched to it.

    Controls are genes in the same confounder stratum that cause no Mendelian
    disease. They are drawn without replacement within one gene's draw and may
    serve more than one gene, which is how the primary design draws controls
    within a disease. The count of genes whose stratum cannot supply k is
    returned rather than filled from a neighbouring stratum.
    """
    from src.features import matching

    rng = np.random.default_rng(seed)
    assigned = matching.strata(pool, bins=bins)
    candidates: dict[str, list[int]] = {}
    for position, (name, gene) in enumerate(zip(assigned, pool["ensembl_gene_id"], strict=True)):
        if gene not in disease_genes:
            candidates.setdefault(name, []).append(position)

    blocks, shortfalls = [], 0
    for position, gene in enumerate(pool["ensembl_gene_id"]):
        if gene not in stratum_genes:
            continue
        available = candidates.get(assigned.iloc[position], [])
        if len(available) < k:
            shortfalls += 1
        size = min(k, len(available))
        drawn = rng.choice(available, size=size, replace=False) if size else []
        blocks.append((position, [int(index) for index in drawn]))
    return blocks, shortfalls


def by_inheritance_matched(
    positives: pd.DataFrame,
    features: pd.DataFrame,
    metrics: dict[str, str],
    seed: int,
    k: int,
    bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Discrimination within each inheritance stratum, against matched controls.

    Returns the per-metric figures, the paired differences against s_het, and the
    balance the matching achieves on each covariate.
    """
    pool = scored_pool(features, metrics)
    values = {
        name: pool[column].to_numpy(dtype=float) * (-1.0 if name in INVERTED else 1.0)
        for name, column in metrics.items()
    }
    disease_genes = set(positives["gene_id"])
    rng = np.random.default_rng(seed)

    rows, differences, balance = [], [], []
    for stratum in STRATA:
        genes = set(positives.loc[positives["inheritance_mode"] == stratum, "gene_id"])
        blocks, shortfalls = matched_blocks(genes, pool, disease_genes, k, seed, bins)
        if len(blocks) < 20:
            continue

        positive_rows = np.array([block[0] for block in blocks])
        control_rows = np.array([index for _, controls in blocks for index in controls])
        edges = np.cumsum([0] + [len(controls) for _, controls in blocks])
        rows_all = np.concatenate([positive_rows, control_rows])
        y_all = np.concatenate([np.ones(positive_rows.size), np.zeros(control_rows.size)])
        observed = {name: rank_auroc(values[name][rows_all], y_all) for name in metrics}

        draws: dict[str, list[float]] = {name: [] for name in metrics}
        for _ in range(DRAWS):
            drawn = rng.integers(0, len(blocks), len(blocks))
            taken = np.concatenate(
                [positive_rows[drawn]] + [control_rows[edges[i] : edges[i + 1]] for i in drawn]
            )
            labels = np.concatenate([np.ones(drawn.size), np.zeros(taken.size - drawn.size)])
            for name in metrics:
                draws[name].append(rank_auroc(values[name][taken], labels))

        for name in metrics:
            low, high = np.percentile(np.array(draws[name]), [2.5, 97.5])
            rows.append(
                {
                    "stratum": stratum,
                    "metric": name,
                    "n_disease_genes": len(blocks),
                    "n_controls": int(control_rows.size),
                    "controls_per_gene": round(control_rows.size / len(blocks), 2),
                    "genes_short_of_k": shortfalls,
                    "auroc": observed[name],
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "excludes_chance": bool(low > 0.5 or high < 0.5),
                }
            )

        for left in metrics:
            if left == "s_het":
                continue
            paired = np.array(draws[left]) - np.array(draws["s_het"])
            low, high = np.percentile(paired, [2.5, 97.5])
            differences.append(
                {
                    "stratum": stratum,
                    "comparison": f"{left} minus s_het",
                    "observed_difference": observed[left] - observed["s_het"],
                    "draws": int(paired.size),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "excludes_zero": bool(low * high > 0),
                }
            )

        for covariate in MATCHING_COVARIATES:
            series = pool[covariate].to_numpy(dtype=float)
            treated, control = series[positive_rows], series[control_rows]
            spread = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
            balance.append(
                {
                    "stratum": stratum,
                    "covariate": covariate,
                    "median_disease_genes": float(np.median(treated)),
                    "median_matched_controls": float(np.median(control)),
                    "median_pool": float(np.median(series)),
                    "smd_matched": (
                        float((treated.mean() - control.mean()) / spread)
                        if spread > 0
                        else float("nan")
                    ),
                    "smd_pool": (
                        float((treated.mean() - series.mean()) / series.std(ddof=1))
                        if series.std(ddof=1) > 0
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(differences), pd.DataFrame(balance)


def by_inheritance(
    positives: pd.DataFrame, features: pd.DataFrame, metrics: dict[str, str], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discrimination within each inheritance stratum, against the scored pool.

    The negatives are every other scored gene rather than matched controls,
    because the question is where a prioritisation would place these genes, not
    whether they differ from a control set built to hold confounders fixed.
    """
    scored = features.drop_duplicates("ensembl_gene_id").dropna(subset=list(metrics.values()))
    pool = scored["ensembl_gene_id"].to_numpy()
    values = {
        name: scored[column].to_numpy(dtype=float) * (-1.0 if name in INVERTED else 1.0)
        for name, column in metrics.items()
    }
    rng = np.random.default_rng(seed)

    rows, differences = [], []
    for stratum in STRATA:
        genes = set(positives.loc[positives["inheritance_mode"] == stratum, "gene_id"])
        y = np.isin(pool, list(genes)).astype(float)
        if y.sum() < 20:
            continue

        index = np.arange(pool.size)
        draws: dict[str, list[float]] = {name: [] for name in metrics}
        for _ in range(DRAWS):
            taken = rng.choice(index, size=index.size, replace=True)
            if y[taken].sum() < 20:
                continue
            for name in metrics:
                draws[name].append(rank_auroc(values[name][taken], y[taken]))

        for name in metrics:
            array = np.array(draws[name])
            low, high = np.percentile(array, [2.5, 97.5]) if array.size > 20 else (np.nan, np.nan)
            rows.append(
                {
                    "stratum": stratum,
                    "metric": name,
                    "n_disease_genes": int(y.sum()),
                    "n_pool": int(pool.size),
                    "auroc": rank_auroc(values[name], y),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "excludes_chance": bool(low > 0.5 or high < 0.5),
                }
            )

        # Paired on the same resample, which is the comparison the divergence
        # claim rests on. Marginal intervals would overstate its width.
        length = min(len(draws[name]) for name in metrics)
        for left in metrics:
            if left == "s_het":
                continue
            paired = np.array(draws[left][:length]) - np.array(draws["s_het"][:length])
            low, high = np.percentile(paired, [2.5, 97.5]) if paired.size > 20 else (np.nan, np.nan)
            differences.append(
                {
                    "stratum": stratum,
                    "comparison": f"{left} minus s_het",
                    "observed_difference": rank_auroc(values[left], y)
                    - rank_auroc(values["s_het"], y),
                    "draws": int(paired.size),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "excludes_zero": bool(low * high > 0),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(differences)


def run(evidence_source: str = "a_mendelian") -> dict[str, pd.DataFrame]:
    settings = config.analysis()
    seed, k = settings["seed"], settings["matching"]["k"]
    minimum = int(settings["presentation"]["per_disease_auroc_min_positives"])

    from src.features import evolutionary
    from src.models import metric_mechanism, robustness

    positives = pd.read_parquet(config.positives_path())
    positives = positives[positives["evidence_source"] == evidence_source]
    features = evolutionary.load()
    positives, features = robustness.prepare_inputs(positives, features)

    design = primary.build_design(
        positives,
        features,
        evolutionary.CONSERVATION_PRIMARY,
        evolutionary.CONSTRAINT_PRIMARY,
        seed,
        k,
        matched=True,
    )
    fitted = primary.estimate_gamma(design)

    onset = by_onset(design, fitted, seed)
    diseases = per_disease_auroc(design, minimum)
    # The pool is every gene carrying all three constraint metrics. The
    # conservation metric is not required, because no contrast is being fitted.
    strata, paired = by_inheritance(positives, features, metric_mechanism.CONSTRAINT_METRICS, seed)
    matched, matched_paired, balance = by_inheritance_matched(
        positives, features, metric_mechanism.CONSTRAINT_METRICS, seed, k
    )

    out = config.derived_dir() / "estimands"
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "onset_discrimination": onset,
        "per_disease_auroc": diseases,
        "inheritance_discrimination": strata,
        "inheritance_paired_difference": paired,
        "inheritance_discrimination_matched": matched,
        "inheritance_paired_difference_matched": matched_paired,
        "inheritance_matching_balance": balance,
    }
    for name, table in tables.items():
        table.to_parquet(out / f"{name}_{evidence_source}.parquet", index=False)
    with (out / f"discrimination_{evidence_source}.json").open("w", encoding="utf-8") as handle:
        payload = {name: frame.to_dict(orient="records") for name, frame in tables.items()}
        json.dump(payload, handle, indent=2, default=float)
        handle.write("\n")

    for name, table in tables.items():
        print(f"--- {name}")
        print(table.round(4).to_string(index=False))
        print()
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Discrimination by onset level and inheritance")
    parser.add_argument("--source", default="a_mendelian")
    args = parser.parse_args()
    config.ensure_dirs()
    run(evidence_source=args.source)


if __name__ == "__main__":
    main()
