"""Power simulation on the realised design.

This runs before anything else is built, because it decides whether the rest is
worth building. The question it answers is narrow: given the gene-disease counts
the data actually contain, how large would Gamma have to be before this design
could distinguish it from zero.

Two steps.

Calibration. The main effect is not guessed. The pooled discrimination of s_het
and LOEUF for Orphanet disease genes against matched controls is estimated from
the real data, and the simulation runs at that discrimination. The score by
onset interaction is deliberately not fitted at this step, and the
pre-registration records that it was not inspected before filing. Estimating a
main effect for design calibration is a clean use of pilot data; estimating the
estimand and then pre-registering it would not be.

Simulation. Gamma is defined as the contrast of two decay coefficients, so it is
a linear contrast of two terms in one logistic regression fitted to the same
gene-disease rows. Standard errors are clustered on disease, which is the
practical stand-in for the disease random effect at simulation scale. The
correlation between the conservation and constraint scores is not known until
the conservation tracks are aggregated, and it drives the variance of a
difference, so it is carried as a simulation axis rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config
from src.features import gene_table, matching

SCORE_CORRELATIONS = (0.2, 0.4, 0.6)

CONSERVATION_PRIMARY = "phylop_241way_fraction_above"
CONSTRAINT_PRIMARY = "s_het_log"


def auroc_to_d(auroc: float) -> float:
    """Standardised mean difference implied by an AUROC under equal variances."""
    return float(np.sqrt(2.0) * norm.ppf(auroc))


def d_to_auroc(d: float) -> float:
    return float(norm.cdf(d / np.sqrt(2.0)))


def auroc_to_d_array(auroc: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0) * norm.ppf(auroc)


def rank_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC by rank, which needs no probability model.

    Ties take midranks. Ordinal ranks would bias the estimate wherever a score
    repeats, and pLI, thresholded features and any rounded score repeat heavily.
    """
    from scipy.stats import rankdata

    ranks = rankdata(scores, method="average")
    positive = labels == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass
class Design:
    """The realised gene-disease design, per collapsed onset level."""

    bins: list[str]
    diseases_per_bin: list[int]
    genes_per_disease: dict[str, list[int]] = field(default_factory=dict)
    k_controls: int = 10

    @property
    def pairs_per_bin(self) -> list[int]:
        return [int(sum(self.genes_per_disease[b])) for b in self.bins]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "onset_bin": self.bins,
                "n_diseases": self.diseases_per_bin,
                "n_pairs": self.pairs_per_bin,
            }
        )


def design_from_positives(positives: pd.DataFrame, k: int) -> Design:
    """Read the design off the real positives rather than parameterising it."""
    counts = positives.groupby(["onset_bin", "disease_id"])["gene_id"].nunique().rename("n_genes")
    order = [
        b for b in config.analysis()["onset"]["collapse"]["order"] if b in counts.index.levels[0]
    ]
    return Design(
        bins=order,
        diseases_per_bin=[int(counts.loc[b].shape[0]) for b in order],
        genes_per_disease={b: counts.loc[b].tolist() for b in order},
        k_controls=k,
    )


SCORE_DIRECTIONS = (("s_het_log", 1), ("loeuf", -1), ("missense_z", 1))


def measured_score_correlation() -> float | None:
    """Spearman correlation between the primary conservation and constraint scores.

    It drives the variance of their difference, and Gamma is a difference, so it
    is the single largest lever on the minimum detectable effect. Once the
    conservation aggregates exist the simulated grid no longer has to stand in
    for it.
    """
    from scipy.stats import spearmanr

    path = config.interim_dir() / "conservation_phylop_241way.parquet"
    if not path.exists():
        return None

    conservation = pd.read_parquet(path)
    if CONSERVATION_PRIMARY not in conservation.columns:
        return None

    joined = (
        gene_table.load()
        .merge(
            conservation[["ensembl_gene_id", CONSERVATION_PRIMARY]],
            on="ensembl_gene_id",
            how="inner",
        )
        .dropna(subset=[CONSTRAINT_PRIMARY, CONSERVATION_PRIMARY])
    )
    if joined.empty:
        return None
    return float(spearmanr(joined[CONSTRAINT_PRIMARY], joined[CONSERVATION_PRIMARY]).statistic)


def calibrate(positives: pd.DataFrame, genes: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Pooled marginal discrimination of the constraint metrics.

    Reported matched and unmatched, because the two bound the answer from
    opposite directions and the design is calibrated against both. Matching on
    expected loss-of-function count is legitimate under the causal graph, but
    that count is the denominator of every constraint metric, so the matched
    figure is the conservative end of the range rather than the true one.

    Pooled across onset levels on purpose. No interaction is fitted here.
    """
    rows = []

    positive_genes = set(positives["gene_id"].dropna())
    pool = genes.dropna(subset=["cds_length_decile", "expected_lof_decile", "gc_content"])
    unmatched_y = pool["ensembl_gene_id"].isin(positive_genes).to_numpy().astype(int)
    for score, direction in SCORE_DIRECTIONS:
        usable = pool[score].notna().to_numpy()
        if not usable.any():
            continue
        auroc = rank_auroc(direction * pool.loc[usable, score].to_numpy(), unmatched_y[usable])
        rows.append(
            {
                "design": "unmatched",
                "score": score,
                "n_positive": int(unmatched_y[usable].sum()),
                "n_control": int((unmatched_y[usable] == 0).sum()),
                "auroc": round(auroc, 4),
                "implied_d": round(auroc_to_d(auroc), 4),
            }
        )

    matched = matching.draw_controls(positives, genes, k=10, seed=seed)
    joined = matched.merge(
        genes.rename(columns={"ensembl_gene_id": "gene_id"}), on="gene_id", how="left"
    )
    for score, direction in SCORE_DIRECTIONS:
        usable = joined.dropna(subset=[score])
        if usable.empty:
            continue
        auroc = rank_auroc(direction * usable[score].to_numpy(), usable["y"].to_numpy())
        rows.append(
            {
                "design": "matched",
                "score": score,
                "n_positive": int((usable["y"] == 1).sum()),
                "n_control": int((usable["y"] == 0).sum()),
                "auroc": round(auroc, 4),
                "implied_d": round(auroc_to_d(auroc), 4),
            }
        )

    report = pd.DataFrame(rows)
    report.attrs["control_shortfalls"] = matched.attrs.get("control_shortfalls", 0)
    report.attrs["balance"] = matching.standardised_mean_differences(
        matched,
        genes,
        ["cds_length", "gc_content", "expected_lof", "s_het_log", "loeuf", "missense_z"],
    )
    return report


def _simulate_once(args: tuple) -> tuple[float, float]:
    """One replicate. Returns the Gamma estimate and its standard error."""
    design, base_auroc, gamma, correlation, k, seed = args
    rng = np.random.default_rng(seed)

    disease_index: list[int] = []
    onset_level: list[int] = []
    label: list[int] = []
    counter = 0
    for level, bin_name in enumerate(design.bins):
        for n_genes in design.genes_per_disease[bin_name]:
            counter += 1
            n_rows = n_genes * (1 + k)
            disease_index.extend([counter] * n_rows)
            onset_level.extend([level] * n_rows)
            label.extend([1] * n_genes + [0] * n_genes * k)

    disease_id = np.asarray(disease_index)
    onset = np.asarray(onset_level, dtype=float)
    y = np.asarray(label, dtype=float)
    n = len(y)

    span = max(len(design.bins) - 1, 1)
    onset_scaled = onset / span

    # Constraint holds its discrimination across the onset range; conservation
    # loses Gamma AUROC points across it. Only the contrast is identified, so
    # placing the whole decay on one arm is a normalisation, not an assumption.
    d_base = auroc_to_d(base_auroc)
    d_constraint = np.full(n, d_base)
    decayed = np.clip(base_auroc - gamma * onset_scaled, 0.05, 0.95)
    d_conservation = auroc_to_d_array(decayed)

    cov = np.array([[1.0, correlation], [correlation, 1.0]])
    noise = rng.multivariate_normal(np.zeros(2), cov, size=n)
    z_conservation = noise[:, 0] + y * d_conservation
    z_constraint = noise[:, 1] + y * d_constraint

    # Standardise as the analysis would, so coefficients are on a common scale.
    z_conservation = (z_conservation - z_conservation.mean()) / z_conservation.std()
    z_constraint = (z_constraint - z_constraint.mean()) / z_constraint.std()

    centred = onset_scaled - onset_scaled.mean()
    x = np.column_stack(
        [
            np.ones(n),
            z_conservation,
            z_constraint,
            centred,
            z_conservation * centred,
            z_constraint * centred,
        ]
    )

    beta, cov_beta = _logit_cluster(x, y, disease_id)
    if beta is None:
        return float("nan"), float("nan")

    contrast = np.zeros(x.shape[1])
    contrast[4], contrast[5] = -1.0, 1.0
    estimate = float(contrast @ beta)
    variance = float(contrast @ cov_beta @ contrast)
    return estimate, float(np.sqrt(variance)) if variance > 0 else float("nan")


def _logit_cluster(x: np.ndarray, y: np.ndarray, clusters: np.ndarray):
    """Logistic regression by Newton-Raphson with cluster-robust covariance."""
    beta = np.zeros(x.shape[1])
    for _ in range(60):
        eta = np.clip(x @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        gradient = x.T @ (y - mu)
        hessian = (x * w[:, None]).T @ x
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None, None
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    else:
        return None, None

    eta = np.clip(x @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1 - mu), 1e-9, None)
    bread = np.linalg.inv((x * w[:, None]).T @ x)

    residual = (y - mu)[:, None] * x
    order = np.argsort(clusters, kind="mergesort")
    sorted_clusters = clusters[order]
    boundaries = np.flatnonzero(np.diff(sorted_clusters)) + 1
    meat = np.zeros((x.shape[1], x.shape[1]))
    for block in np.split(residual[order], boundaries):
        total = block.sum(axis=0)
        meat += np.outer(total, total)
    return beta, bread @ meat @ bread


def run_power(
    design: Design,
    base_auroc: float,
    deltas: list[float],
    replicates: int,
    correlations=SCORE_CORRELATIONS,
    seed: int = 0,
    workers: int = 8,
) -> pd.DataFrame:
    alpha = config.analysis()["estimands"]["gamma"]["alpha"]
    critical = norm.ppf(1 - alpha / 2)

    jobs, index = [], []
    for correlation in correlations:
        for gamma in deltas:
            for replicate in range(replicates):
                jobs.append(
                    (
                        design,
                        base_auroc,
                        gamma,
                        correlation,
                        design.k_controls,
                        seed + hash((correlation, gamma, replicate)) % 2**31,
                    )
                )
                index.append((correlation, gamma))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_simulate_once, jobs, chunksize=8))

    frame = pd.DataFrame(index, columns=["score_correlation", "gamma"])
    frame["estimate"] = [value for value, _ in results]
    frame["se"] = [value for _, value in results]
    frame["reject"] = (frame["estimate"].abs() / frame["se"]) > critical

    summary = (
        frame.groupby(["score_correlation", "gamma"])
        .agg(
            power=("reject", "mean"),
            mean_estimate=("estimate", "mean"),
            mean_se=("se", "mean"),
            n=("reject", "size"),
        )
        .reset_index()
    )
    return summary


def minimum_detectable_effect(summary: pd.DataFrame, target: float = 0.80) -> pd.DataFrame:
    """Smallest Gamma reaching the target power, interpolated between grid points."""
    rows = []
    for correlation, group in summary.groupby("score_correlation"):
        group = group.sort_values("gamma")
        gammas = group["gamma"].to_numpy()
        powers = group["power"].to_numpy()
        reached = np.flatnonzero(powers >= target)
        if len(reached) == 0:
            mde = float("nan")
        elif reached[0] == 0:
            mde = float(gammas[0])
        else:
            i = reached[0]
            span = powers[i] - powers[i - 1]
            fraction = (target - powers[i - 1]) / span if span > 0 else 0.0
            mde = float(gammas[i - 1] + fraction * (gammas[i] - gammas[i - 1]))
        rows.append({"score_correlation": correlation, "mde_auroc_equivalent": round(mde, 4)})
    return pd.DataFrame(rows)


# Two axes. Inheritance mode, because s_het is only interpretable where
# heterozygotes carry a fitness cost; and the number of onset levels, because
# concentrating the design on the contrast that carries the signal is the named
# re-scope option if five levels prove underpowered.
STRATA = {
    "all_mendelian": None,
    "dominant_only": ("dominant",),
}

TWO_LEVEL = {
    "perinatal": "early",
    "infancy": "early",
    "childhood": "early",
    "adolescent": "early",
    "adult": "adult",
}


def to_two_levels(positives: pd.DataFrame) -> pd.DataFrame:
    frame = positives.copy()
    frame["onset_bin"] = frame["onset_bin"].map(TWO_LEVEL)
    return frame.dropna(subset=["onset_bin"])


def two_level_design(positives: pd.DataFrame, k: int) -> Design:
    frame = to_two_levels(positives)
    counts = frame.groupby(["onset_bin", "disease_id"])["gene_id"].nunique().rename("n_genes")
    order = [b for b in ("early", "adult") if b in counts.index.get_level_values(0)]
    return Design(
        bins=order,
        diseases_per_bin=[int(counts.loc[b].shape[0]) for b in order],
        genes_per_disease={b: counts.loc[b].tolist() for b in order},
        k_controls=k,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Power simulation on the realised design")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    settings = config.analysis()
    seed = settings["seed"]
    replicates = args.replicates or settings["power_simulation"]["replicates"]
    deltas = settings["power_simulation"]["deltas"]
    k = settings["matching"]["k"]

    out_dir = config.derived_dir() / "analysis_set"
    out_dir.mkdir(parents=True, exist_ok=True)

    positives = pd.read_parquet(out_dir / "positives_collapsed.parquet")
    mendelian = positives[
        (positives["evidence_source"] == "a_mendelian") & positives["onset_bin"].notna()
    ]
    genes = gene_table.load()

    measured = measured_score_correlation()
    correlations = list(SCORE_CORRELATIONS)
    if measured is not None:
        correlations = sorted({*correlations, round(measured, 3)})
        print(
            f"Measured correlation between {CONSERVATION_PRIMARY} and "
            f"{CONSTRAINT_PRIMARY}: {measured:.3f}"
        )
    else:
        print("Conservation aggregates absent; the correlation stays a simulated axis")
    print()

    pilots, designs, curves, mdes = [], {}, [], []

    for stratum, modes in STRATA.items():
        subset = (
            mendelian if modes is None else mendelian[mendelian["inheritance_mode"].isin(modes)]
        )

        pilot = calibrate(subset, genes, seed=seed)
        pilot.insert(0, "stratum", stratum)
        pilots.append(pilot)

        variants = {
            "five_level": design_from_positives(subset, k=k),
            "two_level": two_level_design(subset, k=k),
        }
        designs[stratum] = variants["five_level"]

        print(f"[{stratum}] pooled marginal discrimination, no interaction fitted")
        print(pilot.to_string(index=False))
        for name, design in variants.items():
            print()
            print(f"[{stratum}] {name} design")
            print(design.summary().to_string(index=False))
        print()

        for name, design in variants.items():
            for label in ("matched", "unmatched"):
                selected = pilot[(pilot["design"] == label) & (pilot["score"] == "s_het_log")]
                if not len(selected):
                    continue
                base_auroc = float(selected["auroc"].iloc[0])
                summary = run_power(
                    design,
                    base_auroc,
                    deltas,
                    replicates,
                    correlations=correlations,
                    seed=seed,
                    workers=args.workers,
                )
                for position, (column, value) in enumerate(
                    (
                        ("stratum", stratum),
                        ("levels", name),
                        ("calibration", label),
                        ("base_auroc", round(base_auroc, 4)),
                    )
                ):
                    summary.insert(position, column, value)
                curves.append(summary)

                mde = minimum_detectable_effect(
                    summary, settings["power_simulation"]["target_power"]
                )
                for position, (column, value) in enumerate(
                    (
                        ("stratum", stratum),
                        ("levels", name),
                        ("calibration", label),
                        ("base_auroc", round(base_auroc, 4)),
                    )
                ):
                    mde.insert(position, column, value)
                mdes.append(mde)

    pilot_table = pd.concat(pilots, ignore_index=True)
    pilot_table.to_csv(out_dir / "pilot_discrimination.csv", index=False)

    balance = pilots[0].attrs["balance"]
    balance.to_csv(out_dir / "matching_balance.csv", index=False)
    print("Covariate balance after matching, all Mendelian")
    print(balance.to_string(index=False))

    curve = pd.concat(curves, ignore_index=True)
    curve.to_csv(out_dir / "power_curve.csv", index=False)
    mde_table = pd.concat(mdes, ignore_index=True)

    print()
    print("Minimum detectable effect for Gamma at 80 percent power")
    print(mde_table.to_string(index=False))

    record = {
        "measured_score_correlation": measured,
        "replicates": replicates,
        "k_controls": k,
        "strata": {
            name: design.summary().to_dict(orient="records") for name, design in designs.items()
        },
        "pilot": pilot_table.to_dict(orient="records"),
        "balance": balance.to_dict(orient="records"),
        "power_curve": curve.to_dict(orient="records"),
        "mde": mde_table.to_dict(orient="records"),
    }
    with (out_dir / "power_record.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=float)
        handle.write("\n")


if __name__ == "__main__":
    main()
