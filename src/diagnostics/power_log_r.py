"""Minimum detectable effect for the second primary estimand.

The power simulation reports what size of Gamma this design can resolve. It says
nothing about the log R slope, and counts are not power: knowing that 50
Mendelian and 278 complex diseases clear the discrimination floor does not say
what slope across them could be detected. Without that number, a null result on
the second estimand would have nothing to be bounded against, which is the
exposure that motivated pre-registering in the first place.

This runs on the realised design rather than an assumed one. Each disease
contributes its own measured baseline discrimination and its own count of
positives, both taken from the measured baseline, so the simulation inherits
the actual spread of disease sizes rather than a convenient average.

The quantity simulated is the sampling error of a per-disease AUROC, which has a
closed form. For a disease with p positives and n controls, the variance of an
AUROC A is the Hanley and McNeil expression, which is what turns a disease with
eleven positives into a wide interval and a disease with three hundred into a
narrow one.

Two assumptions are stated rather than buried.

**The two AUROCs are treated as independent.** They are not: M1 contains M0, they
are computed on the same rows, and their errors are positively correlated, which
makes the variance of their ratio smaller than independence implies. Treating
them as independent therefore overstates the noise and returns a minimum
detectable effect that is too large. That is the conservative direction for a
pre-registration, and the correlated cases are reported alongside so the size of
the conservatism is visible rather than assumed.

**The intercept has to be set.** The estimand is a slope, but the variance of
log R depends on where R sits, so a baseline has to be chosen. It cannot be
measured without fitting M1, which is the estimand. A grid is reported instead.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config

ARMS = ("a_mendelian", "b_gwas")

# Change in log R across the full onset range. The estimand is the slope on onset
# scaled to the unit interval, so a slope of -0.5 means R falls by about 40 per
# cent between the earliest and latest onset.
SLOPE_GRID = (-1.6, -1.2, -0.9, -0.7, -0.5, -0.35, -0.25, -0.15, -0.1)

# Where R sits at the earliest onset. Reported across a grid because it cannot be
# measured without fitting the model the estimand comes from.
BASELINE_RATIOS = (1.15, 1.30, 1.60)

# Correlation between the sampling errors of the two nested AUROCs. Zero is the
# conservative primary; the others show how much conservatism that buys.
ERROR_CORRELATIONS = (0.0, 0.5, 0.8)

REPLICATES = 2000
TARGET_POWER = 0.80


def hanley_mcneil_variance(auroc: np.ndarray, positives: np.ndarray, controls: np.ndarray):
    """Sampling variance of an AUROC, from the counts on each side.

    This is what makes the simulation reflect the realised design. A disease with
    eleven positives and a disease with three hundred are not the same evidence,
    and averaging them into one standard error would hide exactly the imbalance
    the second estimand is exposed to.
    """
    a = np.clip(auroc, 0.5001, 0.9999)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    numerator = a * (1.0 - a) + (positives - 1.0) * (q1 - a * a) + (controls - 1.0) * (q2 - a * a)
    return np.clip(numerator / (positives * controls), 1e-12, None)


def realised_design(arm: str, k: int) -> pd.DataFrame:
    """Per-disease baseline discrimination, positive counts and onset position."""
    from src.models import primary

    path = config.derived_dir() / "descriptives" / f"baseline_discrimination_{arm}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run src.diagnostics.descriptives before this simulation"
        )
    table = pd.read_parquet(path)

    positives = pd.read_parquet(config.positives_path())
    arm_rows = positives[positives["evidence_source"] == arm]
    column = primary.onset_column_for(arm_rows)
    onset = arm_rows.drop_duplicates("disease_id").set_index("disease_id")[column]

    table["onset_bin"] = table["disease_id"].map(onset)
    table["onset"] = primary.onset_position(table["onset_bin"])
    table["n_controls"] = table["n_positives"] * k
    table["lift_m0"] = table["auc_m0"] - 0.5
    return table.dropna(subset=["onset", "auc_m0"]).reset_index(drop=True)


def _slope_with_error(onset: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Slope of values on onset with a heteroskedasticity-robust standard error."""
    x = np.column_stack([np.ones(len(onset)), onset])
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ values
    residual = values - x @ beta
    n, width = len(onset), x.shape[1]
    meat = (x * (residual**2)[:, None]).T @ x * (n / max(n - width, 1))
    covariance = inverse @ meat @ inverse
    return float(beta[1]), float(np.sqrt(max(covariance[1, 1], 0.0)))


def power_at(
    design: pd.DataFrame,
    slope: float,
    baseline_ratio: float,
    correlation: float,
    floor: float,
    alpha: float,
    replicates: int,
    seed: int,
) -> float:
    """Share of replicates whose interval excludes zero, under a true slope."""
    rng = np.random.default_rng(seed)
    onset = design["onset"].to_numpy(float)
    lift_m0 = design["lift_m0"].to_numpy(float)
    positives = design["n_positives"].to_numpy(float)
    controls = design["n_controls"].to_numpy(float)

    true_log_ratio = np.log(baseline_ratio) + slope * onset
    true_lift_m1 = lift_m0 * np.exp(true_log_ratio)

    sd_m0 = np.sqrt(hanley_mcneil_variance(lift_m0 + 0.5, positives, controls))
    sd_m1 = np.sqrt(
        hanley_mcneil_variance(np.clip(true_lift_m1 + 0.5, 0.5, 0.999), positives, controls)
    )

    critical = norm.ppf(1 - alpha / 2)
    detected = 0
    for _ in range(replicates):
        shared = rng.standard_normal(len(onset))
        independent = rng.standard_normal(len(onset))
        noise_m1 = correlation * shared + np.sqrt(max(1.0 - correlation**2, 0.0)) * independent

        observed_m0 = lift_m0 + sd_m0 * shared
        observed_m1 = true_lift_m1 + sd_m1 * noise_m1

        usable = (observed_m0 > floor) & (observed_m1 > 0)
        if usable.sum() < 5:
            continue
        ratio = observed_m1[usable] / observed_m0[usable]
        estimate, error = _slope_with_error(onset[usable], np.log(ratio))
        if error > 0 and abs(estimate) > critical * error:
            detected += 1
    return detected / replicates


def minimum_detectable_slope(powers: list[tuple[float, float]], target: float) -> float:
    """Smallest slope magnitude reaching the target power, interpolated."""
    ordered = sorted(powers, key=lambda item: abs(item[0]))
    for (low_slope, low_power), (high_slope, high_power) in zip(ordered, ordered[1:], strict=False):
        if low_power < target <= high_power:
            span = high_power - low_power
            if span <= 0:
                return abs(high_slope)
            weight = (target - low_power) / span
            return abs(low_slope) + weight * (abs(high_slope) - abs(low_slope))
    return float("nan")


def run(replicates: int = REPLICATES) -> dict:
    settings = config.analysis()
    seed = settings["seed"]
    k = settings["matching"]["k"]
    alpha = settings["estimands"]["gamma"]["alpha"]

    from src.models import log_r

    results: dict[str, object] = {
        "replicates": replicates,
        "alpha": alpha,
        "floor": log_r.MIN_BASELINE_LIFT,
        "target_power": TARGET_POWER,
        "note": (
            "the two nested AUROCs are treated as independent in the primary case, "
            "which overstates the noise and returns a conservative minimum detectable effect"
        ),
    }

    for arm in ARMS:
        try:
            design = realised_design(arm, k)
        except FileNotFoundError as error:
            results[arm] = {"note": str(error)}
            continue

        arm_result: dict[str, object] = {
            "n_diseases": int(len(design)),
            "n_clearing_floor": int((design["lift_m0"] > log_r.MIN_BASELINE_LIFT).sum()),
            "median_positives": float(design["n_positives"].median()),
            "median_auc_m0": float(design["auc_m0"].median()),
            "onset_levels": int(design["onset"].nunique()),
        }

        grid = {}
        for ratio in BASELINE_RATIOS:
            for correlation in ERROR_CORRELATIONS:
                powers = [
                    (
                        slope,
                        power_at(
                            design,
                            slope,
                            ratio,
                            correlation,
                            log_r.MIN_BASELINE_LIFT,
                            alpha,
                            replicates,
                            seed,
                        ),
                    )
                    for slope in SLOPE_GRID
                ]
                grid[f"ratio_{ratio}_correlation_{correlation}"] = {
                    "power_by_slope": {str(s): round(p, 4) for s, p in powers},
                    "mde": round(minimum_detectable_slope(powers, TARGET_POWER), 4),
                }
        arm_result["grid"] = grid
        arm_result["mde_primary"] = grid[f"ratio_{BASELINE_RATIOS[1]}_correlation_0.0"]["mde"]
        results[arm] = arm_result

        print(f"=== {arm} ===")
        print(
            f"   diseases {arm_result['n_diseases']}, clearing the floor "
            f"{arm_result['n_clearing_floor']}, median positives "
            f"{arm_result['median_positives']:.0f}"
        )
        for key, value in grid.items():
            print(f"   {key:<34} mde {value['mde']}")
        print()

    out_dir = config.derived_dir() / "analysis_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "power_log_r.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=float)
        handle.write("\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimum detectable log R slope")
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args()
    config.ensure_dirs()
    run(replicates=args.replicates)


if __name__ == "__main__":
    main()
