"""T2: the complex-disease onset axis, from FinnGen through Risteys.

What a register records is age at first recorded event, which is not age of
onset:

    age_recorded = age_onset + diagnostic_delay + registry_left_truncation

The measurement model handles what can be handled from published aggregates and
states plainly what cannot.

**Left truncation.** Finnish registers begin at fixed calendar years, so a
participant born in 1940 cannot contribute an event at age 20. The execution
plan asks for a per-birth-cohort observability window and a primary analysis
restricted to fully covered cohorts. That requires individual-level birth years,
which Risteys does not publish. What is available is the calendar-year
distribution of first events per endpoint, which bounds the observation window
at the endpoint level. Endpoints whose events are concentrated at the start of
the recording window are flagged, because those are the ones where truncation
bites hardest.

**Resolution floor.** Risteys aggregates distribution bars to at least five
individuals, so a rare endpoint carries a cumulative incidence too coarse to
differentiate into a hazard. Those endpoints are excluded from the hazard
conversion rather than carried with a noisy derivative.

**Timestamp unreliability.** Electronic health record timestamps reflect adult
onset poorly, so T2 is also produced as an ordinal rank for the sensitivity arm
alongside the continuous value.

**Direction of bias.** All of this inflates apparent onset for genuinely
early-onset conditions, compresses the onset axis, and biases the H1 contrast
toward zero. If Gamma survives, it survives a conservative measurement. That is
recorded with the estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.ingest import risteys
from src.models import demography
from src.onset import tiers


def _spec() -> dict:
    return config.analysis()["onset"]["t2"]


def _histogram_span(block: list | None) -> tuple[float | None, float | None, int]:
    """Left edge, right edge and total count of a Risteys histogram."""
    if not block:
        return None, None, 0
    lefts = [bin_["interval"]["left"] for bin_ in block if bin_["interval"].get("left") is not None]
    rights = [
        bin_["interval"]["right"] for bin_ in block if bin_["interval"].get("right") is not None
    ]
    total = sum(bin_["count"] for bin_ in block)
    return (min(lefts) if lefts else None, max(rights) if rights else None, int(total))


def _first_bin_share(block: list | None) -> float | None:
    """Share of events falling in the earliest calendar bin.

    A high share means the endpoint's events pile up at the start of the
    recording window, which is the signature of left truncation rather than of
    a genuine early peak.
    """
    if not block:
        return None
    total = sum(bin_["count"] for bin_ in block)
    if total == 0:
        return None
    ordered = sorted(
        (bin_ for bin_ in block if bin_["interval"].get("left") is not None),
        key=lambda bin_: bin_["interval"]["left"],
    )
    return ordered[0]["count"] / total if ordered else None


def cif_series(record: dict, sex: str | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Age grid and cumulative incidence for one sex, or pooled across sexes."""
    series = record.get("cif") or []
    if sex is not None:
        series = [entry for entry in series if (entry.get("name") or "").lower() == sex]

    # An endpoint can carry a plot with no points, or one point, when it is too
    # rare for Risteys to publish a curve. Those contribute nothing and are
    # dropped rather than interpolated from.
    series = [entry for entry in series if len(entry.get("age") or []) >= 2]
    if not series:
        return None

    ages = sorted({age for entry in series for age in entry["age"]})
    if len(ages) < 2:
        return None

    grid = np.asarray(ages, dtype=float)
    stacked = [
        np.interp(
            grid,
            np.asarray(entry["age"], dtype=float),
            np.asarray(entry["cumulative_incidence"], dtype=float),
        )
        for entry in series
    ]
    return grid, np.mean(stacked, axis=0)


def hazard_from_record(record: dict, max_age: int) -> np.ndarray | None:
    """Age-specific hazard on a one-year grid, or None when unusable.

    The cumulative incidence is monotone by construction, so any decrease is
    interpolation noise rather than signal and is removed before differencing.
    """
    resolved = cif_series(record)
    if resolved is None:
        return None
    grid, cif = resolved
    if len(grid) < _spec()["min_cif_points"]:
        return None

    ages = np.arange(max_age + 1, dtype=float)
    interpolated = np.interp(ages, grid, cif, left=0.0, right=float(cif[-1]))
    interpolated = np.maximum.accumulate(interpolated)
    return demography.cif_to_hazard(interpolated)


def build(records: dict[str, dict] | None = None) -> pd.DataFrame:
    """One row per endpoint with the onset value and its usability flags."""
    records = records if records is not None else risteys.load_cache()
    spec = _spec()
    max_age = config.analysis()["fitness_cost"]["max_age"]

    rows = []
    for endpoint, record in records.items():
        counts = record.get("n_individuals") or {}
        medians = record.get("median_age_first_event") or {}
        histograms = record.get("histograms") or {}

        n_all = counts.get("all")
        median_all = medians.get("all")

        year_low, year_high, _ = _histogram_span(histograms.get("year"))
        age_low, age_high, age_total = _histogram_span(histograms.get("age"))

        hazard = hazard_from_record(record, max_age)

        rows.append(
            {
                "endpoint": endpoint,
                "longname": record.get("longname"),
                "efo_id": record.get("efo_id"),
                "doid": record.get("doid"),
                "n_cases": n_all,
                "n_cases_female": counts.get("female"),
                "n_cases_male": counts.get("male"),
                "median_age_first_event": median_all,
                "median_age_female": medians.get("female"),
                "median_age_male": medians.get("male"),
                "age_histogram_span_low": age_low,
                "age_histogram_span_high": age_high,
                "age_histogram_total": age_total,
                "record_year_low": year_low,
                "record_year_high": year_high,
                "first_year_bin_share": _first_bin_share(histograms.get("year")),
                "has_cif": bool(record.get("cif")),
                "hazard_usable": hazard is not None,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table["meets_case_floor"] = table["n_cases"].fillna(0) >= spec["min_cases"]
    table["has_onset"] = table["median_age_first_event"].notna()
    table["truncation_flag"] = (
        table["first_year_bin_share"].fillna(0) > spec["max_first_year_share"]
    )

    table["usable_for_onset"] = table["meets_case_floor"] & table["has_onset"]
    table["usable_for_hazard"] = table["usable_for_onset"] & table["hazard_usable"]

    usable = table["usable_for_onset"]
    table["onset_bin"] = pd.NA
    table.loc[usable, "onset_bin"] = tiers.bin_continuous(
        table.loc[usable, "median_age_first_event"]
    ).astype(object)

    # The ordinal sensitivity encoding. Health-record timestamps reflect adult
    # onset poorly, so the rank is carried alongside the continuous value rather
    # than instead of it.
    table["onset_rank"] = pd.NA
    table.loc[usable, "onset_rank"] = table.loc[usable, "median_age_first_event"].rank(
        method="average", pct=True
    )

    return table.sort_values("endpoint").reset_index(drop=True)


def coverage(table: pd.DataFrame) -> dict:
    return {
        "endpoints_retrieved": int(len(table)),
        "with_onset": int(table["has_onset"].sum()),
        "meeting_case_floor": int(table["meets_case_floor"].sum()),
        "usable_for_onset": int(table["usable_for_onset"].sum()),
        "usable_for_hazard": int(table["usable_for_hazard"].sum()),
        "with_efo": int(table["efo_id"].notna().sum()),
        "usable_with_efo": int((table["usable_for_onset"] & table["efo_id"].notna()).sum()),
        "flagged_for_truncation": int(table["truncation_flag"].sum()),
    }


def run() -> pd.DataFrame:
    table = build()
    if table.empty:
        raise SystemExit("no Risteys records cached; run src.ingest.risteys first")

    target = config.interim_dir() / "onset_t2.parquet"
    table.to_parquet(target, index=False)

    counts = coverage(table)
    print("T2 onset axis from Risteys")
    for key, value in counts.items():
        print(f"   {key.replace('_', ' ')}: {value:,}")

    usable = table[table["usable_for_onset"]]
    if not usable.empty:
        print()
        print("   median age at first event, distribution across usable endpoints")
        print(usable["median_age_first_event"].describe().to_string())
        print()
        print("   endpoints per decade bin")
        print(usable["onset_bin"].value_counts().sort_index().to_string())
    return table


def load() -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / "onset_t2.parquet")


if __name__ == "__main__":
    config.ensure_dirs()
    run()
