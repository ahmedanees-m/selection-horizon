"""Onset tiers.

T1a  Orphanet average age of onset intervals. Primary Mendelian axis.
T1b  HPO onset sub-ontology. Secondary, used only for the concordance figure.
T2   Risteys median age at first event, subject to the registry measurement
     model in src/onset/measurement_model.py.
T3   Kuan et al. median age at first record, independent of T2.

A disease may carry several Orphanet intervals. The primary encoding is the
earliest, because selection acts on the earliest fitness-relevant manifestation
rather than on the average one. The sensitivity encoding is the midpoint of the
span between the earliest and the latest interval.
"""

from __future__ import annotations

import pandas as pd

from src import config

ALL_AGES = "All ages"


def interval_table() -> pd.DataFrame:
    """Interval names, ordinal ranks and midpoint ages.

    Orphanet writes the interval names in title case. Everything here is keyed
    on the lower-case form so that the analysis configuration and the upstream
    vocabulary cannot drift apart on capitalisation alone.
    """
    spec = config.analysis()["onset"]["t1a_intervals"]
    rows = [
        {"interval_key": name.lower(), "rank": values["rank"], "midpoint": values["midpoint"]}
        for name, values in spec.items()
    ]
    return pd.DataFrame(rows)


def _ranked(long: pd.DataFrame) -> pd.DataFrame:
    table = interval_table()
    merged = long.assign(interval_key=long["onset_interval"].str.lower()).merge(
        table, on="interval_key", how="left"
    )
    unknown = set(merged.loc[merged["rank"].isna(), "interval_key"].dropna()) - {ALL_AGES.lower()}
    if unknown:
        raise ValueError(f"onset intervals absent from config/analysis.yaml: {sorted(unknown)}")
    return merged[merged["rank"].notna()].copy()


def build_t1a(onset_long: pd.DataFrame) -> pd.DataFrame:
    """Collapse the Orphanet long table to one row per ORPHAcode.

    Diseases annotated only as "All ages" carry no ordinal position and are
    dropped from the continuous axis. They are retained in the coverage counts
    so that the record separates "no annotation" from "annotated as
    spanning every interval".
    """
    ranked = _ranked(onset_long)
    if ranked.empty:
        return pd.DataFrame(
            columns=[
                "orpha_code",
                "onset_earliest",
                "onset_rank",
                "onset_midpoint",
                "onset_latest_rank",
                "onset_span_midpoint",
                "n_intervals",
            ]
        )

    grouped = ranked.groupby("orpha_code")
    first = grouped["rank"].min()
    last = grouped["rank"].max()
    counts = grouped["rank"].count()

    table = interval_table().dropna(subset=["rank"]).set_index("rank")
    out = pd.DataFrame(
        {
            "orpha_code": first.index,
            "onset_rank": first.to_numpy(),
            "onset_latest_rank": last.to_numpy(),
            "n_intervals": counts.to_numpy(),
        }
    )
    out["onset_earliest"] = out["onset_rank"].map(table["interval_key"])
    out["onset_midpoint"] = out["onset_rank"].map(table["midpoint"])
    out["onset_span_midpoint"] = (
        out["onset_rank"].map(table["midpoint"]) + out["onset_latest_rank"].map(table["midpoint"])
    ) / 2.0
    return out.reset_index(drop=True)


def coverage(onset_long: pd.DataFrame) -> dict[str, int]:
    """Counts that the analysis set record reports."""
    total = onset_long["orpha_code"].nunique()
    annotated = onset_long.loc[onset_long["onset_interval"].notna(), "orpha_code"].nunique()
    all_ages_only = (
        onset_long.groupby("orpha_code")["onset_interval"]
        .apply(lambda values: set(values.dropna()) == {ALL_AGES})
        .sum()
    )
    ordinal = build_t1a(onset_long)["orpha_code"].nunique()
    return {
        "orphacodes_in_product": int(total),
        "with_any_onset_annotation": int(annotated),
        "all_ages_only": int(all_ages_only),
        "with_ordinal_onset": int(ordinal),
    }


def bin_continuous(ages: pd.Series) -> pd.Series:
    """Decade bins for the complex-disease tiers."""
    edges = config.analysis()["onset"]["t2_t3_bins"]
    labels = [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)]
    return pd.cut(ages, bins=edges, labels=labels, right=False)


def collapse_spec() -> dict:
    return config.analysis()["onset"]["collapse"]


def collapse(intervals: pd.Series) -> pd.Series:
    """Map the seven Orphanet intervals onto the five analysed levels.

    The collapse is fixed in config/analysis.yaml and applied before any score
    is inspected. Antenatal and neonatal merge because both lie far before
    first reproduction, so the distinction carries no selection-relevant
    information. Elderly merges into adult under the minimum-count rule.
    """
    mapping = collapse_spec()["map"]
    unknown = set(intervals.dropna().unique()) - set(mapping)
    if unknown:
        raise ValueError(f"onset intervals absent from the collapse map: {sorted(unknown)}")
    return intervals.map(mapping)


def collapsed_order() -> list[str]:
    return list(collapse_spec()["order"])


def collapsed_midpoint(bins: pd.Series) -> pd.Series:
    return bins.map(collapse_spec()["midpoints"])


def collapsed_rank(bins: pd.Series) -> pd.Series:
    order = {name: index for index, name in enumerate(collapsed_order())}
    return bins.map(order)


def check_collapse_counts(counts: pd.Series) -> list[str]:
    """Report any collapsed bin that still falls below the minimum count.

    The collapse map is pre-registered, so a bin that comes in short is
    reported rather than silently merged further. If this fires, the map is
    revised in the configuration, not in passing.
    """
    minimum = collapse_spec()["min_diseases"]
    return [
        f"{name}: {int(value)} diseases, below the {minimum} the collapse rule requires"
        for name, value in counts.items()
        if value < minimum
    ]
