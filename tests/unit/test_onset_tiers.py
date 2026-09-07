"""Tests for onset tier construction."""

from __future__ import annotations

import pandas as pd
import pytest

from src.onset import tiers


def long_frame(rows: list[tuple[int, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["orpha_code", "onset_interval"])


def test_earliest_interval_wins_for_a_multi_label_disease():
    frame = long_frame([(1, "Neonatal"), (1, "Adult"), (1, "Childhood")])
    result = tiers.build_t1a(frame)
    assert result.loc[0, "onset_earliest"] == "neonatal"
    assert result.loc[0, "n_intervals"] == 3


def test_span_midpoint_sits_between_the_earliest_and_latest_midpoints():
    frame = long_frame([(1, "Neonatal"), (1, "Adult")])
    result = tiers.build_t1a(frame)
    assert result.loc[0, "onset_midpoint"] < result.loc[0, "onset_span_midpoint"]
    assert result.loc[0, "onset_span_midpoint"] < 40.0


def test_orphanet_title_case_matches_the_configured_intervals():
    """Orphanet writes Title Case; the configuration is lower case."""
    frame = long_frame([(1, "Antenatal"), (2, "ELDERLY"), (3, "adolescent")])
    result = tiers.build_t1a(frame)
    assert set(result["onset_earliest"]) == {"antenatal", "elderly", "adolescent"}


def test_all_ages_carries_no_ordinal_position():
    frame = long_frame([(1, "All ages"), (2, "Adult")])
    result = tiers.build_t1a(frame)
    assert set(result["orpha_code"]) == {2}


def test_unknown_interval_raises_rather_than_dropping_silently():
    frame = long_frame([(1, "Middle age")])
    with pytest.raises(ValueError, match="absent from config"):
        tiers.build_t1a(frame)


def test_coverage_separates_no_annotation_from_all_ages():
    frame = long_frame([(1, None), (2, "All ages"), (3, "Adult"), (3, "Elderly")])
    counts = tiers.coverage(frame)
    assert counts["orphacodes_in_product"] == 3
    assert counts["with_any_onset_annotation"] == 2
    assert counts["all_ages_only"] == 1
    assert counts["with_ordinal_onset"] == 1


def test_ranks_are_strictly_ordered_across_the_interval_vocabulary():
    table = tiers.interval_table().dropna(subset=["rank"]).sort_values("rank")
    assert list(table["interval_key"]) == [
        "antenatal",
        "neonatal",
        "infancy",
        "childhood",
        "adolescent",
        "adult",
        "elderly",
    ]
    assert table["midpoint"].is_monotonic_increasing


def test_continuous_bins_are_left_closed():
    binned = tiers.bin_continuous(pd.Series([0.0, 9.9, 10.0, 65.0]))
    assert list(binned.astype(str)) == ["0-10", "0-10", "10-20", "60-70"]
