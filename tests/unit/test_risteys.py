"""Tests for the Risteys parser and the T2 onset construction.

The parser reads structured data out of a rendered page, which is the fragile
kind of source. These tests pin the things that would break silently: picking
the wrong table, losing the sex columns, leaving the cumulative incidence on a
percentage scale, and treating a rare endpoint with no curve as if it had one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.ingest import risteys
from src.onset import complex_onset

PAGE = """
<html>
<div class="title">
<h1 class="pt-4 pl-6">Major coronary heart disease event</h1>
<p class="links">
<a href="https://www.ebi.ac.uk/ols/search?q=11847&amp;ontology=doid">DOID</a>
<a href="https://www.ebi.ac.uk/gwas/efotraits/EFO_0001645">EFO</a>
</p>
</div>

<p>FinRegistry</p>
<table><thead><tr><th></th><th>All</th></tr></thead>
<tbody><tr><th>Number of individuals</th><td>1</td></tr></tbody></table>

<p>Endpoint definition steps</p>
<table><thead><tr><th>Step</th><th>Rule</th></tr></thead>
<tbody><tr><th>1</th><td>ICD-10</td></tr></tbody></table>

<p>FinnGen</p>
<table><thead><tr><th></th><th>All</th><th>Female</th><th>Male</th></tr></thead>
<tbody>
<tr><th>Number of individuals</th><td>90714</td><td>33175</td><td>57539</td></tr>
<tr><th>Unadjusted period prevalence (%)</th><td>21.02</td><td>11.77</td><td>26.36</td></tr>
<tr><th>Median age at first event (years)</th><td>66.12</td><td>67.57</td><td>65.29</td></tr>
</tbody></table>

<div data-histogram-x-axis-label="Age"
     data-histogram-values="{age_hist}"></div>
<div data-histogram-x-axis-label="Year"
     data-histogram-values="{year_hist}"></div>
<div data-cif-data="{cif}"></div>
</html>
"""


def _escape(payload) -> str:
    return json.dumps(payload).replace('"', "&quot;")


def build_page(cif=None, age_hist=None, year_hist=None) -> str:
    age_hist = age_hist or [
        {"count": 100, "interval": {"left": 40.0, "right": 50.0}},
        {"count": 300, "interval": {"left": 50.0, "right": 60.0}},
    ]
    year_hist = year_hist or [
        {"count": 50, "interval": {"left": 1990.0, "right": 2000.0}},
        {"count": 350, "interval": {"left": 2000.0, "right": 2010.0}},
    ]
    cif = (
        cif
        if cif is not None
        else [
            {
                "name": "female",
                "max_value": 45.06,
                "cumulinc": [{"age": float(a), "value": float(a)} for a in range(20, 80, 5)],
            }
        ]
    )
    return PAGE.format(age_hist=_escape(age_hist), year_hist=_escape(year_hist), cif=_escape(cif))


def test_the_finngen_table_is_chosen_over_finregistry_and_over_other_tables():
    record = risteys.parse_endpoint("I9_CHD", build_page())
    assert record["source"] == "finngen"
    assert record["n_individuals"]["all"] == 90714
    assert record["columns"] == ["all", "female", "male"]


def test_sex_columns_are_kept_apart():
    record = risteys.parse_endpoint("I9_CHD", build_page())
    assert record["median_age_first_event"] == {"all": 66.12, "female": 67.57, "male": 65.29}


def test_name_and_cross_references_are_extracted():
    record = risteys.parse_endpoint("I9_CHD", build_page())
    assert record["longname"] == "Major coronary heart disease event"
    assert record["efo_id"] == "EFO_0001645"
    assert record["doid"] == "DOID:11847"


def test_cumulative_incidence_is_converted_from_percent_to_fraction():
    record = risteys.parse_endpoint("I9_CHD", build_page())
    series = record["cif"][0]
    assert max(series["cumulative_incidence"]) < 1.0
    assert series["cumulative_incidence"][0] == pytest.approx(0.20)


def test_both_histograms_are_kept_and_keyed_by_axis():
    record = risteys.parse_endpoint("I9_CHD", build_page())
    assert set(record["histograms"]) == {"age", "year"}
    assert record["age_histogram"] is record["histograms"]["age"]


def test_a_curve_with_too_few_points_is_not_interpolated_from():
    page = build_page(cif=[{"name": "female", "cumulinc": [{"age": 40.0, "value": 1.0}]}])
    record = risteys.parse_endpoint("RARE", page)
    assert complex_onset.cif_series(record) is None


def test_an_empty_curve_does_not_raise():
    page = build_page(cif=[{"name": "female", "cumulinc": []}])
    record = risteys.parse_endpoint("RARE", page)
    assert complex_onset.cif_series(record) is None
    assert complex_onset.hazard_from_record(record, max_age=100) is None


def test_hazard_conversion_produces_a_bounded_series():
    record = risteys.parse_endpoint("I9_CHD", build_page())
    hazard = complex_onset.hazard_from_record(record, max_age=100)
    assert hazard is not None
    assert hazard.shape == (101,)
    assert np.all(hazard >= 0) and np.all(hazard <= 1)


def test_the_first_year_bin_share_flags_events_piled_at_the_window_start():
    truncated = [
        {"count": 900, "interval": {"left": 1990.0, "right": 2000.0}},
        {"count": 100, "interval": {"left": 2000.0, "right": 2010.0}},
    ]
    record = risteys.parse_endpoint("X", build_page(year_hist=truncated))
    share = complex_onset._first_bin_share(record["histograms"]["year"])
    assert share == pytest.approx(0.9)


def test_endpoints_below_the_case_floor_are_excluded_from_the_onset_axis():
    small = build_page().replace("<td>90714</td>", "<td>12</td>")
    records = {
        "BIG": risteys.parse_endpoint("BIG", build_page()),
        "SMALL": risteys.parse_endpoint("SMALL", small),
    }
    table = complex_onset.build(records).set_index("endpoint")
    assert bool(table.loc["BIG", "usable_for_onset"])
    assert not bool(table.loc["SMALL", "usable_for_onset"])


def test_onset_bins_follow_the_median_age():
    table = complex_onset.build({"BIG": risteys.parse_endpoint("BIG", build_page())}).set_index(
        "endpoint"
    )
    assert table.loc["BIG", "onset_bin"] == "60-70"


def test_a_page_with_no_key_figures_yields_an_empty_table_rather_than_raising():
    figures = risteys.parse_key_figures("<html><p>nothing here</p></html>")
    assert figures["rows"] == {}
    assert figures["columns"] == []
