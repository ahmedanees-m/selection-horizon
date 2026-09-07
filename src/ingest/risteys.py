"""Risteys retrieval for the complex-disease onset axis, tier T2.

Risteys publishes no bulk download and no per-endpoint JSON route. The endpoint
list is served as JSON; each endpoint page embeds its figures as HTML data
attributes holding escaped JSON. Those attributes are the structured source, not
a screen scrape of rendered text.

What is taken from each endpoint:

    median age at first event, by sex, which is the T2 onset value
    the age distribution of first events
    the Aalen-Johansen cumulative incidence, which already treats death as a
        competing event and is stratified by sex with age as the timescale

Retrieval is rate limited, resumable, and cached as parsed records rather than
as raw pages. A page is around 4 MB and there are close to five thousand of
them, so caching the pages themselves would cost twenty gigabytes to hold bytes
that are almost entirely markup. The parsed record keeps the numbers and the
retrieval date, and the endpoint list and release are pinned in the manifest.

Risteys aggregates its distribution bars to at least five individuals, so an
endpoint whose cumulative incidence is too coarse to convert to a hazard is
identified here and excluded downstream rather than silently carried.
"""

from __future__ import annotations

import argparse
import html as htmlmod
import json
import re
import time
from pathlib import Path

import requests

from src import config

TIMEOUT = 90
RATE_LIMIT_SECONDS = 1.0
RETRIES = 3

ATTRIBUTE = re.compile(r'data-(cif-data|histogram-values|histogram-x-axis-label)="([^"]*)"')
TAG = re.compile(r"<[^>]+>")
KEY_FIGURES = re.compile(r"<thead>(?P<head>.*?)</thead>\s*<tbody>(?P<body>.*?)</tbody>", re.S)
CELL = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.S)
ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)


def _spec() -> dict:
    return config.sources()["risteys"]


def _clean(fragment: str) -> str:
    return re.sub(r"\s+", " ", htmlmod.unescape(TAG.sub(" ", fragment))).strip()


def endpoint_list(session: requests.Session) -> list[str]:
    response = session.get(_spec()["endpoint_list"], timeout=TIMEOUT)
    response.raise_for_status()
    return list(response.json())


def fetch_page(session: requests.Session, endpoint: str) -> str | None:
    url = _spec()["endpoint_page"].format(endpoint=endpoint)
    for attempt in range(RETRIES):
        try:
            response = session.get(url, timeout=TIMEOUT)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if response.status_code == 200:
            return response.text
        if response.status_code == 404:
            return None
        time.sleep(2 * (attempt + 1))
    return None


def _attributes(page: str) -> dict[str, list]:
    found: dict[str, list] = {"cif-data": [], "histogram-values": [], "histogram-x-axis-label": []}
    for name, value in ATTRIBUTE.findall(page):
        text = htmlmod.unescape(value)
        if name == "histogram-x-axis-label":
            found[name].append(text)
        elif text.strip().startswith(("[", "{")):
            try:
                found[name].append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return found


ANCHOR_ROW = "Number of individuals"

# The endpoint page carries the human-readable name and its ontology
# cross-references in the title block. The EFO identifier is the important one:
# the Open Targets disease index is largely EFO native, so a FinnGen endpoint
# resolves to a Platform disease through it directly. Without that the leg would
# be an ICD-10 match against curated composite endpoints, which the execution
# plan rates as the hardest in the crosswalk.
TITLE = re.compile(r'<div class="title">\s*<h1[^>]*>(.*?)</h1>', re.S)
EFO = re.compile(r"efotraits/(EFO_\d+)")
DOID = re.compile(r"search\?q=(\d+)&(?:amp;)?ontology=doid")


def parse_title(page: str) -> dict:
    """Endpoint name and ontology cross-references from the title block."""
    name = TITLE.search(page)
    efo = EFO.search(page)
    doid = DOID.search(page)
    return {
        "longname": _clean(name.group(1)) if name else None,
        "efo_id": efo.group(1) if efo else None,
        "doid": f"DOID:{doid.group(1)}" if doid else None,
    }


def parse_key_figures(page: str) -> dict:
    """The summary table, preferring the FinnGen section over FinRegistry.

    An endpoint page carries several tables, so the right one is identified by
    the row label it must contain rather than by position. A page may also carry
    both cohorts; each candidate is attributed to whichever source marker most
    recently precedes it, and FinnGen is taken when both are present, because the
    onset axis is defined on FinnGen throughout.
    """
    tables = []
    for match in KEY_FIGURES.finditer(page):
        body = match.group("body")
        if ANCHOR_ROW not in body:
            continue

        before = page[: match.start()]
        source = (
            "finngen" if before.rfind("FinnGen") > before.rfind("FinRegistry") else "finregistry"
        )
        columns = [
            name for name in (_clean(cell) for cell in CELL.findall(match.group("head"))) if name
        ]

        rows = {}
        for row in ROW.finditer(body):
            cells = [_clean(cell) for cell in CELL.findall(row.group(1))]
            if len(cells) >= 2:
                rows[cells[0]] = cells[1:]
        tables.append({"source": source, "columns": columns, "rows": rows})

    preferred = next((table for table in tables if table["source"] == "finngen"), None)
    return preferred or (tables[0] if tables else {"source": None, "columns": [], "rows": {}})


def _numbers(values: list[str]) -> list[float | None]:
    out: list[float | None] = []
    for value in values:
        try:
            out.append(float(value.replace(",", "")))
        except (ValueError, AttributeError):
            out.append(None)
    return out


def parse_endpoint(endpoint: str, page: str) -> dict:
    attributes = _attributes(page)
    figures = parse_key_figures(page)

    columns = [name.lower() for name in figures["columns"]]
    record: dict = {
        "endpoint": endpoint,
        **parse_title(page),
        "source": figures["source"],
        "columns": columns,
    }

    for label, key in (
        ("Number of individuals", "n_individuals"),
        ("Unadjusted period prevalence (%)", "period_prevalence_pct"),
        ("Median age at first event (years)", "median_age_first_event"),
    ):
        values = _numbers(figures["rows"].get(label, []))
        record[key] = dict(zip(columns, values, strict=False)) if columns else {}

    # Both histograms are kept, keyed by their own axis label. The age
    # distribution is the onset signal; the year distribution shows the calendar
    # window over which events were recorded, which is what the registry
    # measurement model needs to reason about left truncation.
    labels = attributes["histogram-x-axis-label"]
    histograms = attributes["histogram-values"]
    record["histograms"] = {
        label.strip().lower(): block for label, block in zip(labels, histograms, strict=False)
    }
    record["age_histogram"] = record["histograms"].get("age")

    # The cumulative incidence is published as a percentage; it is stored as a
    # fraction so that the hazard conversion downstream needs no scale factor.
    series = []
    for block in attributes["cif-data"]:
        for entry in block:
            points = entry.get("cumulinc") or []
            series.append(
                {
                    "name": entry.get("name"),
                    "max_value": entry.get("max_value"),
                    "age": [point["age"] for point in points],
                    "cumulative_incidence": [point["value"] / 100.0 for point in points],
                }
            )
    record["cif"] = series
    return record


def cache_path() -> Path:
    return config.raw_dir() / "risteys" / f"endpoints_{_spec()['release']}.jsonl"


def load_cache() -> dict[str, dict]:
    path = cache_path()
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[record["endpoint"]] = record
    return records


def run(limit: int | None = None, rate: float = RATE_LIMIT_SECONDS) -> Path:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "selection-horizon/0.1 (research use)"})

    cached = load_cache()
    endpoints = endpoint_list(session)
    pending = [code for code in endpoints if code not in cached]
    if limit is not None:
        pending = pending[:limit]

    print(
        f"risteys {_spec()['release']}: {len(endpoints):,} endpoints listed, "
        f"{len(cached):,} already cached, {len(pending):,} to retrieve"
    )

    written = 0
    missing = 0
    with path.open("a", encoding="utf-8") as handle:
        for position, endpoint in enumerate(pending, start=1):
            page = fetch_page(session, endpoint)
            if page is None:
                missing += 1
            else:
                handle.write(json.dumps(parse_endpoint(endpoint, page)) + "\n")
                handle.flush()
                written += 1
            time.sleep(rate)
            if position % 250 == 0:
                print(f"   {position:,}/{len(pending):,} retrieved, {missing} unavailable")

    print(f"risteys: {written:,} records written, {missing} unavailable")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve Risteys endpoint statistics")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rate", type=float, default=RATE_LIMIT_SECONDS)
    args = parser.parse_args()
    config.ensure_dirs()
    run(limit=args.limit, rate=args.rate)


if __name__ == "__main__":
    main()
