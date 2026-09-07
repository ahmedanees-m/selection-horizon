"""Independent verification of every source the analysis depends on.

Provenance discipline is only worth something if it is checked. This module
answers, for each source, four questions that a reader should not have to take
on trust.

    Does the URL still resolve, and to the version claimed?
    Does the identifier or DOI point at what it is said to point at?
    Do the retrieved bytes still hash to what was recorded?
    Does the parsed content look like the thing it claims to be?

The last is the one that matters most and is the easiest to skip. A file can
download cleanly, hash correctly, and still be the wrong release, or carry
identifiers in a format the joins will silently miss. Every check below either
passes with the observed value attached or fails with the discrepancy stated.

Run as `python -m src.diagnostics.verify_sources`. Network checks can be skipped with
`--offline` when only the local content checks are wanted.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from src import config
from src.ingest import manifest

TIMEOUT = 60

# Identifier shapes the joins depend on. A value that does not match is not a
# formatting nuisance; it is a row that will silently fail to join.
IDENTIFIER_PATTERNS = {
    "ensembl_gene": re.compile(r"^ENSG\d{11}$"),
    "hgnc_numeric": re.compile(r"^\d+$"),
    "orpha": re.compile(r"^\d+$"),
    "efo": re.compile(r"^EFO_\d{7}$"),
    "mondo": re.compile(r"^MONDO_\d{7}$"),
    "hpo": re.compile(r"^HP:\d{7}$"),
    "doid": re.compile(r"^DOID:\d+$"),
}

# Load-bearing citations. The research plan flags several as not independently
# verified; these are resolved against Crossref and the returned title is
# compared with the recorded description.
CITATIONS = {
    "10.1038/s41586-024-07316-0": "Refining the impact of genetic evidence on clinical success",
    "10.1038/s41586-020-2308-7": (
        "The mutational constraint spectrum quantified from variation in 141,456 humans"
    ),
    "10.1038/s41588-024-01820-9": (
        "Bayesian estimation of gene constraint from an evolutionary model with gene features"
    ),
    "10.1038/s41586-025-09703-7": (
        "Specificity, length and luck drive gene rankings in association studies"
    ),
    "10.1038/s43587-021-00051-5": "Common genetic associations between age-related diseases",
    "10.1038/s41576-025-00904-4": "Genomics of drug target prioritization",
    "10.1038/s41588-024-01854-z": (
        "Genetic factors associated with reasons for clinical trial stoppage"
    ),
    "10.1016/S2589-7500(19)30012-3": (
        "A chronological map of 308 physical and mental health conditions"
    ),
    "10.1093/gbe/evaa013": ("BetaScan2: Standardized Statistics to Detect Balancing Selection"),
}

ZENODO_RECORDS = {
    "10403680": "s_het estimates from GeneBayes",
    "10783210": "genetic_support",
    "7842447": "BetaScan2: Standardized Statistics to Detect Balancing Selection",
}


@dataclass
class Check:
    name: str
    passed: bool
    observed: str
    expected: str = ""
    note: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, observed, expected="", note="") -> None:
        self.checks.append(Check(name, bool(passed), str(observed), str(expected), str(note)))

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "check": check.name,
                    "result": "pass" if check.passed else "FAIL",
                    "observed": check.observed,
                    "expected": check.expected,
                    "note": check.note,
                }
                for check in self.checks
            ]
        )


def _iter_urls() -> Iterator[tuple[str, str]]:
    """Every URL in the pinned source configuration, with the path that holds it."""

    def walk(node, trail):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"url", "endpoint_list", "base_url"} and isinstance(value, str):
                    yield ".".join([*trail, key]), value
                else:
                    yield from walk(value, [*trail, str(key)])
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from walk(value, [*trail, str(index)])

    yield from walk(config.sources(), [])


def check_urls(report: Report) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "selection-horizon/0.1 (verification)"})

    for path, url in _iter_urls():
        try:
            response = session.head(url, timeout=TIMEOUT, allow_redirects=True)
            if response.status_code >= 400:
                response = session.get(url, timeout=TIMEOUT, stream=True)
                response.close()
            code = response.status_code
        except requests.RequestException as error:
            report.add(f"url {path}", False, f"unreachable: {type(error).__name__}", "HTTP 200")
            continue
        report.add(f"url {path}", code < 400, f"HTTP {code}", "HTTP < 400", url)


def check_citations(report: Report) -> None:
    """Resolve each load-bearing DOI and compare the returned title."""
    for doi, expected in CITATIONS.items():
        try:
            response = requests.get(
                f"https://api.crossref.org/works/{doi}",
                timeout=TIMEOUT,
                headers={"User-Agent": "selection-horizon/0.1 (mailto:research)"},
            )
            response.raise_for_status()
            message = response.json()["message"]
        except (requests.RequestException, KeyError, ValueError) as error:
            report.add(f"doi {doi}", False, f"unresolved: {type(error).__name__}", expected)
            continue

        title = (message.get("title") or [""])[0]
        year = None
        for key in ("published-print", "published-online", "issued"):
            parts = message.get(key, {}).get("date-parts") or [[None]]
            if parts and parts[0] and parts[0][0]:
                year = parts[0][0]
                break

        stem = expected.lower()[:40]
        matches = stem in title.lower()
        report.add(
            f"doi {doi}",
            matches,
            f"{title[:70]} ({year})",
            expected[:70],
            "" if matches else "title does not match the work this DOI is cited as",
        )


def check_zenodo(report: Report) -> None:
    for record, expected in ZENODO_RECORDS.items():
        try:
            response = requests.get(f"https://zenodo.org/api/records/{record}", timeout=TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            report.add(f"zenodo {record}", False, f"unresolved: {type(error).__name__}", expected)
            continue

        title = payload.get("metadata", {}).get("title", "")
        doi = payload.get("doi", "")
        matches = expected.lower()[:20] in title.lower()
        report.add(
            f"zenodo {record}",
            matches,
            f"{title[:60]} | doi {doi}",
            expected,
            "" if matches else "record title does not match the dataset it is cited as",
        )


def _orphanet_header(path: Path) -> tuple[str | None, str | None, int | None]:
    """Version, date and declared disorder count from the product header."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        head = handle.read(4000)
    version = re.search(r'version="([^"]+)"', head)
    date = re.search(r'date="([^"]+)"', head)
    count = re.search(r'<DisorderList count="(\d+)"', head)
    return (
        version.group(1) if version else None,
        date.group(1) if date else None,
        int(count.group(1)) if count else None,
    )


def check_versions(report: Report) -> None:
    """Version claims read off the retrieved files, not off the configuration."""
    raw = config.raw_dir()

    for name in ("en_product9_ages.xml", "en_product6.xml", "en_product1.xml", "en_product4.xml"):
        path = raw / "orphanet" / name
        if not path.exists():
            report.add(f"orphanet {name}", False, "absent", "present")
            continue
        version, date, declared = _orphanet_header(path)
        report.add(
            f"orphanet {name} version",
            version is not None,
            f"{version} dated {date}, declares {declared} disorders",
            "a version and date in the header",
        )

    obo = raw / "hpo" / "hp.obo"
    if obo.exists():
        head = obo.read_text(encoding="utf-8", errors="replace")[:2000]
        release = re.search(r"data-version:\s*(\S+)", head)
        report.add(
            "hpo release",
            release is not None,
            release.group(1) if release else "no data-version line",
            "a data-version line",
        )
    else:
        report.add("hpo release", False, "hp.obo absent", "present")

    gencode = raw / "gencode" / "gencode.v44.pc_transcripts.fa.gz"
    if gencode.exists():
        with gzip.open(gencode, "rt", encoding="utf-8", errors="replace") as handle:
            first = handle.readline()
        report.add(
            "gencode transcript header shape",
            "|CDS:" in first,
            first.strip()[:80],
            "a header carrying a CDS interval",
        )
    else:
        report.add("gencode transcript file", False, "absent", "present")

    release = config.sources()["open_targets"]["release"]
    base = config.sources()["open_targets"]["base_url"]
    report.add(
        "open targets release pinned in the base url",
        release in base,
        base,
        f"a base url containing {release}",
    )

    # The Kuan appendix is a PDF, so the parse is the weak point rather than the
    # download. The article's own title states the number of conditions, which
    # gives the parse an external number to agree with: both tables must hold
    # exactly that many, and a parse that has drifted onto the neighbouring
    # table or dropped a wrapped row will not.
    appendix = raw / "kuan_2019" / "mmc1.pdf"
    if appendix.exists():
        import pymupdf

        from src.onset import kuan

        with pymupdf.open(appendix) as document:
            counts = {
                "S1": len(kuan.read_conditions(document)),
                "S7A": len(kuan.read_ages(document)),
            }
        for table, count in counts.items():
            report.add(
                f"kuan appendix table {table} row count",
                count == kuan.EXPECTED_CONDITIONS,
                f"{count} conditions",
                f"{kuan.EXPECTED_CONDITIONS}, the count the article reports",
            )
    else:
        report.add("kuan appendix", False, "absent", "present")


def _identifier_report(values: pd.Series, kind: str) -> tuple[int, int, list[str]]:
    pattern = IDENTIFIER_PATTERNS[kind]
    present = values.dropna().astype(str)
    bad = present[~present.str.match(pattern)]
    return len(present), len(bad), bad.head(3).tolist()


def check_identifiers(report: Report) -> None:
    """Identifier formats in every table the joins run on."""
    interim = config.interim_dir()

    targets = [
        ("gene_table.parquet", "ensembl_gene_id", "ensembl_gene"),
        ("gene_table.parquet", "hgnc_id", "hgnc_numeric"),
        ("evolutionary_features.parquet", "ensembl_gene_id", "ensembl_gene"),
        ("baseline_features.parquet", "ensembl_gene_id", "ensembl_gene"),
        ("pleiotropy.parquet", "ensembl_gene_id", "ensembl_gene"),
        ("orphanet/orphanet_phenotypes.parquet", "hpo_id", "hpo"),
        ("hpo_annotations.parquet", "hpo_id", "hpo"),
    ]

    for filename, column, kind in targets:
        path = interim / filename
        if not path.exists():
            report.add(f"identifiers {filename}:{column}", False, "table absent", "present")
            continue
        frame = pd.read_parquet(path, columns=[column])
        total, bad, examples = _identifier_report(frame[column], kind)
        report.add(
            f"identifiers {filename}:{column}",
            bad == 0,
            f"{total:,} values, {bad} malformed",
            f"all matching {IDENTIFIER_PATTERNS[kind].pattern}",
            ", ".join(examples),
        )

    t2 = interim / "onset_t2.parquet"
    if t2.exists():
        total, bad, examples = _identifier_report(
            pd.read_parquet(t2, columns=["doid"])["doid"], "doid"
        )
        report.add(
            "identifiers onset_t2:doid",
            bad == 0,
            f"{total:,} values, {bad} malformed",
            "all matching the DOID shape",
            ", ".join(examples),
        )


def check_crosswalk(report: Report) -> None:
    """Whether the FinnGen identifiers actually resolve, not merely parse.

    Format is the weaker test. A FinnGen endpoint can carry a well-formed EFO
    identifier that is obsolete, or one that points at a metabolite measurement
    rather than a disease. Both parse; neither is usable.
    """
    path = config.interim_dir() / "crosswalk" / "finngen_to_platform.parquet"
    if not path.exists():
        report.add("finngen crosswalk", False, "not built", "present")
        return

    table = pd.read_parquet(path)
    resolved = int(table["resolved"].sum())
    confident = int(table["confident"].sum())
    measurements = int(table["target_is_measurement"].sum())

    report.add(
        "finngen EFO identifiers resolve to a Platform term",
        resolved / max(len(table), 1) > 0.5,
        f"{resolved:,} of {len(table):,} resolve",
        "more than half",
        "direct joins reach only a small minority; the rest arrive through the "
        "obsolete-term replacement the Platform publishes",
    )
    report.add(
        "finngen mappings survive a name check",
        confident > 0,
        f"{confident:,} confident, {len(table) - resolved:,} unresolved, "
        f"{measurements:,} pointing at a measurement",
        "a confident tier that is not empty",
    )
    report.add(
        "mis-annotated FinnGen endpoints are excluded rather than carried",
        bool((~table.loc[table["target_is_measurement"], "confident"]).all()),
        f"{measurements:,} measurement targets, none marked confident",
        "none carried into the analysis",
    )


def check_value_ranges(report: Report) -> None:
    """Values that must lie in a known range if the parse was correct."""
    interim = config.interim_dir()

    rules = [
        ("gene_table.parquet", "s_het", 0.0, 1.0, "a selection coefficient"),
        ("gene_table.parquet", "gc_content", 0.0, 1.0, "a base composition fraction"),
        ("gene_table.parquet", "loeuf", 0.0, 10.0, "an observed to expected ratio bound"),
        ("gene_table.parquet", "pli", 0.0, 1.0, "a probability"),
        (
            "conservation_phastcons_100way.parquet",
            "phastcons_100way_mean",
            0.0,
            1.0,
            "a posterior probability",
        ),
        (
            "conservation_phylop_241way.parquet",
            "phylop_241way_fraction_above",
            0.0,
            1.0,
            "a fraction of bases",
        ),
        ("severity_index.parquet", "kappa_d", 0.0, 1.0, "a fraction of reproduction removed"),
        ("onset_t2.parquet", "median_age_first_event", 0.0, 120.0, "an age in years"),
        ("onset_t3.parquet", "median_age_both_sexes", 0.0, 120.0, "an age in years"),
        (
            "balancing_selection.parquet",
            "b2_n_sites_CEU",
            1.0,
            1e9,
            "a count of scored sites inside a gene span",
        ),
        ("demography_modern.parquet", "survivorship", 0.0, 1.0, "a survival probability"),
    ]

    for filename, column, low, high, meaning in rules:
        path = interim / filename
        if not path.exists():
            report.add(f"range {filename}:{column}", False, "table absent", "present")
            continue
        frame = pd.read_parquet(path, columns=[column])
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            report.add(f"range {filename}:{column}", False, "no values", f"[{low}, {high}]")
            continue
        outside = int(((values < low) | (values > high)).sum())
        report.add(
            f"range {filename}:{column}",
            outside == 0,
            f"[{values.min():.4g}, {values.max():.4g}], {outside} outside",
            f"[{low}, {high}], {meaning}",
        )


def check_cross_source(report: Report) -> None:
    """Overlaps between sources that must be large if the keys are right."""
    interim = config.interim_dir()

    gene_table = interim / "gene_table.parquet"
    if gene_table.exists():
        genes = pd.read_parquet(gene_table, columns=["ensembl_gene_id", "s_het", "loeuf"])
        both = genes.dropna(subset=["s_het", "loeuf"])
        report.add(
            "overlap s_het and gnomAD constraint",
            len(both) > 15000,
            f"{len(both):,} genes carry both",
            "more than 15,000",
        )

    evolutionary = interim / "evolutionary_features.parquet"
    if evolutionary.exists():
        frame = pd.read_parquet(evolutionary, columns=["s_het_log", "phylop_241way_fraction_above"])
        both = frame.dropna()
        report.add(
            "overlap constraint and conservation",
            len(both) > 15000,
            f"{len(both):,} genes carry both primary scores",
            "more than 15,000",
        )

    onset = interim / "orphanet" / "orphanet_onset.parquet"
    genes_file = interim / "orphanet" / "orphanet_genes.parquet"
    if onset.exists() and genes_file.exists():
        onset_codes = set(pd.read_parquet(onset, columns=["orpha_code"])["orpha_code"].dropna())
        gene_codes = set(pd.read_parquet(genes_file, columns=["orpha_code"])["orpha_code"].dropna())
        report.add(
            "overlap Orphanet onset and gene products",
            len(onset_codes & gene_codes) > 3000,
            f"{len(onset_codes & gene_codes):,} ORPHAcodes in both",
            "more than 3,000",
        )


def check_manifest_integrity(report: Report) -> None:
    entries = manifest.load()["entries"]
    retained = {key: entry for key, entry in entries.items() if entry.get("retained", True)}
    released = {key: entry for key, entry in entries.items() if not entry.get("retained", True)}

    failures = manifest.verify(strict=False)
    report.add(
        "manifest hashes",
        not failures,
        f"{len(retained)} retained files, {len(failures)} mismatched or missing",
        "no mismatches",
        "; ".join(failures[:3]),
    )

    without_provenance = [
        key
        for key, entry in released.items()
        if not entry.get("upstream_md5") and not entry.get("sha256")
    ]
    report.add(
        "released files carry an upstream checksum",
        not without_provenance,
        f"{len(released)} released, {len(without_provenance)} without a checksum",
        "every released file carries one",
        "; ".join(without_provenance[:3]),
    )


def run(offline: bool = False) -> Report:
    report = Report()

    check_versions(report)
    check_identifiers(report)
    check_crosswalk(report)
    check_value_ranges(report)
    check_cross_source(report)
    check_manifest_integrity(report)

    if not offline:
        check_urls(report)
        check_zenodo(report)
        check_citations(report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify every source the analysis depends on")
    parser.add_argument("--offline", action="store_true", help="skip network checks")
    args = parser.parse_args()

    config.ensure_dirs()
    report = run(offline=args.offline)

    frame = report.to_frame()
    out_dir = config.derived_dir() / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "source_verification.csv", index=False)
    with (out_dir / "source_verification.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated": datetime.now(UTC).isoformat(timespec="seconds"),
                "checks": frame.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    with pd.option_context("display.max_colwidth", 78, "display.width", 200):
        print(frame.to_string(index=False))

    print()
    print(f"{len(report.checks) - len(report.failures)} of {len(report.checks)} checks passed")
    if report.failures:
        print()
        print("Failures")
        for check in report.failures:
            print(f"   {check.name}: {check.observed} (expected {check.expected}) {check.note}")


if __name__ == "__main__":
    main()
