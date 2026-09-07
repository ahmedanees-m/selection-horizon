"""T3: the independent complex-disease onset axis.

T2 puts the complex arm on an onset axis using Finnish health-registry records.
That axis is one health system, one population and one set of endpoint
definitions, so on its own it cannot say whether the ordering it produces is a
property of disease or a property of Finnish record keeping. T3 is the external
check: median age of first record for 308 conditions across roughly 3.9 million
individuals in the English National Health Service, from a different registry, a
different population and an independently curated phenotype library.

The two axes are not interchangeable and T3 does not replace T2 in the primary
model. T2 carries the gene-level annotations through the FinnGen crosswalk and
T3 does not, so T3 earns its place as a concordance figure rather than as a
second estimand. What the concordance answers is narrow and worth stating
plainly. If the two registries order the same conditions the same way, the
ordering is a property of the diseases. If they do not, nothing downstream of
the complex onset axis should be read as a property of disease.

The source is the article's supplementary appendix, deposited with the
open-access article and served by the Europe PMC supplementary-file endpoint.
Two of its tables are read. Table S1 carries the full condition name against the
abbreviated term the rest of the appendix uses, and table S7A carries the median
and interquartile range of age at first record. They are read separately and
joined on the abbreviated term, which also gives the parse a check: both tables
are expected to hold exactly the 308 conditions the article describes, and a
parse returning any other count is a parse that has gone wrong.

Two limits travel with every result from this tier. Age at first record is not
age of onset, which is the limit T2 carries as well. And the conditions are
CALIBER phenotypes rather than ontology terms, so the join to T2 is on disease
names and inherits the imprecision of name matching.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.ingest import fetch
from src.onset import donertas, tiers

APPENDIX_MEMBER = "mmc1.pdf"
EXPECTED_CONDITIONS = 308

CONDITIONS_TABLE = "Supplementary Table S1."
CONDITIONS_END = "Supplementary Table S2."
AGES_TABLE = "Supplementary Table S7A."
AGES_END = "Supplementary Table S7B."
CONDITIONS_HEADER = "Condition"

# Median and interquartile range arrive in one cell, as "63     (52, 72)". A
# condition with too few cases in one sex is released as "NA     (NA, NA)"
# rather than omitted, so the pattern has to accept it.
CELL = re.compile(r"^(\d+(?:\.\d+)?|NA)\s*\((\d+(?:\.\d+)?|NA),\s*(\d+(?:\.\d+)?|NA)\)$")

# The appendix encodes an en dash in a way the text extractor cannot resolve, so
# it arrives as the replacement character. Both tables carry it inside the same
# condition names and the join between them is on those names, so it is
# normalised identically on both sides rather than left to chance.
UNRESOLVED_CHARACTER = "\ufffd"


# The appendix spells ten conditions differently in its two tables, so the join
# on the abbreviated term leaves ten orphans on each side. Each S7A spelling
# pairs with exactly one S1 spelling and the two sets are exhausted by these ten
# pairs, which is what makes the map forced rather than chosen. It is written
# out rather than resolved by string similarity, because a similarity match that
# is wrong relabels a condition silently. Two pairs are more than punctuation:
# S7A shortens uterovaginal prolapse to prolapse, and names Bell's palsy as
# Bell's syndrome.
TERM_ALIASES = {
    "Barretts Oesophagus": "Barrett's Oesophagus",
    "Bells Syndrome": "Bell's Palsy",
    "Downs Syndrome": "Down Syndrome",
    "Low HDL": "Low HDL-C",
    "Menieres Disease": "Meniere's Disease",
    "Parkinsons Disease": "Parkinson's Disease",
    "Prolapse": "Uterovaginal Prolapse",
    "Raised LDL": "Raised LDL-C",
    "Scelroderma": "Scleroderma",
    "Sepsis": "Septicaemia",
}


def _spec() -> dict:
    return config.sources()["kuan_2019"]


def appendix() -> Path:
    """The supplementary appendix, out of the Europe PMC supplementary package."""
    spec = _spec()
    entry = spec["files"]["appendix"]
    root = config.raw_dir() / "kuan_2019"
    package = fetch.download(
        entry["url"],
        root / "supplementary_files.zip",
        source="kuan_2019",
        version=spec["version"],
        licence=spec["licence"],
        redistribute=spec["redistribute"],
    )

    target = root / APPENDIX_MEMBER
    if not target.exists():
        with zipfile.ZipFile(package) as archive:
            if APPENDIX_MEMBER not in archive.namelist():
                raise ValueError(
                    f"{APPENDIX_MEMBER} is not in the supplementary package, "
                    f"which holds {sorted(archive.namelist())}"
                )
            with archive.open(APPENDIX_MEMBER) as source, target.open("wb") as handle:
                handle.write(source.read())
    return target


def _clean(value: object) -> str:
    text = str(value or "").replace(UNRESOLVED_CHARACTER, "-")
    return re.sub(r"\s+", " ", text).strip()


def _caption_position(document, caption: str) -> tuple[int, float] | None:
    """Page index and vertical position of a caption, or None if it is absent."""
    for index in range(document.page_count):
        found = document[index].search_for(caption)
        if found:
            return index, min(rect.y0 for rect in found)
    return None


def _rows(document, caption: str, following: str) -> list[list[str]]:
    """Every row of one table, bounded by its own caption and the next one.

    Bounding on captions rather than on page numbers keeps the parse stable if
    the appendix is repaginated, and bounding on the vertical position as well
    as the page is what stops the next table being read as a continuation of
    this one when both sit on the same page.
    """
    start = _caption_position(document, caption)
    if start is None:
        raise ValueError(f"{caption} is not in the appendix")
    start_page, start_y = start

    end = _caption_position(document, following)
    end_page, end_y = end if end is not None else (document.page_count - 1, None)

    extracted = []
    for index in range(start_page, end_page + 1):
        for table in document[index].find_tables().tables:
            top = table.bbox[1]
            if index == start_page and top < start_y:
                continue
            if index == end_page and end_y is not None and top >= end_y:
                continue
            for row in table.extract():
                extracted.append([_clean(cell) for cell in row])
    return extracted


def read_conditions(document) -> pd.DataFrame:
    """Table S1: full condition name, abbreviated term and disease category."""
    rows = [
        row
        for row in _rows(document, CONDITIONS_TABLE, CONDITIONS_END)
        if len(row) == 3 and row[0] and row[0] != CONDITIONS_HEADER
    ]
    table = pd.DataFrame(rows, columns=["condition", "abbreviated_term", "category_full"])
    return table.drop_duplicates("abbreviated_term").reset_index(drop=True)


def read_ages(document) -> pd.DataFrame:
    """Table S7A: median and interquartile range of age at first record."""
    rows = [
        row for row in _rows(document, AGES_TABLE, AGES_END) if len(row) == 5 and CELL.match(row[2])
    ]

    records = []
    for abbreviated, category, *cells in rows:
        record: dict[str, object] = {"abbreviated_term": abbreviated, "category": category}
        for name, cell in zip(("both_sexes", "female", "male"), cells, strict=True):
            match = CELL.match(cell)
            values = match.groups() if match else ("NA", "NA", "NA")
            median, low, high = (np.nan if value == "NA" else float(value) for value in values)
            record[f"median_age_{name}"] = median
            if name == "both_sexes":
                record["iqr_low"] = low
                record["iqr_high"] = high
        records.append(record)
    return pd.DataFrame(records).drop_duplicates("abbreviated_term").reset_index(drop=True)


def build() -> pd.DataFrame:
    import pymupdf

    with pymupdf.open(appendix()) as document:
        conditions = read_conditions(document)
        ages = read_ages(document)

    for name, table in (("S1", conditions), ("S7A", ages)):
        if len(table) != EXPECTED_CONDITIONS:
            raise ValueError(
                f"table {name} parsed to {len(table)} conditions, not the "
                f"{EXPECTED_CONDITIONS} the article reports"
            )

    unused = set(TERM_ALIASES) - set(ages["abbreviated_term"])
    if unused:
        raise ValueError(f"the alias map carries terms S7A does not use: {sorted(unused)}")
    ages = ages.assign(abbreviated_term=ages["abbreviated_term"].replace(TERM_ALIASES))

    table = ages.merge(conditions, on="abbreviated_term", how="left")
    unmatched = table.loc[table["condition"].isna(), "abbreviated_term"].tolist()
    if unmatched:
        raise ValueError(f"conditions in S7A with no full name in S1: {sorted(unmatched)}")

    table["has_onset"] = table["median_age_both_sexes"].notna()
    table["onset_bin"] = pd.NA
    table.loc[table["has_onset"], "onset_bin"] = tiers.bin_continuous(
        table.loc[table["has_onset"], "median_age_both_sexes"]
    ).astype(object)

    # The sex difference is carried rather than averaged away. A condition whose
    # median age differs by a decade between the sexes is one whose single
    # median is a poor summary, and the concordance should be able to say so.
    table["sex_difference"] = table["median_age_female"] - table["median_age_male"]
    table["single_sex"] = table["median_age_female"].isna() | table["median_age_male"].isna()
    return table.sort_values("condition").reset_index(drop=True)


def concordance(t3: pd.DataFrame, threshold: float = donertas.MATCH_THRESHOLD) -> pd.DataFrame:
    """Kuan conditions matched to Risteys endpoints, for the T2 to T3 figure.

    The match is on disease names, using the token overlap and kind veto the
    Donertas cross-check uses, because the two problems are the same problem:
    free-text condition names against curated registry endpoints.
    """
    from src.onset import complex_onset

    t2 = complex_onset.load()
    usable = t2[t2["usable_for_onset"] & t2["longname"].notna()]
    if usable.empty:
        return pd.DataFrame(columns=["condition", "endpoint", "longname", "score"])

    endpoints = list(
        zip(usable["endpoint"], usable["longname"], usable["median_age_first_event"], strict=True)
    )

    rows = []
    for _, entry in t3[t3["has_onset"]].iterrows():
        name = entry["condition"]
        best = max(endpoints, key=lambda item: donertas.similarity(name, item[1]))
        score = donertas.similarity(name, best[1])
        if score < threshold:
            continue
        rows.append(
            {
                "condition": name,
                "category": entry["category"],
                "endpoint": best[0],
                "longname": best[1],
                "score": round(score, 3),
                "kuan_median_age": entry["median_age_both_sexes"],
                "risteys_median_age": best[2],
            }
        )

    matched = pd.DataFrame(rows)
    if matched.empty:
        return matched

    # One Kuan condition can be the best match for several endpoints and the
    # reverse, so the higher-scoring match wins on each side and the concordance
    # is computed over distinct pairs.
    matched = matched.sort_values("score", ascending=False)
    matched = matched.drop_duplicates("endpoint").drop_duplicates("condition")
    return matched.sort_values("condition").reset_index(drop=True)


def concordance_statistics(matched: pd.DataFrame) -> dict:
    from scipy import stats

    if len(matched) < 3:
        return {"n_matched": len(matched), "note": "too few matches to correlate"}

    spearman = stats.spearmanr(matched["kuan_median_age"], matched["risteys_median_age"])
    pearson = stats.pearsonr(matched["kuan_median_age"], matched["risteys_median_age"])
    difference = matched["kuan_median_age"] - matched["risteys_median_age"]
    return {
        "n_matched": int(len(matched)),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson": float(pearson.statistic),
        "median_difference_years": float(difference.median()),
        "mean_absolute_difference_years": float(difference.abs().mean()),
    }


def run() -> pd.DataFrame:
    table = build()
    target = config.interim_dir() / "onset_t3.parquet"
    table.to_parquet(target, index=False)

    with_onset = table[table["has_onset"]]
    print(f"onset_t3: {len(table):,} conditions, {len(with_onset):,} with a median age")
    print()
    print("   median age of first record, across conditions")
    print(with_onset["median_age_both_sexes"].describe().to_string())
    print()
    print("   conditions per decade bin")
    print(with_onset["onset_bin"].value_counts().sort_index().to_string())

    both = with_onset[~with_onset["single_sex"]]
    print()
    print(f"   conditions recorded in both sexes: {len(both):,}")
    print(f"   median absolute sex difference: {both['sex_difference'].abs().median():.1f} years")

    try:
        matched = concordance(table)
    except FileNotFoundError:
        print()
        print("   T2 is not built, so the concordance is not computed")
        return table

    statistics = concordance_statistics(matched)
    print()
    print("   concordance against T2, the Finnish registry axis")
    for name, value in statistics.items():
        print(f"      {name}: {value if isinstance(value, str) else round(float(value), 4)}")

    if not matched.empty:
        matched.to_parquet(config.interim_dir() / "onset_t2_t3_concordance.parquet", index=False)
    return table


def load() -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / "onset_t3.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the T3 complex onset axis")
    parser.parse_args()
    config.ensure_dirs()
    run()


if __name__ == "__main__":
    main()
