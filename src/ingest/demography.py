"""Modern survivorship and fertility schedules for the selection gradient.

Hamilton's gradient needs l(x), the probability of surviving from birth to age
x, and m(x), the expected offspring produced between x and x + 1. Both come from
the UN World Population Prospects, which is openly licensed, rather than from
the Human Mortality Database, which requires registration and cannot be
redistributed.

The schedules are female and single-year-of-age, because the one-sex gradient is
defined on the maternal line. The two-sex extension is built on
top of these in src/models/demography.py.

Finland is the default population. The complex-disease hazards come from
FinnGen, so taking the demography from the same population keeps the two halves
of s_d on a common footing. A world schedule is available as a sensitivity arm.

The files are large and mostly other countries, so they are streamed and
filtered rather than loaded.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd

from src import config
from src.ingest import fetch

SOURCE = "demography"


def _spec() -> dict:
    return config.sources()[SOURCE]["un_wpp"]


def _download(name: str) -> Path:
    spec = _spec()
    entry = spec["files"][name]
    dest = config.raw_dir() / SOURCE / Path(entry["url"].split("?")[0]).name
    return fetch.download(
        entry["url"],
        dest,
        source=SOURCE,
        version=spec["version"],
        licence=spec["licence"],
        redistribute=spec["redistribute"],
    )


def _stream_filter(path: Path, location: str, year: int, columns: list[str]) -> pd.DataFrame:
    """Read a gzipped WPP export keeping only one location and year."""
    location_token = location.lower()
    year_token = str(year)

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split(",")
        index = {name: position for position, name in enumerate(header)}
        missing = [name for name in columns if name not in index]
        if missing:
            raise KeyError(f"{path.name} lacks columns {missing}; it has {header[:12]}")

        location_column = index["Location"]
        time_column = index["Time"]
        wanted = [index[name] for name in columns]

        rows = []
        for line in handle:
            # Cheap rejection before splitting, since most rows are other
            # countries and other years.
            if year_token not in line:
                continue
            fields = line.rstrip("\n").split(",")
            if len(fields) <= max(wanted):
                continue
            if fields[location_column].strip('"').lower() != location_token:
                continue
            if fields[time_column].strip('"') != year_token:
                continue
            rows.append([fields[position].strip('"') for position in wanted])

    return pd.DataFrame(rows, columns=columns)


def survivorship(location: str, year: int) -> pd.DataFrame:
    """Female l(x) on a single-year grid, normalised to l(0) = 1."""
    path = _download("life_table_female")
    frame = _stream_filter(path, location, year, ["AgeGrpStart", "lx"])
    if frame.empty:
        raise ValueError(f"no life table rows for {location} in {year}")

    frame = frame.astype({"AgeGrpStart": int, "lx": float}).sort_values("AgeGrpStart")
    frame = frame.rename(columns={"AgeGrpStart": "age"})
    radix = frame["lx"].iloc[0]
    frame["survivorship"] = frame["lx"] / radix
    return frame[["age", "survivorship"]].reset_index(drop=True)


def fertility(location: str, year: int) -> pd.DataFrame:
    """Age-specific fertility on a single-year grid, as daughters per woman per year.

    Two conversions, and neither would fail an assertion downstream if it were
    skipped. WPP publishes the rate per 1,000 women, so it is divided through.
    And it publishes all births, while the one-sex gradient runs on the maternal
    line and needs daughters, so it is multiplied by the female share of births.
    Without the second step the net reproduction rate comes out around twice its
    true value and the intrinsic growth rate takes the wrong sign.
    """
    path = _download("fertility_by_age")
    frame = _stream_filter(path, location, year, ["AgeGrpStart", "ASFR"])
    if frame.empty:
        raise ValueError(f"no fertility rows for {location} in {year}")

    female_share = config.analysis()["fitness_cost"]["female_birth_fraction"]
    frame = frame.astype({"AgeGrpStart": int, "ASFR": float}).sort_values("AgeGrpStart")
    frame = frame.rename(columns={"AgeGrpStart": "age"})
    frame["fertility_all_births"] = frame["ASFR"] / 1000.0
    frame["fertility"] = frame["fertility_all_births"] * female_share
    return frame[["age", "fertility", "fertility_all_births"]].reset_index(drop=True)


def build(location: str, year: int, max_age: int) -> pd.DataFrame:
    """One row per year of age with survivorship and fertility aligned."""
    ages = pd.DataFrame({"age": range(max_age + 1)})
    table = (
        ages.merge(survivorship(location, year), on="age", how="left")
        .merge(fertility(location, year), on="age", how="left")
        .assign(location=location, year=year)
    )

    # Survivorship is defined at every age; fertility is published only over the
    # reproductive span and is zero elsewhere by definition, not missing.
    table["survivorship"] = table["survivorship"].ffill()
    table["fertility"] = table["fertility"].fillna(0.0)
    table["fertility_all_births"] = table["fertility_all_births"].fillna(0.0)

    if table["survivorship"].isna().any():
        raise ValueError("survivorship is undefined at some ages after alignment")
    return table


def run(location: str | None = None, year: int | None = None) -> pd.DataFrame:
    settings = config.analysis()["fitness_cost"]
    location = location or settings["demography_location"]
    year = year or settings["demography_year"]
    max_age = settings["max_age"]

    table = build(location, year, max_age)
    target = config.interim_dir() / "demography_modern.parquet"
    table.to_parquet(target, index=False)

    fertile = table[table["fertility"] > 0]
    print(f"demography_modern: {location} {year}, ages 0 to {max_age}")
    print(f"   survivorship at 50: {float(table.loc[50, 'survivorship']):.4f}")
    print(f"   total fertility rate: {float(table['fertility_all_births'].sum()):.3f}")
    print(
        f"   net reproduction rate: "
        f"{float((table['survivorship'] * table['fertility']).sum()):.3f}"
    )
    print(
        f"   fertile span: {int(fertile['age'].min())} to {int(fertile['age'].max())}, "
        f"peak at {int(fertile.loc[fertile['fertility'].idxmax(), 'age'])}"
    )
    return table


def load() -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / "demography_modern.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a modern demographic schedule")
    parser.add_argument("--location", default=None)
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()
    config.ensure_dirs()
    run(location=args.location, year=args.year)


if __name__ == "__main__":
    main()
