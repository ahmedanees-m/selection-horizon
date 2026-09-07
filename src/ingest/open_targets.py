"""Open Targets Platform ingest.

Release 26.06 is pinned. That release ships EFO 3.88.0, in which a large block
of EFO disease terms was replaced by Mondo terms and the study and evidence
mapping followed the change. Everything downstream is keyed on Mondo, and the
row-count assertions in src/crosswalk/qc.py exist because a join against a
deprecated EFO identifier drops rows silently rather than failing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.ingest import fetch

SOURCE = "open_targets"


def _spec() -> dict:
    return config.sources()["open_targets"]


def dataset_dir(name: str) -> Path:
    return config.raw_dir() / SOURCE / name


def download(name: str) -> Path:
    spec = _spec()
    entry = spec["datasets"][name]
    dest = dataset_dir(name)

    if entry["layout"] == "single_file":
        target = dest / Path(entry["path"]).name
        fetch.download(
            f"{spec['base_url']}/{entry['path']}",
            target,
            source=SOURCE,
            version=spec["release"],
            licence=spec["licence"],
            redistribute=spec["redistribute"],
        )
    else:
        fetch.download_spark_dataset(
            spec["base_url"],
            entry["path"],
            dest,
            source=SOURCE,
            version=spec["release"],
            licence=spec["licence"],
            redistribute=spec["redistribute"],
        )
    return dest


def read(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a downloaded dataset, concatenating Spark parts."""
    directory = dataset_dir(name)
    parts = sorted(directory.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"{name} has not been downloaded to {directory}")
    frames = [pd.read_parquet(part, columns=columns) for part in parts]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def run(names: list[str] | None = None) -> None:
    spec = _spec()
    for name in names or list(spec["datasets"]):
        directory = download(name)
        parts = sorted(directory.glob("*.parquet"))
        total = sum(part.stat().st_size for part in parts)
        print(f"{name}: {len(parts)} file(s), {total / 1e6:,.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Open Targets datasets")
    parser.add_argument("datasets", nargs="*", help="dataset names; default is all pinned datasets")
    args = parser.parse_args()
    config.ensure_dirs()
    run(args.datasets or None)


if __name__ == "__main__":
    main()
