"""Retrieve, aggregate and release one conservation track at a time.

The tracks total 25.4 GB. Holding all three would leave too little headroom on
the shared machine, and nothing belonging to another project is removed to make
room. Each track is therefore pulled, aggregated to per-gene values, and
deleted, which caps the footprint at roughly one track.

Deleting the track does not lose provenance. UCSC publishes an md5 alongside
each file; that upstream md5 is recorded in the manifest with a retained flag of
false, so the aggregate remains regenerable from the recorded URL and hash.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import requests

from src import config
from src.features import conservation
from src.ingest import fetch, gencode, manifest

TIMEOUT = 120


def upstream_md5(url: str) -> str | None:
    """The md5 UCSC publishes beside the track, if there is one.

    UCSC serves some tracks under both an assembly-prefixed name and a bare
    alias, while md5sum.txt lists only the prefixed one. The lookup therefore
    accepts a suffix match, so an alias in the configured URL still resolves.
    """
    directory, name = url.rsplit("/", 1)
    try:
        response = requests.get(f"{directory}/md5sum.txt", timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        return None

    candidates = {}
    for line in response.text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            candidates[parts[1]] = parts[0]
    if name in candidates:
        return candidates[name]
    for listed, digest in candidates.items():
        if listed.endswith(name) or name.endswith(listed):
            return digest
    return None


def record_released(path: Path, url: str, digest: str | None, spec: dict) -> None:
    """Manifest entry for a file that was retrieved, used, and removed."""
    entries = manifest.load()
    key = str(path.relative_to(config.data_root()))
    entries["entries"][key] = {
        "sha256": None,
        "upstream_md5": digest,
        "bytes": spec.get("size_bytes"),
        "source": "conservation",
        "url": url,
        "version": spec.get("assembly", ""),
        "retrieved": manifest.datetime.now(manifest.UTC).date().isoformat(),
        "licence": config.sources()["conservation"]["licence"],
        "redistribute": False,
        "retained": False,
    }
    manifest.save(entries)


def ensure_annotation() -> Path:
    spec = config.sources()["gencode"]
    entry = spec["files"]["annotation"]
    dest = config.raw_dir() / "gencode" / Path(entry["url"]).name
    return fetch.download(
        entry["url"],
        dest,
        source="gencode",
        version=spec["release"],
        licence=spec["licence"],
        redistribute=spec["redistribute"],
    )


def process(track: str, keep: bool = False) -> Path:
    spec = config.sources()["conservation"]["files"][track]
    url = spec["url"]
    thresholds = config.analysis()["conservation_aggregation"]["thresholds"]
    if track not in thresholds:
        raise KeyError(f"no aggregation threshold configured for {track}")
    threshold = thresholds[track]

    if not (config.interim_dir() / "gencode_cds.parquet").exists():
        gencode.run()

    annotation = ensure_annotation()
    sequences, gene_of = conservation.load_sequences()
    exons = conservation.parse_cds_exons(annotation, set(sequences))
    print(f"{track}: {len(exons):,} transcripts with CDS exons")

    track_path = config.raw_dir() / "conservation" / Path(url).name
    track_path.parent.mkdir(parents=True, exist_ok=True)
    digest = upstream_md5(url)

    if not track_path.exists():
        print(f"{track}: retrieving {spec.get('size_bytes', 0) / 1e9:.1f} GB")
        tmp = track_path.with_suffix(track_path.suffix + ".partial")
        with requests.get(url, stream=True, timeout=TIMEOUT) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(1 << 22):
                    handle.write(chunk)
        shutil.move(tmp, track_path)

    frame = conservation.aggregate(track_path, exons, sequences, gene_of, threshold)
    frame = frame.rename(
        columns={
            "fraction_above_threshold": f"{track}_fraction_above",
            "mean_score": f"{track}_mean",
            "mean_fourfold": f"{track}_mean_fourfold",
            "n_cds_bases_scored": f"{track}_n_scored",
            "n_fourfold_scored": f"{track}_n_fourfold",
        }
    )

    target = config.interim_dir() / f"conservation_{track}.parquet"
    frame.to_parquet(target, index=False)
    median_fraction = frame[f"{track}_fraction_above"].median()
    print(
        f"{track}: {len(frame):,} genes aggregated, "
        f"median fraction above {threshold} = {median_fraction:.3f}"
    )
    if median_fraction in (0.0, 1.0):
        print(
            f"{track}: the median fraction is degenerate at {median_fraction}; "
            f"the threshold of {threshold} does not separate this track"
        )

    record_released(track_path, url, digest, spec)
    if not keep:
        track_path.unlink()
        print(f"{track}: track released, upstream md5 {digest or 'not published'} recorded")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate conservation tracks per gene")
    parser.add_argument(
        "tracks",
        nargs="*",
        default=["phylop_241way", "phylop_100way", "phastcons_100way"],
    )
    parser.add_argument("--keep", action="store_true", help="do not delete the track afterwards")
    args = parser.parse_args()

    config.ensure_dirs()
    for track in args.tracks:
        process(track, keep=args.keep)


if __name__ == "__main__":
    main()
