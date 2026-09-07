"""Download helper for the raw layer.

Retrieval is resumable and write-once. A file already present with a matching
manifest entry is left alone; a file present without an entry is re-hashed and
recorded.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import urljoin

import requests

from src import config
from src.ingest import manifest

TIMEOUT = 120
CHUNK = 1 << 20
PART_PATTERN = re.compile(r'href="(part-[^"]+\.parquet)"')


def download(
    url: str,
    dest: Path,
    *,
    source: str,
    version: str,
    licence: str,
    redistribute: bool,
    force: bool = False,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    entries = manifest.load()["entries"]
    key = str(dest.relative_to(config.data_root()))

    if dest.exists() and not force:
        if key in entries and manifest.sha256(dest) == entries[key]["sha256"]:
            return dest
        manifest.record(
            dest,
            source=source,
            url=url,
            version=version,
            licence=licence,
            redistribute=redistribute,
        )
        return dest

    tmp = dest.with_suffix(dest.suffix + ".partial")
    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(CHUNK):
                handle.write(chunk)
    shutil.move(tmp, dest)

    manifest.record(
        dest,
        source=source,
        url=url,
        version=version,
        licence=licence,
        redistribute=redistribute,
    )
    return dest


def list_spark_parts(directory_url: str) -> list[str]:
    """Return the parquet part file names under a Spark output directory."""
    response = requests.get(directory_url, timeout=TIMEOUT)
    response.raise_for_status()
    return sorted(set(PART_PATTERN.findall(response.text)))


def download_spark_dataset(
    base_url: str,
    relative_path: str,
    dest_dir: Path,
    *,
    source: str,
    version: str,
    licence: str,
    redistribute: bool,
) -> list[Path]:
    directory_url = f"{base_url.rstrip('/')}/{relative_path.strip('/')}/"
    parts = list_spark_parts(directory_url)
    if not parts:
        raise RuntimeError(f"No parquet parts found under {directory_url}")

    paths = []
    for part in parts:
        paths.append(
            download(
                urljoin(directory_url, part),
                dest_dir / part,
                source=source,
                version=version,
                licence=licence,
                redistribute=redistribute,
            )
        )
    return paths
