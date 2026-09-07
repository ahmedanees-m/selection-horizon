"""The download manifest.

Upstream resources change without notice and without a version bump. The raw
layer is therefore write-once: a file is downloaded, hashed, and recorded here.
On every later run the recorded hash is checked, and a mismatch stops the
pipeline instead of letting a changed input propagate into a result that still
looks plausible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src import config

CHUNK = 1 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def load() -> dict[str, Any]:
    path = config.manifest_path()
    if not path.exists():
        return {"entries": {}}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save(manifest: dict[str, Any]) -> None:
    path = config.manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def record(
    path: Path,
    *,
    source: str,
    url: str,
    version: str,
    licence: str,
    redistribute: bool,
) -> str:
    """Hash a retrieved file and write its entry."""
    manifest = load()
    key = str(path.relative_to(config.data_root()))
    digest = sha256(path)
    manifest["entries"][key] = {
        "sha256": digest,
        "bytes": path.stat().st_size,
        "source": source,
        "url": url,
        "version": version,
        "retrieved": datetime.now(UTC).date().isoformat(),
        "licence": licence,
        "redistribute": redistribute,
    }
    save(manifest)
    return digest


def verify(strict: bool = True) -> list[str]:
    """Re-hash every retained file. Returns the keys that no longer match.

    Entries flagged `retained: false` were aggregated and deleted on purpose,
    with an upstream checksum recorded in their place. They are absent by
    design, so treating their absence as a failure would make the check cry
    wolf on every run.
    """
    manifest = load()
    root = config.data_root()
    failures: list[str] = []
    for key, entry in sorted(manifest["entries"].items()):
        if not entry.get("retained", True):
            continue
        path = root / key
        if not path.exists():
            failures.append(f"{key}: missing")
            continue
        if sha256(path) != entry["sha256"]:
            failures.append(f"{key}: hash mismatch")
    if failures and strict:
        raise SystemExit("Manifest verification failed:\n  " + "\n  ".join(failures))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or verify the download manifest")
    parser.add_argument("--verify", action="store_true", help="re-hash every recorded file")
    args = parser.parse_args()

    manifest = load()
    if args.verify:
        verify(strict=True)
        print(f"{len(manifest['entries'])} entries verified")
        return

    for key, entry in sorted(manifest["entries"].items()):
        flag = "public" if entry["redistribute"] else "restricted"
        print(f"{key}  {entry['bytes']:>13,}  {entry['sha256'][:12]}  {flag}")


if __name__ == "__main__":
    main()
