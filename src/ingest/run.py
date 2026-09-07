"""Ingest entry point.

Order matters only in that the Open Targets disease index is needed before the
crosswalk runs. Everything else is independent, and each module is idempotent:
a file already present with a matching manifest hash is left alone.
"""

from __future__ import annotations

import argparse

from src import config
from src.ingest import manifest, open_targets, orphanet, simple_tables

STAGES = {
    "open_targets": lambda: open_targets.run(),
    "orphanet": lambda: orphanet.run(),
    "tables": lambda: simple_tables.run(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve and normalise upstream sources")
    parser.add_argument("--all", action="store_true", help="run every stage")
    parser.add_argument("--stage", action="append", choices=sorted(STAGES), default=[])
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="do not re-hash previously recorded files before starting",
    )
    args = parser.parse_args()

    config.ensure_dirs()

    if not args.skip_verify:
        failures = manifest.verify(strict=False)
        if failures:
            raise SystemExit(
                "Manifest verification failed. An upstream file changed under a pinned "
                "version, or the raw layer was edited.\n  " + "\n  ".join(failures)
            )

    stages = sorted(STAGES) if args.all or not args.stage else args.stage
    for name in stages:
        print(f"[{name}]")
        STAGES[name]()
        print()


if __name__ == "__main__":
    main()
