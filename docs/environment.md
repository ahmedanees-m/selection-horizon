# Computational environment

Every step runs in a container. The analysis image is built from the Dockerfile
at the repository root on `python:3.11-slim-bookworm`, with dependencies pinned
in `pyproject.toml`.

```bash
make image
make all
```

Hardware used: a single Linux host with 64 GB of memory and an RTX A4000. No
step requires a GPU. The genomic language model arm reached a hosted endpoint
rather than a local model and needs a credential, read from `NVIDIA_API_KEY` or
from the file named by `NVIDIA_API_KEY_FILE`.

## Data provenance

`data/MANIFEST.json` records, for every input file, the source name, the release
version, the retrieval URL, the retrieval date, the SHA-256 checksum and the
redistribution terms. The pipeline verifies checksums on load and halts if one no
longer matches, so a silently changed upstream release cannot propagate into a
result.

Several upstream resources restrict redistribution. Three conservation tracks
totalling 25 GB were aggregated per gene and then deleted, with the upstream
checksum retained in place of the file; the manifest marks these as not retained
and verification skips them rather than reporting them missing.

Source versions are pinned in `config/sources.yaml`. Verification of
those pins, covering URL resolution, digital object identifiers resolved through
Crossref, version claims read off the retrieved files rather than the
configuration, identifier formats, value ranges and cross-source overlaps, runs
as `make verify`.

## Determinism

Every random draw takes its seed from `config/analysis.yaml`: matched control
selection, cross-validation folds, and the permutation null. Repeated runs on the
same inputs produce the same outputs.
