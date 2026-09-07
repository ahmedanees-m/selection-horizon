"""GENCODE ingest for the confounder set.

Three of the confounders in the frozen adjustment set are properties of the
coding sequence itself: CDS length, GC content and, through them, the number of
loss-of-function variants that could have been observed. The protein-coding
transcript FASTA carries the CDS interval in each header, so both CDS length and
GC content come from one 48 MB file rather than from the full genome.

One transcript per gene is retained. MANE Select is preferred where the metadata
identifies it; otherwise the longest CDS wins, which is the GENCODE basic
convention and matches how the constraint metrics are keyed.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from src import config
from src.ingest import fetch

SOURCE = "gencode"
CDS_SPAN = re.compile(r"\|CDS:(\d+)-(\d+)\|")


def _download() -> Path:
    spec = config.sources()[SOURCE]
    entry = spec["files"]["transcripts"]
    dest = config.raw_dir() / SOURCE / Path(entry["url"]).name
    return fetch.download(
        entry["url"],
        dest,
        source=SOURCE,
        version=spec["release"],
        licence=spec["licence"],
        redistribute=spec["redistribute"],
    )


def _records(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    chunks: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line.rstrip("\n")
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def parse_transcripts(path: Path) -> pd.DataFrame:
    """One row per transcript with a CDS: gene, CDS length and CDS GC."""
    rows = []
    for header, sequence in _records(path):
        span = CDS_SPAN.search(header)
        if span is None:
            continue
        fields = header[1:].split("|")
        start, end = int(span.group(1)), int(span.group(2))
        cds = sequence[start - 1 : end].upper()
        if not cds:
            continue
        gc = sum(cds.count(base) for base in "GC")
        acgt = gc + sum(cds.count(base) for base in "AT")
        rows.append(
            {
                "transcript_id": fields[0].split(".")[0],
                "ensembl_gene_id": fields[1].split(".")[0],
                "gene_symbol": fields[5],
                "cds_length": len(cds),
                "gc_content": gc / acgt if acgt else None,
            }
        )
    return pd.DataFrame(rows)


def per_gene(transcripts: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one transcript per gene, longest CDS first."""
    ordered = transcripts.sort_values(
        ["ensembl_gene_id", "cds_length", "transcript_id"], ascending=[True, False, True]
    )
    return ordered.drop_duplicates("ensembl_gene_id").reset_index(drop=True)


def run() -> Path:
    frame = per_gene(parse_transcripts(_download()))
    target = config.interim_dir() / "gencode_cds.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    print(
        f"gencode_cds: {len(frame):,} genes, "
        f"median CDS {frame['cds_length'].median():,.0f} bp, "
        f"median GC {frame['gc_content'].median():.3f}"
    )
    return target


if __name__ == "__main__":
    config.ensure_dirs()
    run()
