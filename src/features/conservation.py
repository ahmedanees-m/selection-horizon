"""Per-gene conservation aggregates from the UCSC tracks.

The tracks are large: 9.6 GB for the Zoonomia 241-way phyloP, 9.9 GB for the
100-way, 5.9 GB for phastCons. They are pulled one at a time, aggregated in the
same pass, and deleted, so the peak footprint is about 10 GB rather than 25.4 GB.
Provenance survives deletion: UCSC publishes an md5 alongside each track, and
that upstream md5 is recorded in the manifest against a retained flag of false.
The aggregate table is regenerable from the recorded URL and hash.

Aggregation is not the raw CDS mean, which is dominated by codon-position
structure rather than by selection. Two aggregates are computed in one pass.

Primary: the fraction of coding bases whose score exceeds a fixed threshold.
Secondary: the mean score at four-fold degenerate sites, where synonymous change
is unconstrained at the protein level, so the residual signal is closer to
selection on the nucleotide itself.
"""

from __future__ import annotations

import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# Codons whose first two bases fix the amino acid regardless of the third.
FOURFOLD_PREFIXES = frozenset({"GC", "CG", "GG", "CT", "CC", "TC", "AC", "GT"})


def parse_cds_exons(gtf_path: Path, transcripts: set[str]) -> dict[str, list[tuple]]:
    """CDS exons per transcript, as (chrom, start, end, strand, frame).

    Coordinates are converted from the GTF's one-based inclusive convention to
    the half-open convention pyBigWig expects.
    """
    exons: dict[str, list[tuple]] = defaultdict(list)
    with gzip.open(gtf_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "CDS":
                continue
            attributes = fields[8]
            marker = attributes.find('transcript_id "')
            if marker < 0:
                continue
            start = marker + len('transcript_id "')
            transcript = attributes[start : attributes.index('"', start)].split(".")[0]
            if transcript not in transcripts:
                continue
            exons[transcript].append(
                (fields[0], int(fields[3]) - 1, int(fields[4]), fields[6], int(fields[7]))
            )

    for transcript, blocks in exons.items():
        reverse = blocks[0][3] == "-"
        exons[transcript] = sorted(blocks, key=lambda block: block[1], reverse=reverse)
    return exons


def coding_positions(blocks: list[tuple]) -> tuple[str, np.ndarray]:
    """Genomic positions in transcription order, so index i is CDS base i."""
    chrom = blocks[0][0]
    reverse = blocks[0][3] == "-"
    pieces = []
    for _, start, end, _, _ in blocks:
        positions = np.arange(start, end, dtype=np.int64)
        pieces.append(positions[::-1] if reverse else positions)
    return chrom, np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def fourfold_mask(cds: str) -> np.ndarray:
    """True at the third base of every four-fold degenerate codon."""
    length = len(cds) - len(cds) % 3
    mask = np.zeros(len(cds), dtype=bool)
    for offset in range(0, length, 3):
        if cds[offset : offset + 2].upper() in FOURFOLD_PREFIXES:
            mask[offset + 2] = True
    return mask


def _fetch(handle, chrom: str, positions: np.ndarray) -> np.ndarray:
    """Scores at the given genomic positions, NaN where the track has no value."""
    if positions.size == 0:
        return np.empty(0)
    low, high = int(positions.min()), int(positions.max()) + 1
    try:
        window = np.asarray(handle.values(chrom, low, high), dtype=float)
    except (RuntimeError, ValueError):
        return np.full(positions.size, np.nan)
    return window[positions - low]


def check_threshold(
    track_path: Path, threshold: float, chrom_limit: int = 3
) -> tuple[float, float]:
    """Refuse a threshold that the track cannot reach.

    The tracks are not on a common scale. phyloP is a signed log p-value and
    phastCons a posterior probability bounded in [0, 1], so a threshold chosen
    for one is silently unreachable on the other and returns zero for every
    gene, which is a plausible-looking number that no downstream assertion
    catches. The observed range is read off the track header and the threshold
    is required to sit inside it.
    """
    import pyBigWig

    handle = pyBigWig.open(str(track_path))
    try:
        low, high = float("inf"), float("-inf")
        for chrom, length in list(handle.chroms().items())[:chrom_limit]:
            stats_min = handle.stats(chrom, 0, length, type="min")[0]
            stats_max = handle.stats(chrom, 0, length, type="max")[0]
            if stats_min is not None:
                low = min(low, float(stats_min))
            if stats_max is not None:
                high = max(high, float(stats_max))
    finally:
        handle.close()

    if not (low <= threshold <= high):
        raise ValueError(
            f"threshold {threshold} lies outside the range this track takes "
            f"([{low:.3f}, {high:.3f}]); the fraction above it would be "
            f"degenerate for every gene"
        )
    return low, high


def aggregate(
    track_path: Path,
    exons: dict[str, list[tuple]],
    sequences: dict[str, str],
    gene_of: dict[str, str],
    threshold: float,
) -> pd.DataFrame:
    import pyBigWig

    check_threshold(track_path, threshold)

    handle = pyBigWig.open(str(track_path))
    chroms = set(handle.chroms())

    rows = []
    for transcript, blocks in exons.items():
        chrom, positions = coding_positions(blocks)
        if chrom not in chroms or positions.size == 0:
            continue
        scores = _fetch(handle, chrom, positions)
        finite = np.isfinite(scores)
        if not finite.any():
            continue

        sequence = sequences.get(transcript, "")
        degenerate = fourfold_mask(sequence)[: positions.size] if sequence else np.zeros(0, bool)
        if degenerate.size < positions.size:
            degenerate = np.pad(degenerate, (0, positions.size - degenerate.size))

        usable = finite & degenerate
        rows.append(
            {
                "ensembl_gene_id": gene_of[transcript],
                "transcript_id": transcript,
                "n_cds_bases_scored": int(finite.sum()),
                "fraction_above_threshold": float((scores[finite] > threshold).mean()),
                "mean_score": float(scores[finite].mean()),
                "n_fourfold_scored": int(usable.sum()),
                "mean_fourfold": float(scores[usable].mean()) if usable.any() else np.nan,
            }
        )
    handle.close()
    return pd.DataFrame(rows)


def load_sequences() -> tuple[dict[str, str], dict[str, str]]:
    """CDS sequence and gene assignment for the transcript chosen per gene."""
    from src.ingest.gencode import CDS_SPAN, _records

    chosen = set(pd.read_parquet(config.interim_dir() / "gencode_cds.parquet")["transcript_id"])
    path = config.raw_dir() / "gencode" / "gencode.v44.pc_transcripts.fa.gz"

    sequences, gene_of = {}, {}
    for header, sequence in _records(path):
        fields = header[1:].split("|")
        transcript = fields[0].split(".")[0]
        if transcript not in chosen:
            continue
        span = CDS_SPAN.search(header)
        if span is None:
            continue
        start, end = int(span.group(1)), int(span.group(2))
        sequences[transcript] = sequence[start - 1 : end].upper()
        gene_of[transcript] = fields[1].split(".")[0]
    return sequences, gene_of
