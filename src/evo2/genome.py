"""Random access to the primary assembly, without a new dependency.

The Evo 2 arm needs an 8 kb genomic window around a coding start for each of
roughly 19,000 genes. Reading a 3 GB FASTA once per gene is not an option and
the container carries no indexed-FASTA library, so the index is built here. It
is the same index a FASTA index file records: for each sequence, where its bases
begin in the file, how many bases sit on a line, and how many bytes that line
occupies. With those three numbers a base offset converts to a byte offset
arithmetically and any window is one seek and one read.

This works only on a FASTA whose lines are a fixed width within each sequence,
which every reference assembly release satisfies. The builder checks it rather
than assuming it, because a file that violates it would return sequence from the
wrong coordinates rather than fail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


@dataclass(frozen=True)
class Record:
    """Where one sequence lives in the file, in bases and in bytes."""

    offset: int
    length: int
    line_bases: int
    line_bytes: int


def build_index(path: Path) -> dict[str, Record]:
    """One pass over the file, recording where each sequence starts."""
    index: dict[str, Record] = {}
    name: str | None = None
    offset = line_bases = line_bytes = length = 0
    ragged = False

    with path.open("rb") as handle:
        position = 0
        for raw in handle:
            width = len(raw)
            if raw.startswith(b">"):
                if name is not None:
                    index[name] = Record(offset, length, line_bases, line_bytes)
                name = raw[1:].split()[0].decode("ascii")
                offset = position + width
                line_bases = line_bytes = length = 0
                ragged = False
            elif name is not None:
                bases = len(raw.rstrip(b"\r\n"))
                if line_bases == 0:
                    line_bases, line_bytes = bases, width
                elif bases != line_bases and not ragged:
                    # The last line of a sequence is short by construction. A
                    # short line anywhere else means the arithmetic below does
                    # not hold for this sequence.
                    ragged = True
                elif ragged and bases:
                    raise ValueError(f"{name} has lines of varying width; the index cannot be used")
                length += bases
            position += width

    if name is not None:
        index[name] = Record(offset, length, line_bases, line_bytes)
    return index


def load_index(fasta: Path) -> dict[str, Record]:
    """The index for a FASTA, built once and cached beside it."""
    cache = fasta.with_suffix(fasta.suffix + ".index.json")
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        return {name: Record(**values) for name, values in stored.items()}

    index = build_index(fasta)
    cache.write_text(
        json.dumps({name: vars(record) for name, record in index.items()}), encoding="utf-8"
    )
    return index


class Genome:
    """Sequence by coordinate, on a one-based inclusive convention."""

    def __init__(self, fasta: Path):
        self.path = Path(fasta)
        self.index = load_index(self.path)
        self._handle = self.path.open("rb")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Genome:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def contigs(self) -> list[str]:
        return sorted(self.index)

    def fetch(self, contig: str, start: int, end: int) -> str:
        """Bases from start to end inclusive, clipped to the contig."""
        record = self.index.get(contig)
        if record is None:
            raise KeyError(f"{contig} is not in {self.path.name}")

        start = max(1, start)
        end = min(record.length, end)
        if end < start:
            return ""

        def byte_offset(base: int) -> int:
            whole_lines, within = divmod(base - 1, record.line_bases)
            return record.offset + whole_lines * record.line_bytes + within

        self._handle.seek(byte_offset(start))
        span = byte_offset(end) - byte_offset(start) + 1
        raw = self._handle.read(span)
        return raw.replace(b"\n", b"").replace(b"\r", b"").decode("ascii").upper()


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]
