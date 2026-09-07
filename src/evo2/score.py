"""Per-gene Evo 2 scores from stop-codon likelihood ratios.

The arm asks whether a genomic language model carries information about disease
genes that the alignment-based scores do not. The measurement is the one the
published work uses: how much less likely a sequence becomes when a premature
stop codon is inserted early in its coding region. A gene whose disruption
matters should be a gene the model is more surprised to see disrupted.

    score = ( log P(mutant) - log P(wild type) ) / window length

The score is negative by construction and more negative for genes the model
treats as less tolerant of a premature stop.

The route is the hosted forward endpoint with the unembedding layer requested,
which returns the vocabulary logit vector at every position of the supplied
sequence. This is what lets the arm run at 40B, which is the size the published
validation number came from.

Three choices are recorded here rather than left to the reader.

**The window.** 8 kb centred on the coding start, taken from the primary
assembly and reverse complemented for genes on the minus strand. Evo 2 is
pretrained on genomic sequence including introns, so a spliced transcript would
be out of distribution.

**The stop position.** A stop codon replaces the codon at nucleotide offsets 15,
30 and 45 from the coding start, which are codons 6, 11 and 16. Three positions
are scored and averaged, so the result does not depend on one arbitrary choice.
The positions are close to the coding start on purpose: they stay inside the
first coding exon for almost every gene, which keeps the edit a plain
substitution in genomic coordinates rather than something that has to be
threaded through a splice structure.

**Which genes.** A gene is scored only if its first coding exon is long enough
to hold the furthest insertion with a codon to spare. Genes with a shorter first
coding exon are not scored and are counted, rather than being scored by a rule
that differs from the one stated here.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import gzip
import io
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.evo2 import genome as genome_module

ENDPOINT = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"
OUTPUT_LAYER = "unembed"
WINDOW = 8192
STOP_OFFSETS = (15, 30, 45)

# The span over which the likelihood difference is summed, beginning at the edit.
# Fixed at 200 bases before the restricted scoring was run. Measured afterwards,
# 88 per cent of the difference already falls inside it, so the whole-window sum
# and this one rank genes almost identically.
SCORED_SPAN = 200
STOP_CODON = "TGA"
KEY_ENVIRONMENT = "NVIDIA_API_KEY"
KEY_PATH_ENVIRONMENT = "NVIDIA_API_KEY_FILE"

# A run of this length will meet network outages, endpoint restarts and rate
# limiting, and none of those should cost a gene. A request is retried with
# exponential backoff for up to this long before it is given up on, and a gene
# given up on is left out of the checkpoint so that a later pass takes it again.
RETRY_BUDGET_SECONDS = 3600.0
BACKOFF_INITIAL = 5.0
BACKOFF_CEILING = 300.0

# Codes that will never succeed on retry. A bad credential or a malformed
# request should stop the run and be seen, not be retried for an hour.
FATAL_STATUS = frozenset({400, 401, 403, 404, 405, 422})

GENE_ID = re.compile(r'gene_id "([^".]+)')


class FatalRequestError(RuntimeError):
    """A request that will not succeed however many times it is sent."""


TAG_CANONICAL = 'tag "Ensembl_canonical"'


def api_key() -> str:
    """The credential, from the environment or from a file named by it.

    It is never read from inside the repository and never written into an
    output, so that a checkpoint file can be copied around without carrying it.
    """
    key = os.environ.get(KEY_ENVIRONMENT)
    if key:
        return key.strip()
    path = os.environ.get(KEY_PATH_ENVIRONMENT)
    if path and Path(path).exists():
        text = Path(path).read_text(encoding="utf-8")
        found = re.search(r"nvapi-[A-Za-z0-9_-]+", text)
        if found:
            return found.group(0)
    raise RuntimeError(
        f"set {KEY_ENVIRONMENT} to the credential, or {KEY_PATH_ENVIRONMENT} to a file holding it"
    )


def canonical_coding_starts() -> pd.DataFrame:
    """Coding start and first coding exon length for each protein-coding gene.

    The canonical transcript is the one GENCODE tags as such. Its first coding
    exon is the first in transcription order, which is the last one in file
    order on the minus strand.
    """
    spec = config.sources()["gencode"]
    path = config.raw_dir() / "gencode" / Path(spec["files"]["annotation"]["url"]).name

    blocks: dict[str, list[tuple[str, int, int, str]]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "CDS" or TAG_CANONICAL not in fields[8]:
                continue
            match = GENE_ID.search(fields[8])
            if match is None:
                continue
            blocks.setdefault(match.group(1), []).append(
                (fields[0], int(fields[3]), int(fields[4]), fields[6])
            )

    rows = []
    for gene, exons in blocks.items():
        contig, strand = exons[0][0], exons[0][3]
        if strand == "+":
            first = min(exons, key=lambda item: item[1])
            start = first[1]
        else:
            first = max(exons, key=lambda item: item[2])
            start = first[2]
        rows.append(
            {
                "ensembl_gene_id": gene,
                "contig": contig,
                "strand": strand,
                "coding_start": start,
                "first_exon_coding_length": first[2] - first[1] + 1,
                "n_coding_exons": len(exons),
            }
        )
    return pd.DataFrame(rows).sort_values("ensembl_gene_id").reset_index(drop=True)


def window_for(source: genome_module.Genome, row) -> str:
    """The 8 kb window around a coding start, in transcription orientation."""
    half = WINDOW // 2
    start = int(row["coding_start"])
    if row["strand"] == "+":
        sequence = source.fetch(row["contig"], start - half, start + half - 1)
    else:
        sequence = genome_module.reverse_complement(
            source.fetch(row["contig"], start - half + 1, start + half)
        )
    return sequence


def insert_stop(sequence: str, offset: int) -> str:
    """Replace the codon at a nucleotide offset from the coding start.

    The coding start sits at the centre of the window, so the codon to replace
    begins there plus the offset.
    """
    position = WINDOW // 2 + offset
    if position + 3 > len(sequence):
        raise ValueError("the stop position falls outside the window")
    return sequence[:position] + STOP_CODON + sequence[position + 3 :]


def _decode(payload: dict) -> np.ndarray:
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(payload["data"])))
    return np.load(io.BytesIO(archive.read(archive.namelist()[0])))[0]


def forward(sequence: str, key: str) -> np.ndarray:
    """Vocabulary logits at every position of the sequence.

    Retries anything that could be transient for up to the retry budget, with
    exponential backoff and jitter so that eight workers coming back from an
    outage do not arrive together. Anything that cannot succeed on retry is
    raised as fatal immediately.
    """
    body = json.dumps({"sequence": sequence, "output_layers": [OUTPUT_LAYER]}).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    deadline = time.time() + RETRY_BUDGET_SECONDS
    delay = BACKOFF_INITIAL
    last: Exception | None = None
    attempts = 0

    while True:
        attempts += 1
        try:
            request = urllib.request.Request(ENDPOINT, data=body, headers=headers)
            with urllib.request.urlopen(request, timeout=300) as response:
                return _decode(json.loads(response.read()))
        except urllib.error.HTTPError as error:
            if error.code in FATAL_STATUS:
                detail = ""
                with contextlib.suppress(OSError):
                    detail = error.read().decode("utf-8", "replace")[:200]
                raise FatalRequestError(f"HTTP {error.code} from the endpoint: {detail}") from error
            last = error
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as error:
            last = error

        if time.time() >= deadline:
            raise RuntimeError(
                f"gave up after {attempts} attempts over "
                f"{RETRY_BUDGET_SECONDS / 60:.0f} minutes: {last}"
            )
        time.sleep(min(delay, BACKOFF_CEILING) * (0.5 + random.random()))
        delay = min(delay * 2, BACKOFF_CEILING)


def position_log_probs(sequence: str, logits: np.ndarray) -> np.ndarray:
    """Log probability of the realised base at each position.

    Kept rather than summed on the spot. The first run summed immediately and
    discarded the array, which meant that evaluating any other summation window
    needed the whole set of calls again. These are cached so that it does not.

    Position i carries the distribution over the token at i + 1, so the returned
    array is one shorter than the sequence and its element i is the log
    probability of base i + 1.
    """
    tokens = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8).astype(np.int64)
    predictive = logits[: len(tokens) - 1].astype(np.float64)
    shifted = predictive - predictive.max(axis=1, keepdims=True)
    normaliser = np.log(np.exp(shifted).sum(axis=1))
    return shifted[np.arange(len(tokens) - 1), tokens[1:]] - normaliser


def log_likelihood(sequence: str, logits: np.ndarray) -> float:
    """Total log likelihood of a sequence under its own returned logits.

    Position i carries the distribution over the token at i + 1, so the first
    base is not scored. Evo 2 tokenises single bytes, and the token identifier
    of a base is its ASCII code.
    """
    tokens = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8).astype(np.int64)
    predictive = logits[: len(tokens) - 1].astype(np.float64)
    shifted = predictive - predictive.max(axis=1, keepdims=True)
    normaliser = np.log(np.exp(shifted).sum(axis=1))
    chosen = shifted[np.arange(len(tokens) - 1), tokens[1:]] - normaliser
    return float(chosen.sum())


def score_gene(row, source: genome_module.Genome, key: str, cache: Path | None = None) -> dict:
    sequence = window_for(source, row)
    if len(sequence) < WINDOW or set(sequence) - set("ACGTN"):
        return {"ensembl_gene_id": row["ensembl_gene_id"], "note": "window incomplete"}
    if sequence.count("N") > WINDOW // 10:
        return {"ensembl_gene_id": row["ensembl_gene_id"], "note": "window is mostly unplaced"}

    wild_type = position_log_probs(sequence, forward(sequence, key))
    stored = {"wild_type": wild_type.astype(np.float32)}

    full, restricted = [], []
    for offset in STOP_OFFSETS:
        mutant_sequence = insert_stop(sequence, offset)
        mutant = position_log_probs(mutant_sequence, forward(mutant_sequence, key))
        stored[f"mutant_{offset}"] = mutant.astype(np.float32)

        difference = mutant - wild_type
        full.append(float(difference.sum()))

        # Element i of the array is the log probability of base i + 1, so the
        # first prediction the edit can reach is the one at index p - 1, where p
        # is the first edited base.
        start = max(WINDOW // 2 + offset - 1, 0)
        restricted.append(float(difference[start : start + SCORED_SPAN].sum()))

    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(cache / f"{row['ensembl_gene_id']}.npz"), **stored)  # type: ignore[arg-type]  # the numpy stub reads **arrays as allow_pickle

    return {
        "ensembl_gene_id": row["ensembl_gene_id"],
        "wild_type_log_likelihood": float(wild_type.sum()),
        "mean_delta": float(np.mean(full)),
        "evo2_score": float(np.mean(full) / WINDOW),
        "evo2_score_restricted": float(np.mean(restricted) / SCORED_SPAN),
        "delta_spread": float(np.std(full)),
        "note": "",
    }


def eligible(table: pd.DataFrame) -> pd.DataFrame:
    """Genes whose first coding exon can hold the furthest insertion."""
    needed = max(STOP_OFFSETS) + 3
    return table[table["first_exon_coding_length"] >= needed].reset_index(drop=True)


def run(
    limit: int | None = None,
    workers: int = 8,
    seed: int | None = None,
    rescore: bool = False,
) -> pd.DataFrame:
    key = api_key()
    settings = config.analysis()
    starts = canonical_coding_starts()
    usable = eligible(starts)

    print(f"canonical coding starts: {len(starts):,} genes")
    print(f"   first coding exon long enough for the stop rule: {len(usable):,}")
    print(f"   excluded on that rule: {len(starts) - len(usable):,}")

    if limit:
        usable = usable.sample(
            n=min(limit, len(usable)),
            random_state=seed if seed is not None else settings["seed"],
        ).reset_index(drop=True)
        print(f"   scoring a random subset of {len(usable):,}")

    out_dir = config.interim_dir() / "evo2"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "scores_rescored" if rescore else "scores"
    checkpoint = out_dir / f"{name}.jsonl"
    cache_dir = out_dir / "positions"

    # The pilot rescores genes that already carry a full-window score, so that
    # the two scorings can be compared paired on the same genes rather than as
    # two independent runs on different ones.
    if rescore:
        already = collect("scores")
        usable = usable[usable["ensembl_gene_id"].isin(set(already["ensembl_gene_id"]))]
        print(f"   rescoring from the {len(usable):,} genes that already carry a score")

    done = set()
    if checkpoint.exists():
        with checkpoint.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)["ensembl_gene_id"])
                except (ValueError, KeyError):
                    continue
        print(f"   resuming, {len(done):,} already scored")

    pending = usable[~usable["ensembl_gene_id"].isin(done)]
    if pending.empty:
        print("   nothing left to score")
        return collect(name)

    fasta = config.raw_dir() / "gencode" / "GRCh38.primary_assembly.genome.fa"
    if not fasta.exists():
        raise FileNotFoundError(f"{fasta} is missing; the assembly has to be decompressed first")

    # Every worker opens its own handle on the assembly, and opening one builds
    # the index if it is not yet cached. Building it here first means one pass
    # over three gigabytes rather than one per worker, and no race to write the
    # cache.
    started_index = time.time()
    genome_module.load_index(fasta)
    print(f"   assembly index ready in {time.time() - started_index:.0f}s")

    lock = threading.Lock()
    started = time.time()
    completed = 0
    deferred = 0
    fatal: str | None = None

    def worker(row) -> None:
        nonlocal completed, deferred, fatal
        if fatal:
            return
        with genome_module.Genome(fasta) as source:
            try:
                record = score_gene(row, source, key, cache=cache_dir)
            except FatalRequestError as error:
                with lock:
                    fatal = fatal or str(error)
                return
            except RuntimeError:
                # The retry budget ran out, which is a fact about the network
                # rather than about this gene. It is deliberately left out of
                # the checkpoint so that the next pass takes it again; writing
                # a failure here would retire the gene permanently.
                with lock:
                    deferred += 1
                return
            except ValueError as error:
                record = {"ensembl_gene_id": row["ensembl_gene_id"], "note": str(error)[:200]}
        with lock:
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record))
                handle.write("\n")
                # Flushed to the platter rather than to the page cache, so that
                # a power loss costs the genes in flight and not the hours
                # already spent.
                handle.flush()
                os.fsync(handle.fileno())
            completed += 1
            if completed % 25 == 0:
                rate = completed / max(time.time() - started, 1e-9)
                remaining = (len(pending) - completed) / max(rate, 1e-9) / 3600
                print(
                    f"   {completed:,} of {len(pending):,}, "
                    f"{rate * 3600:.0f} genes/hour, {remaining:.1f} hours left",
                    flush=True,
                )

    rows = [row for _, row in pending.iterrows()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, rows))

    if fatal:
        raise FatalRequestError(fatal)
    if deferred:
        print(f"   {deferred:,} genes deferred to a later pass, not retired")

    table = collect(name)
    table.attrs["deferred"] = deferred
    return table


def collect(name: str = "scores") -> pd.DataFrame:
    """The checkpoint, folded into a table."""
    checkpoint = config.interim_dir() / "evo2" / f"{name}.jsonl"
    if not checkpoint.exists():
        return pd.DataFrame(columns=["ensembl_gene_id", "evo2_score", "note"])

    records = []
    with checkpoint.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue

    table = pd.DataFrame(records).drop_duplicates("ensembl_gene_id", keep="last")
    table = table.reset_index(drop=True)
    table.to_parquet(config.interim_dir() / "evo2" / f"{name}.parquet", index=False)

    scored = table[table.get("evo2_score").notna()] if "evo2_score" in table.columns else table
    print(f"evo2: {len(table):,} genes attempted, {len(scored):,} scored")
    if not scored.empty:
        print(scored[["evo2_score", "delta_spread"]].describe().to_string())
    failed = table[table["note"].astype(str).str.len() > 0] if "note" in table.columns else table
    if not failed.empty:
        print(f"   not scored: {len(failed):,}")
        print(failed["note"].value_counts().head(5).to_string())
    return table


def load(name: str = "scores") -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / "evo2" / f"{name}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score genes with Evo 2 stop-codon likelihoods")
    parser.add_argument("--limit", type=int, default=None, help="score a random subset this size")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="rescore genes that already carry a score, into a separate checkpoint",
    )
    args = parser.parse_args()
    config.ensure_dirs()

    # The exit code is what the supervisor reads. Nothing left to do is a clean
    # exit; genes deferred by a network problem are a retryable failure; a
    # credential or request problem is fatal and must not be looped on.
    try:
        table = run(limit=args.limit, workers=args.workers, rescore=args.rescore)
    except FatalRequestError as error:
        print(f"fatal: {error}", flush=True)
        raise SystemExit(2) from error

    if int(table.attrs.get("deferred", 0)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
