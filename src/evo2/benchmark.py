"""Evo 2 under the perturbation the published benchmark uses.

The inclusion criterion for the conservation family compares a measured AUROC
against the 0.66 of the published evaluation. That figure comes from a different
measurement: the benchmark scores the difference in log likelihood between a
gene's coding sequence and a scramble of it preserving dinucleotide composition,
with no flanking sequence and only the first 8,192 bases of a longer gene. The
rule in `score` instead substitutes a stop codon inside an 8 kb genomic window.
The two differ in perturbation, in window and possibly in normalisation, so a
shortfall against 0.66 under that rule does not say the model fails the
benchmark.

This module applies the benchmark's rule on the same genes and the same labels,
so the criterion can be read against a like-for-like figure. It is a check on the
criterion, not an analysis arm: nothing here enters the onset contrast.

Two calls per gene, a reference and one scramble, against the hosted endpoint
`score` already uses.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src import config
from src.evo2.score import FatalRequestError, api_key, forward, log_likelihood

# The benchmark truncates a longer gene to its first 8,192 coding bases.
CONTEXT = 8192

# A dinucleotide-preserving shuffle needs its last edges to form a tree into the
# final base. Random choices usually do; a few do not and are redrawn.
SHUFFLE_ATTEMPTS = 32


def reaches(tail: dict[str, str], vertices: list[str], last: str) -> bool:
    """Whether following the chosen last edges from every base reaches the end."""
    for vertex in vertices:
        seen: set[str] = set()
        cursor = vertex
        while cursor != last:
            if cursor in seen or cursor not in tail:
                return False
            seen.add(cursor)
            cursor = tail[cursor]
    return True


def dinucleotide_shuffle(sequence: str, rng: np.random.Generator) -> str:
    """A permutation of the sequence with the same dinucleotide counts.

    The Altschul and Erikson construction: the sequence is an Euler path through
    a graph whose edges are its adjacent pairs, so any other Euler path from the
    same first base has the same dinucleotide composition. The last edge out of
    each base is fixed first, so that the walk cannot strand itself, and the rest
    are shuffled.
    """
    if len(sequence) < 4:
        return sequence

    last = sequence[-1]
    edges: dict[str, list[str]] = {}
    for left, right in zip(sequence, sequence[1:], strict=False):
        edges.setdefault(left, []).append(right)

    vertices = sorted(edges)
    tail: dict[str, str] = {}
    for _ in range(SHUFFLE_ATTEMPTS):
        tail = {
            vertex: edges[vertex][int(rng.integers(len(edges[vertex])))]
            for vertex in vertices
            if vertex != last
        }
        if reaches(tail, vertices, last):
            break
    else:
        # No valid choice found, so the sequence is returned unshuffled rather
        # than returning something whose composition is not preserved.
        return sequence

    walk: dict[str, list[str]] = {}
    for vertex in vertices:
        remaining = list(edges[vertex])
        if vertex in tail:
            remaining.remove(tail[vertex])
        order = rng.permutation(len(remaining))
        remaining = [remaining[index] for index in order]
        if vertex in tail:
            remaining.append(tail[vertex])
        walk[vertex] = remaining

    out = [sequence[0]]
    cursor = sequence[0]
    position = dict.fromkeys(vertices, 0)
    for _ in range(len(sequence) - 1):
        step = walk[cursor][position[cursor]]
        position[cursor] += 1
        out.append(step)
        cursor = step
    return "".join(out)


def coding_sequences() -> pd.DataFrame:
    """One coding sequence per gene, truncated as the benchmark truncates."""
    from src.features import conservation

    sequences, gene_of = conservation.load_sequences()
    rows = []
    for transcript, sequence in sequences.items():
        gene = gene_of.get(transcript)
        if gene is None or not sequence:
            continue
        rows.append(
            {
                "ensembl_gene_id": gene,
                "coding_length": len(sequence),
                "sequence": sequence[:CONTEXT],
            }
        )
    frame = pd.DataFrame(rows).drop_duplicates("ensembl_gene_id")
    return frame.sort_values("ensembl_gene_id").reset_index(drop=True)


def labelled_genes() -> set[str]:
    """Genes carrying a DepMap essentiality label, which are the ones scored."""
    from src.features import baseline

    table = baseline.load()
    return set(table.loc[table["depmap_common_essential"].notna(), "ensembl_gene_id"])


def score_gene(gene: str, sequence: str, key: str, seed: int) -> dict:
    """Reference and scrambled log likelihood for one gene."""
    if len(sequence) < 4 or set(sequence) - set("ACGT"):
        return {"ensembl_gene_id": gene, "note": "sequence not usable"}

    rng = np.random.default_rng(abs(hash((seed, gene))) % (2**32))
    scrambled = dinucleotide_shuffle(sequence, rng)
    if scrambled == sequence:
        return {"ensembl_gene_id": gene, "note": "no valid scramble"}

    reference = log_likelihood(sequence, forward(sequence, key))
    shuffled = log_likelihood(scrambled, forward(scrambled, key))
    difference = reference - shuffled
    return {
        "ensembl_gene_id": gene,
        "coding_length": len(sequence),
        "reference_log_likelihood": reference,
        "scrambled_log_likelihood": shuffled,
        "difference": difference,
        "difference_per_base": difference / len(sequence),
        "note": "",
    }


def collect(name: str = "benchmark_scores") -> pd.DataFrame:
    """The checkpoint read back as a table."""
    path = config.interim_dir() / "evo2" / f"{name}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates("ensembl_gene_id") if not frame.empty else frame


def run(limit: int | None = None, workers: int = 8, seed: int | None = None) -> pd.DataFrame:
    """Score every labelled gene under the benchmark's rule, resuming if stopped."""
    key = api_key()
    settings = config.analysis()
    seed = settings["seed"] if seed is None else seed

    table = coding_sequences()
    labelled = labelled_genes()
    table = table[table["ensembl_gene_id"].isin(labelled)].reset_index(drop=True)
    print(f"coding sequences with a DepMap label: {len(table):,} genes")
    print(f"   truncated to {CONTEXT:,} bases: {int((table['coding_length'] > CONTEXT).sum()):,}")

    if limit:
        table = table.sample(n=min(limit, len(table)), random_state=seed).reset_index(drop=True)
        print(f"   scoring a random subset of {len(table):,}")

    out_dir = config.interim_dir() / "evo2"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "benchmark_scores.jsonl"

    done: set[str] = set()
    if checkpoint.exists():
        for row in collect().to_dict("records"):
            done.add(row["ensembl_gene_id"])
        print(f"   resuming, {len(done):,} already scored")

    pending = table[~table["ensembl_gene_id"].isin(done)]
    if pending.empty:
        print("   nothing left to score")
        return collect()

    lock = threading.Lock()
    started = time.time()
    completed = 0
    fatal: str | None = None

    def worker(row) -> None:
        nonlocal completed, fatal
        if fatal:
            return
        try:
            record = score_gene(row["ensembl_gene_id"], row["sequence"], key, seed)
        except FatalRequestError as error:
            with lock:
                fatal = fatal or str(error)
            return
        except RuntimeError:
            # The retry budget ran out, which is a fact about the network rather
            # than about this gene. It is left out so a later run picks it up.
            return
        with lock:
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            completed += 1
            if completed % 100 == 0:
                rate = completed / max(time.time() - started, 1e-9)
                left = (len(pending) - completed) / rate if rate else float("nan")
                print(
                    f"   {completed:,} of {len(pending):,} at {rate * 60:.0f} genes a minute, "
                    f"about {left / 3600:.1f} hours left",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, (row for _, row in pending.iterrows())))

    if fatal:
        raise RuntimeError(fatal)
    print(f"   scored {completed:,} in {(time.time() - started) / 60:.1f} minutes")

    frame = collect()
    frame.to_parquet(out_dir / "benchmark_scores.parquet", index=False)
    return frame


def evaluate() -> dict:
    """Discrimination of DepMap essentiality under the benchmark's rule."""
    from src.diagnostics.power import rank_auroc
    from src.features import baseline

    scores = collect()
    scores = scores[scores["difference"].notna()] if "difference" in scores.columns else scores
    if scores.empty:
        return {"note": "nothing scored yet"}

    labels = baseline.load()[["ensembl_gene_id", "depmap_common_essential"]]
    joined = scores.merge(labels, on="ensembl_gene_id", how="inner")
    joined = joined[joined["depmap_common_essential"].notna()]
    essential = joined["depmap_common_essential"].astype(bool).to_numpy()

    # The two rules reach different gene sets, so the perturbation is isolated
    # only on the genes both of them score.
    from src.evo2 import score as scoring

    try:
        stop_rule = scoring.collect("scores")
    except (FileNotFoundError, ValueError):
        stop_rule = pd.DataFrame()
    if stop_rule.empty:
        # A run carried in from elsewhere arrives as the collected table rather
        # than as the checkpoint the run that produced it wrote.
        carried = config.interim_dir() / "evo2" / "scores.parquet"
        if carried.exists():
            stop_rule = pd.read_parquet(carried)
    shared = (
        joined[joined["ensembl_gene_id"].isin(set(stop_rule["ensembl_gene_id"]))]
        if not stop_rule.empty
        else joined.iloc[:0]
    )

    result = {
        "n_genes": int(len(joined)),
        "n_essential": int(essential.sum()),
        "auroc_total_difference": float(
            rank_auroc(joined["difference"].to_numpy(dtype=float), essential)
        ),
        "auroc_per_base": float(
            rank_auroc(joined["difference_per_base"].to_numpy(dtype=float), essential)
        ),
    }
    if not shared.empty:
        shared_essential = shared["depmap_common_essential"].astype(bool).to_numpy()
        result["n_genes_shared_with_stop_rule"] = int(len(shared))
        result["auroc_total_difference_shared"] = float(
            rank_auroc(shared["difference"].to_numpy(dtype=float), shared_essential)
        )
        result["auroc_per_base_shared"] = float(
            rank_auroc(shared["difference_per_base"].to_numpy(dtype=float), shared_essential)
        )
    spec = config.analysis().get("evo2", {})
    published = float(spec.get("published_auroc", 0.66))
    tolerance = float(spec.get("inclusion_tolerance", 0.03))
    best = max(result["auroc_total_difference"], result["auroc_per_base"])
    result["published_auroc"] = published
    result["inclusion_tolerance"] = tolerance
    result["meets_criterion"] = bool(abs(best - published) <= tolerance)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    if not args.evaluate_only:
        run(limit=args.limit, workers=args.workers, seed=args.seed)

    report = evaluate()
    print(json.dumps(report, indent=2))
    out = config.derived_dir() / "evo2_validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_rule.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
