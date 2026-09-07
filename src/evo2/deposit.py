"""Write the per-gene Evo 2 scores as a standalone deposit.

The scores are released whether or not the arm survives into the analysis. It did
not, and the file is written anyway, because the measurement cost 67,000 forward
passes against a 40B model.

The header records which model produced the scores, over what window, under which
insertion rule, against which gene annotation, how they performed on the
validation, and that the insertion rule is this project's own rather than one
taken from the published Methods.
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import pandas as pd

from src import config
from src.evo2 import score as scoring

COLUMNS = [
    "ensembl_gene_id",
    "evo2_score",
    "wild_type_log_likelihood",
    "mean_delta",
    "delta_spread",
]

FILENAME = "evo2_human_gene_scores_v1.tsv"


def header_lines(table: pd.DataFrame, validation: dict) -> list[str]:
    gencode = config.sources()["gencode"]
    scored = table[table["evo2_score"].notna()]

    auroc = validation.get("evo2_auroc")
    published = validation.get("published_auroc")

    return [
        "# Evo 2 stop-codon likelihood scores for human protein-coding genes",
        f"# generated: {date.today().isoformat()}",
        "#",
        "# model: evo2-40b, hosted forward endpoint, unembedding layer",
        "#   The hosted endpoint serves a fixed checkpoint and does not expose a",
        "#   checkpoint hash, so the model is identified by name and endpoint only.",
        "#",
        f"# annotation: GENCODE release {gencode['release']}, canonical transcripts, GRCh38",
        f"# window: {scoring.WINDOW} bp centred on the coding start, minus-strand genes"
        " reverse complemented",
        "#",
        "# insertion rule: TGA replaces the codon at nucleotide offsets"
        f" {', '.join(str(o) for o in scoring.STOP_OFFSETS)} from the coding start,"
        " which are codons 6, 11 and 16.",
        "#   This rule was specified by this project. The protocol published with the",
        "#   model was not obtained, so these scores are not a reimplementation of it",
        "#   and any comparison against published figures is between two rules rather",
        "#   than between two runs of one rule.",
        "#",
        "# score: mean over the three insertion positions of the difference in sequence",
        "#   log likelihood between the mutant and the wild-type window, divided by the",
        "#   window length. Negative by construction.",
        "#",
        f"# genes attempted: {len(table):,}",
        f"# genes scored: {len(scored):,}",
        "# genes excluded before scoring: 3,639 of 20,446 canonical coding starts, whose",
        "#   first coding exon is shorter than the furthest insertion offset. That set is",
        "#   enriched for DepMap essential genes, at 13.75 per cent against 9.06 per cent",
        "#   among the genes scored here.",
        "#",
        (
            "# validation: separation of DepMap common-essential genes,"
            f" AUROC {auroc:.4f} against a published {published:.2f}"
            if auroc is not None and published is not None
            else "# validation: not available"
        ),
        "#   The arm did not clear its validation threshold and contributes no score to",
        "#   the analysis it was built for. See the repository for what was ruled out.",
        "#",
        "# licence: CC-BY-4.0",
        "# repository: https://github.com/ahmedanees-m/selection-horizon",
    ]


def build() -> tuple[pd.DataFrame, list[str]]:
    table = scoring.load()
    for column in COLUMNS:
        if column not in table.columns:
            raise KeyError(f"the score table lacks {column}")

    record = config.derived_dir() / "evo2_validation" / "validation.json"
    validation = json.loads(record.read_text(encoding="utf-8")) if record.exists() else {}

    out = table[COLUMNS].copy()
    out = out[out["evo2_score"].notna()].sort_values("ensembl_gene_id").reset_index(drop=True)
    for column in ("evo2_score", "wild_type_log_likelihood", "mean_delta", "delta_spread"):
        out[column] = out[column].map(lambda value: f"{value:.6g}")
    return out, header_lines(table, validation)


def run() -> str:
    out, header = build()
    target = config.derived_dir() / "evo2" / FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for line in header:
            handle.write(line)
            handle.write("\n")
        out.to_csv(handle, sep="\t", index=False, lineterminator="\n")

    print(f"wrote {target}")
    print(f"   {len(out):,} genes, {len(header)} header lines")
    print(f"   {target.stat().st_size / 1024:.0f} KB")
    return str(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the Evo 2 score deposit")
    parser.parse_args()
    config.ensure_dirs()
    run()


if __name__ == "__main__":
    main()
