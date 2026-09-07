"""Gene-class markers for the mediation and the positive control.

Early-onset disease genes are developmental, pleiotropic and haploinsufficient
by construction, so constraint may track gene class rather than the fitness cost
of the disease. Separating the two needs gene class measured independently of
onset. Two markers are built.

**Developmental role.** Measured from Gene Ontology annotation, taking the share
of a gene's biological-process terms that fall in the developmental-process
branch. A fetal transcriptome would be the direct measurement; the canonical
developmental series is deposited as alignments rather than as a joinable
per-gene matrix. A Gene Ontology annotation and a fetal expression level are not
the same measurement, and the substitution is stated wherever the marker is
used.

**Haploinsufficiency.** pLI, the probability that a gene is intolerant of losing
one copy. Already in the gene table from gnomAD.

Both are markers of gene class rather than of onset, which is what the mediation
requires. Neither is used in primary matching, because the causal graph
classifies gene class as a confounder to adjust for and not a covariate to match
on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config, ontology
from src.ingest import fetch, open_targets

GO_PREFIX = "GO"

# Declared by name and resolved against the released ontology. See src/ontology.
DEVELOPMENTAL_CONCEPTS = [
    "anatomical structure development",
    "cell fate commitment",
    "pattern specification process",
    "embryo development",
]


def ontology_path() -> Path:
    spec = config.sources()["gene_ontology"]
    entry = spec["files"]["basic"]
    dest = config.raw_dir() / "gene_ontology" / Path(entry["url"]).name
    return fetch.download(
        entry["url"],
        dest,
        source="gene_ontology",
        version=spec["version"],
        licence=spec["licence"],
        redistribute=spec["redistribute"],
    )


def developmental_terms() -> tuple[set[str], list[dict]]:
    """The developmental-process subtree, with what each concept resolved to."""
    go = ontology.load(str(ontology_path()), GO_PREFIX)
    terms: set[str] = set()
    for concept in DEVELOPMENTAL_CONCEPTS:
        terms |= go.subtree_by_name(concept)[1]
    return terms, ontology.provenance(go, DEVELOPMENTAL_CONCEPTS)


def build() -> pd.DataFrame:
    """One row per gene with its developmental and haploinsufficiency markers."""
    from src.features import gene_table

    terms, resolved = developmental_terms()

    target = open_targets.read("target", columns=["id", "biotype", "go"])
    target = target[target["biotype"] == "protein_coding"]

    rows = []
    for gene, terms_for_gene in zip(target["id"], target["go"], strict=True):
        if terms_for_gene is None:
            rows.append({"ensembl_gene_id": gene, "n_process_terms": 0, "n_developmental_terms": 0})
            continue
        process = [
            entry
            for entry in terms_for_gene
            if isinstance(entry, dict) and entry.get("aspect") == "P"
        ]
        developmental = [entry for entry in process if entry.get("id") in terms]
        rows.append(
            {
                "ensembl_gene_id": gene,
                "n_process_terms": len(process),
                "n_developmental_terms": len(developmental),
            }
        )

    table = pd.DataFrame(rows)
    table["is_developmental"] = table["n_developmental_terms"] > 0

    # A share rather than a count, because the count rises with how thoroughly a
    # gene has been annotated. The same adjustment is applied to the care-burden
    # component of the severity index and for the same reason.
    with np.errstate(invalid="ignore", divide="ignore"):
        table["developmental_share"] = np.where(
            table["n_process_terms"] > 0,
            table["n_developmental_terms"] / table["n_process_terms"],
            np.nan,
        )

    genes = gene_table.load()[["ensembl_gene_id", "pli"]]
    table = table.merge(genes, on="ensembl_gene_id", how="left")
    table = table.rename(columns={"pli": "haploinsufficiency"})
    table.attrs["resolved_concepts"] = resolved
    table.attrs["n_developmental_terms_in_subtree"] = len(terms)
    return table.drop_duplicates("ensembl_gene_id").reset_index(drop=True)


def run() -> pd.DataFrame:
    table = build()
    target = config.interim_dir() / "gene_class.parquet"
    table.to_parquet(target, index=False)

    subtree_size = table.attrs["n_developmental_terms_in_subtree"]
    developmental = int(table["is_developmental"].sum())

    print(f"gene_class: {len(table):,} protein-coding genes")
    print(f"   developmental terms in the subtree: {subtree_size:,}")
    print(f"   genes with any developmental annotation: {developmental:,}")
    with_pli = int(table["haploinsufficiency"].notna().sum())
    print(f"   genes with a haploinsufficiency score: {with_pli:,}")
    print()
    print("   resolved concepts")
    for row in table.attrs["resolved_concepts"]:
        print(
            f"      {row['concept']:<38} {row['term_id']:<12} subtree {row['n_terms_in_subtree']:,}"
        )
    print()
    print(
        table[["n_process_terms", "n_developmental_terms", "developmental_share"]]
        .describe()
        .to_string()
    )
    return table


def load() -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / "gene_class.parquet")


if __name__ == "__main__":
    config.ensure_dirs()
    run()
