"""The identifiability crosstab.

The primary estimands are score-by-onset interactions estimated within an
evidence source. They are identified only if diseases vary in onset inside each
source. If Mendelian evidence sits entirely in the early bins and GWAS evidence
entirely in the late ones, onset and evidence class are collinear and no amount
of modelling recovers the interaction.

The crosstab therefore records, for each source, which onset levels are
occupied. What matters is the diagonal, not the margins.

It is built with no minimum-gene-count filter. Most Mendelian diseases have one
causal gene; requiring five or ten would leave only genetically heterogeneous
syndromes, which are overwhelmingly early-onset, and would manufacture the
collinearity the crosstab exists to detect. The one-stage model in
src/models/primary.py pools gene-by-disease rows and does not need the filter.
The filter applies only to the per-disease points drawn on the figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config

RAW_ONSET_ORDER = [
    "antenatal",
    "neonatal",
    "infancy",
    "childhood",
    "adolescent",
    "adult",
    "elderly",
]


def onset_order() -> list[str]:
    """The five collapsed levels the Mendelian arm uses. See config/analysis.yaml."""
    return list(config.analysis()["onset"]["collapse"]["order"])


def decade_order() -> list[str]:
    """The decade bins the complex arm uses, on median age at first event."""
    edges = config.analysis()["onset"]["t2_t3_bins"]
    return [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)]


SOURCE_LABELS = {
    "a_mendelian": "Orphanet Mendelian",
    "b_gwas": "Open Targets GWAS L2G >= 0.5",
    "c_clinvar": "Open Targets ClinVar and somatic",
    "d_gene_burden": "Open Targets gene burden",
}


@dataclass
class CrosstabSummary:
    crosstab: pd.DataFrame
    per_source_bins: pd.DataFrame
    adult_mendelian_diseases: int


def build(positives: pd.DataFrame, order: list[str] | None = None) -> pd.DataFrame:
    """Cells of (number of diseases, gene-disease pairs, median genes per disease).

    positives requires the columns disease_id, gene_id, evidence_source and
    onset_bin. Rows without an onset assignment are kept under the bin label
    "unassigned" so that coverage gaps stay visible rather than silently
    shrinking the denominator.
    """
    required = {"disease_id", "gene_id", "evidence_source", "onset_bin"}
    missing = required - set(positives.columns)
    if missing:
        raise ValueError(f"positives is missing {sorted(missing)}")

    frame = positives.drop_duplicates(["disease_id", "gene_id", "evidence_source"]).copy()
    frame["onset_bin"] = frame["onset_bin"].fillna("unassigned")

    genes_per_disease = (
        frame.groupby(["evidence_source", "onset_bin", "disease_id"])["gene_id"]
        .nunique()
        .rename("n_genes")
        .reset_index()
    )

    cells = (
        genes_per_disease.groupby(["evidence_source", "onset_bin"])
        .agg(
            n_diseases=("disease_id", "nunique"),
            n_pairs=("n_genes", "sum"),
            median_genes_per_disease=("n_genes", "median"),
        )
        .reset_index()
    )

    levels = order or onset_order()
    position = {name: index for index, name in enumerate([*levels, "unassigned"])}
    cells["_order"] = cells["onset_bin"].map(position).fillna(len(position))
    return (
        cells.sort_values(["evidence_source", "_order"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def summarise(cells: pd.DataFrame) -> CrosstabSummary:
    """How many onset levels each evidence source occupies at the minimum sizes.

    A level counts as occupied when it holds at least the minimum number of
    diseases and gene-disease pairs set in config/analysis.yaml, and a source
    spans the onset range when it occupies at least the minimum number of
    levels. The adult Mendelian cell is reported separately because it holds
    every late-onset Mendelian disease.
    """
    rule = config.analysis()["crosstab"]

    occupied = cells[
        (cells["onset_bin"] != "unassigned")
        & (cells["n_diseases"] >= rule["min_diseases_per_bin"])
        & (cells["n_pairs"] >= rule["min_pairs_per_bin"])
    ]
    per_source = (
        occupied.groupby("evidence_source")["onset_bin"]
        .nunique()
        .rename("occupied_bins")
        .reset_index()
    )
    for source in SOURCE_LABELS:
        if source not in set(per_source["evidence_source"]):
            per_source.loc[len(per_source)] = [source, 0]
    per_source = per_source.sort_values("evidence_source").reset_index(drop=True)
    per_source["spans_range"] = per_source["occupied_bins"] >= rule["min_onset_bins_per_source"]

    adult_mendelian = cells[
        (cells["evidence_source"] == "a_mendelian") & (cells["onset_bin"] == "adult")
    ]
    return CrosstabSummary(
        crosstab=cells,
        per_source_bins=per_source,
        adult_mendelian_diseases=int(adult_mendelian["n_diseases"].sum()),
    )


def to_markdown(
    cells: pd.DataFrame, value: str = "n_diseases", order: list[str] | None = None
) -> str:
    wide = cells.pivot(index="onset_bin", columns="evidence_source", values=value)
    levels = [*(order or onset_order()), "unassigned"]
    rows = [bin_name for bin_name in levels if bin_name in wide.index]
    wide = wide.loc[rows].fillna(0).astype(int)

    header = ["onset bin", *wide.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for index, row in wide.iterrows():
        lines.append("| " + " | ".join([str(index), *(f"{v:,}" for v in row)]) + " |")
    return "\n".join(lines)
