"""Gene-level table: confounders and the constraint scores.

This is the join every later step reads. It carries the three confounders from
the frozen adjustment set, the primary and secondary constraint metrics, and
nothing that is a mediator or a collider. Conservation and the model-derived
scores arrive later on their own tracks and are joined on `ensembl_gene_id`.

The GeneBayes gene-feature release also ships conservation columns. They are
inputs to the model that produced s_het, so using them as the conservation arm
of the contrast would compare a score against a function of itself. Conservation
comes from the UCSC tracks directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

DECILE_COLUMNS = ("cds_length", "expected_lof")


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / f"{name}.parquet")


def load_constraint_v2() -> pd.DataFrame:
    """gnomAD v2.1.1 constraint, for the release-swap robustness arm."""
    frame = _read("gnomad_constraint_v2")
    columns = {
        "gene_id": "ensembl_gene_id",
        "oe_lof_upper": "loeuf_v2",
        "pLI": "pli_v2",
        "mis_z": "missense_z_v2",
    }
    present = {key: value for key, value in columns.items() if key in frame.columns}
    if "gene_id" not in frame.columns:
        return pd.DataFrame(columns=["ensembl_gene_id", "loeuf_v2"])
    frame = frame[list(present)].rename(columns=present)
    frame["ensembl_gene_id"] = frame["ensembl_gene_id"].astype(str).str.split(".").str[0]
    frame = frame[frame["ensembl_gene_id"].str.match(r"^ENSG\d{11}$", na=False)]
    return frame.drop_duplicates("ensembl_gene_id").reset_index(drop=True)


def load_constraint() -> pd.DataFrame:
    """gnomAD v4.1 constraint, one row per gene.

    The release is transcript level. MANE Select is preferred, then the
    canonical transcript, then the longest coding sequence.
    """
    frame = _read("gnomad_constraint_v4")
    columns = {
        "gene_id": "ensembl_gene_id",
        "gene": "gene_symbol",
        "lof.oe_ci.upper": "loeuf",
        "lof.pLI": "pli",
        "mis.z_score": "missense_z",
        "lof.exp": "expected_lof",
        "cds_length": "cds_length_gnomad",
    }
    present = {key: value for key, value in columns.items() if key in frame.columns}
    frame = frame[[*present, "mane_select", "canonical"]].rename(columns=present)
    frame["ensembl_gene_id"] = frame["ensembl_gene_id"].astype(str).str.split(".").str[0]

    # The release carries a RefSeq annotation alongside the Ensembl one, and the
    # RefSeq rows key gene_id on an NCBI gene identifier: A1BG appears as "1",
    # A1CF as "29974". Roughly half of all rows are RefSeq. Carrying them
    # forward puts eighteen thousand phantom genes into the table, keyed on
    # identifiers that can never join, and distorts the decile boundaries of the
    # matching covariates that are computed over it.
    frame = frame[frame["ensembl_gene_id"].str.match(r"^ENSG\d{11}$", na=False)]

    frame["_rank"] = (~frame["mane_select"].fillna(False)).astype(int) * 2 + (
        ~frame["canonical"].fillna(False)
    ).astype(int)
    ordered = frame.sort_values(
        ["ensembl_gene_id", "_rank", "cds_length_gnomad"], ascending=[True, True, False]
    )
    return ordered.drop_duplicates("ensembl_gene_id").drop(columns=["_rank"]).reset_index(drop=True)


def load_s_het() -> pd.DataFrame:
    frame = _read("genebayes_s_het")
    frame = frame.rename(
        columns={"ensg": "ensembl_gene_id", "post_mean": "s_het", "hgnc": "hgnc_id"}
    )
    frame["ensembl_gene_id"] = frame["ensembl_gene_id"].astype(str).str.split(".").str[0]
    if "hgnc_id" in frame.columns:
        frame["hgnc_id"] = frame["hgnc_id"].astype(str).str.replace("HGNC:", "", regex=False)
    keep = ["ensembl_gene_id", "hgnc_id", "chrom", "s_het", "post_lower_95", "post_upper_95"]
    return frame[[column for column in keep if column in frame.columns]].drop_duplicates(
        "ensembl_gene_id"
    )


def load_cds() -> pd.DataFrame:
    frame = _read("gencode_cds")
    return frame[["ensembl_gene_id", "gene_symbol", "cds_length", "gc_content"]]


def deciles(values: pd.Series) -> pd.Series:
    """Decile rank, robust to ties and to missing values."""
    ranked = values.rank(method="average", pct=True)
    return np.ceil(ranked * 10).clip(1, 10)


def build() -> pd.DataFrame:
    cds = load_cds()
    constraint = load_constraint()
    s_het = load_s_het()

    table = (
        cds.merge(
            constraint.drop(columns=["gene_symbol"], errors="ignore"),
            on="ensembl_gene_id",
            how="outer",
        )
        .merge(s_het, on="ensembl_gene_id", how="outer")
        .merge(load_constraint_v2(), on="ensembl_gene_id", how="left")
    )

    # CDS length is taken from GENCODE where available. gnomAD's own value fills
    # the gap so that a gene missing from the transcript FASTA still carries the
    # confounder rather than dropping out of every matched set.
    table["cds_length"] = table["cds_length"].fillna(table.get("cds_length_gnomad"))
    table = table.drop(columns=["cds_length_gnomad"], errors="ignore")

    table["s_het_log"] = np.log10(table["s_het"].clip(lower=1e-6))

    # Gene-family markers for the restriction arms. The HLA region and the
    # olfactory receptor family are excluded throughout the balancing-selection
    # work and are a named robustness restriction for the primary estimand.
    symbol = table["gene_symbol"].fillna("")
    table["is_hla"] = symbol.str.match(r"^HLA-", na=False)
    table["is_olfactory"] = symbol.str.match(r"^OR\d+[A-Z]", na=False)
    table["is_autosomal"] = ~table["chrom"].astype(str).isin(["X", "Y", "chrX", "chrY", "MT"])
    table["cds_length_decile"] = deciles(table["cds_length"])
    table["expected_lof_decile"] = deciles(table["expected_lof"])

    return table.sort_values("ensembl_gene_id").reset_index(drop=True)


def coverage(table: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"column": column, "non_null": int(table[column].notna().sum())}
        for column in table.columns
        if column != "ensembl_gene_id"
    ]
    report = pd.DataFrame(rows)
    report["fraction"] = (report["non_null"] / len(table)).round(4)
    return report.sort_values("non_null", ascending=False).reset_index(drop=True)


def run() -> pd.DataFrame:
    table = build()
    target = config.interim_dir() / "gene_table.parquet"
    table.to_parquet(target, index=False)
    print(f"gene_table: {len(table):,} genes")
    print(coverage(table).to_string(index=False))
    return table


def load() -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / "gene_table.parquet")


if __name__ == "__main__":
    config.ensure_dirs()
    run()
