"""The evolutionary feature matrix.

Two score families, kept apart because the contrast between them is the primary
estimand.

    Constraint      recent human population selection. s_het is primary because
                    roughly a third of coding genes are underpowered for LOEUF
                    even at gnomAD scale. LOEUF, pLI and missense Z are secondary
    Conservation    deep-time cross-species selection. Zoonomia 241-way phyloP is
                    primary, phyloP-100way and phastCons-100way secondary

Each conservation track contributes both aggregations: the fraction of coding
bases above a track-specific threshold, which is primary, and the mean at
four-fold degenerate sites, which is the robustness arm. Swapping between them
is a pre-registered arm rather than a choice made after seeing a result.

Scores are standardised within the analysis set at model time, not here, because
the analysis set differs by evidence source and stratum.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from src import config
from src.features import gene_table

TRACKS = ("phylop_241way", "phylop_100way", "phastcons_100way")

CONSTRAINT_SCORES = ("s_het_log", "loeuf", "pli", "missense_z")
CONSERVATION_PRIMARY = "phylop_241way_fraction_above"
CONSTRAINT_PRIMARY = "s_het_log"

# Higher is more constrained or more conserved for every score except LOEUF,
# where a lower upper bound on the observed to expected ratio means more
# constraint. The sign is carried here so that no downstream step has to
# remember it.
SCORE_DIRECTION = {
    "s_het_log": 1,
    "loeuf": -1,
    "pli": 1,
    "missense_z": 1,
    "phylop_241way_fraction_above": 1,
    "phylop_100way_fraction_above": 1,
    "phastcons_100way_fraction_above": 1,
    "phylop_241way_mean_fourfold": 1,
    "phylop_100way_mean_fourfold": 1,
    "phastcons_100way_mean_fourfold": 1,
}


def conservation_columns() -> list[str]:
    return [f"{track}_fraction_above" for track in TRACKS] + [
        f"{track}_mean_fourfold" for track in TRACKS
    ]


def load_track(track: str) -> pd.DataFrame:
    path = config.interim_dir() / f"conservation_{track}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{track} has not been aggregated; run src.ingest.conservation_tracks"
        )
    frame = pd.read_parquet(path)
    keep = ["ensembl_gene_id"] + [column for column in frame.columns if column.startswith(track)]
    return frame[keep]


def build() -> pd.DataFrame:
    table = gene_table.load()
    keep = ["ensembl_gene_id", "gene_symbol", "cds_length", "gc_content", "expected_lof"]
    keep += [column for column in CONSTRAINT_SCORES if column in table.columns]
    keep += [
        column for column in ("cds_length_decile", "expected_lof_decile") if column in table.columns
    ]
    matrix = table[keep].copy()

    for track in TRACKS:
        matrix = matrix.merge(load_track(track), on="ensembl_gene_id", how="left")

    return matrix.sort_values("ensembl_gene_id").reset_index(drop=True)


def family_correlations(matrix: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between every constraint and conservation score.

    Gamma is a difference between two families, so the correlation between them
    drives its variance. It is reported rather than assumed, and it is what the
    power simulation is calibrated against.
    """
    rows = []
    conservation = [column for column in conservation_columns() if column in matrix.columns]
    for constraint in CONSTRAINT_SCORES:
        if constraint not in matrix.columns:
            continue
        for conserved in conservation:
            subset = matrix[[constraint, conserved]].dropna()
            if len(subset) < 100:
                continue
            rho = spearmanr(subset[constraint], subset[conserved]).statistic
            rows.append(
                {
                    "constraint": constraint,
                    "conservation": conserved,
                    "n": len(subset),
                    "spearman": round(float(rho), 4),
                    "oriented": round(
                        float(rho) * SCORE_DIRECTION[constraint] * SCORE_DIRECTION[conserved], 4
                    ),
                }
            )
    return pd.DataFrame(rows)


def coverage(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "feature": column,
            "non_null": int(matrix[column].notna().sum()),
            "fraction": round(float(matrix[column].notna().mean()), 4),
        }
        for column in matrix.columns
        if column != "ensembl_gene_id"
    ]
    return pd.DataFrame(rows).sort_values("non_null", ascending=False).reset_index(drop=True)


def analysis_set(matrix: pd.DataFrame) -> pd.DataFrame:
    """Genes carrying both primary scores and all three confounders.

    The primary estimand is a contrast between the two families, so a gene
    missing either side contributes nothing to it and is reported as excluded
    rather than silently imputed.
    """
    required = [
        CONSTRAINT_PRIMARY,
        CONSERVATION_PRIMARY,
        "cds_length",
        "gc_content",
        "expected_lof",
    ]
    return matrix.dropna(subset=required)


def run() -> pd.DataFrame:
    matrix = build()
    target = config.interim_dir() / "evolutionary_features.parquet"
    matrix.to_parquet(target, index=False)

    usable = analysis_set(matrix)
    print(f"evolutionary_features: {len(matrix):,} genes, {len(usable):,} in the analysis set")
    print()
    print(coverage(matrix).to_string(index=False))
    print()
    print("Between-family Spearman correlations")
    print(family_correlations(matrix).to_string(index=False))
    return matrix


def load() -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / "evolutionary_features.parquet")


if __name__ == "__main__":
    config.ensure_dirs()
    run()
