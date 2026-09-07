"""The balancing-selection arm.

Purifying-selection scores are tested for an onset gradient elsewhere in the
analysis. This arm asks the same question of a balancing-selection statistic.

The statistic is the standardised B2 of Siewert and Voight, computed on 1000
Genomes data for three populations. It is the balancing-selection statistic with
a published genome-wide release, which matters because the alternative is
recomputing one and defending the recomputation.

Two related statistics are not computed. Tajima's D would need per-site allele
frequencies across the whole genome and allele-age against frequency discordance
would need inferred allele ages, and neither has a per-gene release.

Coordinates matter here. The B2 release is on GRCh37, because that is what 1000
Genomes phase 3 is on, while everything else in this project is on GRCh38. The
gene spans for this arm come from the GENCODE GRCh37 mapping rather than from a
liftover of the scores, so no coordinate is converted twice.

The HLA region and the olfactory receptor family are excluded throughout, as the
plan requires: both are under balancing selection for reasons that have nothing
to do with disease onset, and both would dominate any genome-wide ranking.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.ingest import fetch

POPULATIONS = ("CEU", "YRI", "CHB")
GENE_LINE = re.compile(r'gene_id "([^"]+)"')

# The three population archives do not share a layout. CEU and CHB hold
# StdB2Scores/chrN_B2std.out; YRI holds AllB2StdScores/chrN_B2.out. Matching on
# the chromosome prefix rather than on the file suffix reads all three, and the
# column check below is what confirms the statistic is the standardised one
# rather than raw B2.
SCORE_MEMBER = re.compile(r"^chr([0-9]+|X|Y)_B2[^/]*\.out$")
SCORE_COLUMN = "Beta2_std"

NOT_COMPUTED = {
    "tajimas_d": (
        "needs per-site allele frequencies genome wide; no per-gene release exists and "
        "the compute is not justified for a tertiary arm"
    ),
    "allele_age_frequency_discordance": ("needs inferred allele ages; no per-gene release exists"),
    "brigos_barril_positive_control": (
        "the fertility-increasing, disease-increasing allele set is in a journal "
        "supplement and is not machine readable; the arm is reported without its "
        "ready-made positive control until that set is obtained"
    ),
}


def _spec() -> dict:
    return config.sources()["betascan"]


def download(population: str) -> Path:
    spec = _spec()
    entry = spec["files"][population]
    dest = config.raw_dir() / "betascan" / f"{population}StdB2.tar.gz"
    return fetch.download(
        entry["url"],
        dest,
        source="betascan",
        version=spec["version"],
        licence=spec["licence"],
        redistribute=spec["redistribute"],
    )


def gene_spans_grch37() -> pd.DataFrame:
    """Gene spans on GRCh37, which is the assembly the B2 release is on."""
    spec = config.sources()["gencode"]
    entry = spec["files"]["annotation_grch37"]
    path = fetch.download(
        entry["url"],
        config.raw_dir() / "gencode" / Path(entry["url"]).name,
        source="gencode",
        version=spec["release"],
        licence=spec["licence"],
        redistribute=spec["redistribute"],
    )

    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            match = GENE_LINE.search(fields[8])
            if match is None:
                continue
            rows.append(
                {
                    "ensembl_gene_id": match.group(1).split(".")[0],
                    "chrom": fields[0].replace("chr", ""),
                    "start": int(fields[3]),
                    "end": int(fields[4]),
                }
            )
    frame = pd.DataFrame(rows).drop_duplicates("ensembl_gene_id")
    return frame[frame["chrom"].str.match(r"^\d+$", na=False)].reset_index(drop=True)


def read_scores(path: Path) -> dict[str, pd.DataFrame]:
    """Per-chromosome standardised B2 scores out of the released archive."""
    scores: dict[str, pd.DataFrame] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            match = SCORE_MEMBER.match(Path(member.name).name)
            if match is None:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            frame = pd.read_csv(handle, sep="\t")
            if SCORE_COLUMN not in frame.columns:
                raise ValueError(
                    f"{member.name} in {path.name} has no {SCORE_COLUMN} column; "
                    "this release is not the standardised statistic"
                )
            frame = frame.dropna(subset=["Position", SCORE_COLUMN])
            scores[match.group(1)] = frame.sort_values("Position").reset_index(drop=True)

    if not scores:
        raise ValueError(f"no per-chromosome score files found in {path.name}")
    return scores


def aggregate(
    spans: pd.DataFrame, scores: dict[str, pd.DataFrame], population: str
) -> pd.DataFrame:
    """Per-gene summaries of the standardised B2 over sites inside the gene span.

    The maximum is the statistic of interest, because balancing selection acts
    at a locus rather than across a whole gene, and a mean over the span would
    dilute a real peak into the neutral sites around it. The count of scored
    sites travels with it, because a maximum over three sites is not the same
    quantity as a maximum over three hundred.
    """
    rows = []
    for chromosome, block in spans.groupby("chrom"):
        table = scores.get(str(chromosome))
        if table is None or table.empty:
            continue
        positions = table["Position"].to_numpy()
        values = table[SCORE_COLUMN].to_numpy()

        starts = np.searchsorted(positions, block["start"].to_numpy(), side="left")
        ends = np.searchsorted(positions, block["end"].to_numpy(), side="right")

        for gene, low, high in zip(block["ensembl_gene_id"], starts, ends, strict=True):
            if high <= low:
                continue
            window = values[low:high]
            rows.append(
                {
                    "ensembl_gene_id": gene,
                    f"b2_max_{population}": float(np.nanmax(window)),
                    f"b2_q95_{population}": float(np.nanpercentile(window, 95)),
                    f"b2_n_sites_{population}": int(window.size),
                }
            )

    columns = [
        "ensembl_gene_id",
        f"b2_max_{population}",
        f"b2_q95_{population}",
        f"b2_n_sites_{population}",
    ]
    # An empty frame still has to carry its columns, or the merge that combines
    # populations fails on a missing key rather than reporting that one
    # population produced nothing.
    return pd.DataFrame(rows, columns=columns) if not rows else pd.DataFrame(rows)


def build(populations: tuple[str, ...] = POPULATIONS) -> pd.DataFrame:
    spans = gene_spans_grch37()

    table: pd.DataFrame | None = None
    for population in populations:
        scores = read_scores(download(population))
        summary = aggregate(spans, scores, population)
        print(
            f"   {population}: {len(scores)} chromosomes scored, "
            f"{len(summary):,} genes summarised"
        )
        if summary.empty:
            continue
        table = (
            summary if table is None else table.merge(summary, on="ensembl_gene_id", how="outer")
        )

    if table is None:
        raise ValueError("no population produced a gene summary; check the downloads")

    maxima = [column for column in table.columns if column.startswith("b2_max_")]
    table["b2_max_across_populations"] = table[maxima].max(axis=1)

    from src.features import gene_table

    families = gene_table.load()[["ensembl_gene_id", "is_hla", "is_olfactory"]]
    table = table.merge(families, on="ensembl_gene_id", how="left")
    table["excluded_family"] = table["is_hla"].fillna(False) | table["is_olfactory"].fillna(False)
    return table.reset_index(drop=True)


def onset_profile(table: pd.DataFrame) -> pd.DataFrame:
    """Balancing-selection signal per onset bin, on the Mendelian arm.

    The purifying-selection scores are expected to decay with onset. The point of
    this arm is whether this one does. It is reported as a profile rather than a
    single number, because a tertiary arm with no positive control should not be
    compressed into a test statistic.
    """
    positives = pd.read_parquet(config.positives_path())
    mendelian = positives[
        (positives["evidence_source"] == "a_mendelian") & positives["onset_bin"].notna()
    ]

    joined = mendelian.merge(
        table.rename(columns={"ensembl_gene_id": "gene_id"}), on="gene_id", how="inner"
    )
    joined = joined[~joined["excluded_family"].fillna(False)]

    from src.onset import tiers

    order = tiers.collapsed_order()
    summary = (
        joined.groupby("onset_bin")["b2_max_across_populations"]
        .agg(["count", "median", "mean"])
        .reindex([b for b in order if b in set(joined["onset_bin"])])
        .reset_index()
    )
    return summary


def onset_gradient(table: pd.DataFrame, seed: int, draws: int = 400) -> dict:
    """Association between the balancing statistic and onset, on the Mendelian arm.

    Reported as a rank correlation and as a slope on onset scaled to the unit
    interval, both resampled over diseases, with the standard deviation of the
    statistic so the slope can be read against it.
    """
    from scipy.stats import spearmanr

    from src.models import primary

    positives = pd.read_parquet(config.positives_path())
    mendelian = positives[
        (positives["evidence_source"] == "a_mendelian") & positives["onset_bin"].notna()
    ]
    joined = mendelian.merge(
        table.rename(columns={"ensembl_gene_id": "gene_id"}), on="gene_id", how="inner"
    )
    joined = joined[~joined["excluded_family"].fillna(False)]

    onset = primary.onset_position(joined["onset_bin"])
    values = joined["b2_max_across_populations"].to_numpy(dtype=float)
    keep = np.isfinite(onset) & np.isfinite(values)
    onset, values = onset[keep], values[keep]
    diseases = joined.loc[keep, "disease_id"].to_numpy()

    def statistics(index: np.ndarray) -> tuple[float, float]:
        rho = float(spearmanr(values[index], onset[index]).statistic)
        centred = onset[index] - onset[index].mean()
        denominator = float((centred**2).sum())
        slope = (
            float((centred * values[index]).sum() / denominator) if denominator else float("nan")
        )
        return rho, slope

    everything = np.arange(onset.size)
    observed_rho, observed_slope = statistics(everything)

    unique = np.unique(diseases)
    positions = {name: np.flatnonzero(diseases == name) for name in unique}
    rng = np.random.default_rng(seed)
    rhos, slopes = [], []
    for _ in range(draws):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        index = np.concatenate([positions[name] for name in drawn])
        rho, slope = statistics(index)
        if np.isfinite(rho) and np.isfinite(slope):
            rhos.append(rho)
            slopes.append(slope)

    def interval(sample: list[float]) -> tuple[float, float]:
        array = np.array(sample)
        if array.size <= 20:
            return float("nan"), float("nan")
        low, high = np.percentile(array, [2.5, 97.5])
        return float(low), float(high)

    rho_low, rho_high = interval(rhos)
    slope_low, slope_high = interval(slopes)
    return {
        "n_rows": int(onset.size),
        "n_diseases": int(unique.size),
        "standard_deviation": float(np.std(values, ddof=1)),
        "spearman_with_onset": observed_rho,
        "spearman_ci_low": rho_low,
        "spearman_ci_high": rho_high,
        "slope_on_onset": observed_slope,
        "slope_ci_low": slope_low,
        "slope_ci_high": slope_high,
        "draws": len(rhos),
    }


def run() -> pd.DataFrame:
    table = build()
    target = config.interim_dir() / "balancing_selection.parquet"
    table.to_parquet(target, index=False)

    print(f"balancing_selection: {len(table):,} genes with a standardised B2 summary")
    print(f"   excluded gene families, HLA and olfactory: {int(table['excluded_family'].sum()):,}")
    columns = [c for c in table.columns if c.startswith("b2_max_")]
    print()
    print(table[columns].describe().to_string())

    profile = onset_profile(table)
    gradient: dict = {}
    if not profile.empty:
        print()
        print("   balancing-selection signal by onset bin, Mendelian arm")
        print(profile.to_string(index=False))

        gradient = onset_gradient(table, config.analysis()["seed"])
        print()
        print("   association with onset")
        for name, value in gradient.items():
            print(f"      {name}: {value}")

    out_dir = config.derived_dir() / "balancing"
    out_dir.mkdir(parents=True, exist_ok=True)
    profile.to_parquet(out_dir / "onset_profile.parquet", index=False)
    with (out_dir / "onset_gradient.json").open("w", encoding="utf-8") as handle:
        json.dump(gradient, handle, indent=2, default=float)
        handle.write(chr(10))

    print()
    print("   statistics not computed, and why")
    for name, reason in NOT_COMPUTED.items():
        print(f"      {name}: {reason}")
    return table


def load() -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / "balancing_selection.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate balancing-selection statistics")
    parser.add_argument("--populations", nargs="*", default=list(POPULATIONS))
    args = parser.parse_args()
    config.ensure_dirs()
    build(tuple(args.populations))
    run()


if __name__ == "__main__":
    main()
