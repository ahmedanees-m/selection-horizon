"""Evo 2 scoring against the published human essentiality benchmark.

Measures how well the stop-codon likelihood ratio separates DepMap essential
genes from the rest, and compares the result with the reference AUROC in
config/analysis.yaml. The reference was computed at 40B parameters, which is the
model size this arm scores at.

Coding length and GC content are scored alongside, because essential genes are
longer and differ in base composition, and a score that recovers only those two
facts carries no information about essentiality. The output reports the Evo 2
figure next to what each baseline achieves alone and next to a model holding all
three.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.diagnostics.power import rank_auroc

PUBLISHED_AUROC = 0.66
TOLERANCE = 0.03
MIN_GENES = 500

BASELINES = {
    "coding_length": "cds_length",
    "gc_content": "gc_content",
}


def labelled_scores() -> pd.DataFrame:
    """Evo 2 scores joined to essentiality labels and the baseline features."""
    from src.evo2 import score as scoring
    from src.features import baseline, evolutionary

    scores = scoring.load()
    scores = scores[scores["evo2_score"].notna()][["ensembl_gene_id", "evo2_score"]]

    essentiality = baseline.load()[
        ["ensembl_gene_id", "depmap_common_essential", "depmap_gene_effect"]
    ]
    features = evolutionary.load()[["ensembl_gene_id", "cds_length", "gc_content"]]

    table = scores.merge(essentiality, on="ensembl_gene_id", how="inner")
    table = table.merge(features, on="ensembl_gene_id", how="left")
    table["essential"] = table["depmap_common_essential"].fillna(False).astype(bool)
    return table.dropna(subset=["evo2_score"]).reset_index(drop=True)


def logistic_auroc(x: np.ndarray, y: np.ndarray, seed: int, folds: int = 5) -> float:
    """Out-of-fold discrimination of a logistic model on the given columns."""
    from src.models.log_r import cross_validated_auroc

    return cross_validated_auroc(x, y, seed, folds)


def evaluate(table: pd.DataFrame, seed: int) -> dict:
    labels = table["essential"].to_numpy()
    positives = int(labels.sum())
    if len(table) < MIN_GENES or positives < 20:
        return {
            "n_genes": int(len(table)),
            "n_essential": positives,
            "note": "too few scored genes or too few essential ones to evaluate",
        }

    # A more negative score means the model is more surprised by the premature
    # stop, so the score is negated before it is read as a ranking of
    # essentiality. Getting this backwards would report one minus the answer.
    evo2 = -table["evo2_score"].to_numpy()

    results: dict[str, object] = {
        "n_genes": int(len(table)),
        "n_essential": positives,
        "essential_fraction": float(labels.mean()),
        "evo2_auroc": float(rank_auroc(evo2, labels)),
    }
    for name, column in BASELINES.items():
        values = table[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        results[f"{name}_auroc"] = (
            float(rank_auroc(values[finite], labels[finite])) if finite.sum() > MIN_GENES else None
        )

    columns = ["cds_length", "gc_content"]
    matrix = table[columns].to_numpy(dtype=float)
    usable = np.all(np.isfinite(matrix), axis=1)
    if usable.sum() > MIN_GENES:
        results["baselines_together_auroc"] = logistic_auroc(
            matrix[usable], labels[usable].astype(float), seed
        )
        with_evo2 = np.column_stack([matrix[usable], evo2[usable]])
        results["baselines_plus_evo2_auroc"] = logistic_auroc(
            with_evo2, labels[usable].astype(float), seed
        )

    difference = abs(float(results["evo2_auroc"]) - PUBLISHED_AUROC)  # type: ignore[arg-type]
    results["published_auroc"] = PUBLISHED_AUROC
    results["difference_from_published"] = float(difference)
    results["tolerance"] = TOLERANCE
    results["within_tolerance"] = bool(difference <= TOLERANCE)
    return results


def run() -> dict:
    settings = config.analysis()
    table = labelled_scores()
    results = evaluate(table, settings["seed"])

    print("Evo 2 essentiality benchmark")
    for name, value in results.items():
        if isinstance(value, float):
            print(f"   {name}: {value:.4f}")
        else:
            print(f"   {name}: {value}")

    if "within_tolerance" in results:
        print()
        if results["within_tolerance"]:
            print("   the arm reproduces the published figure and is released")
        else:
            print("   the arm does not reproduce the published figure and is not released")
            print("   the stop-codon rule and the context window are the first things to check")

    out_dir = config.derived_dir() / "evo2_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=float)
        handle.write("\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Evo 2 against the benchmark")
    parser.parse_args()
    config.ensure_dirs()
    run()


if __name__ == "__main__":
    main()
