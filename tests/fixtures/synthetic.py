"""Synthetic positives and features with a known generating Gamma.

Pipeline development runs against this rather than against the real gene sets,
so that the primary estimands are not repeatedly inspected before the
pre-registration is filed.

The generator mirrors the realised design: five collapsed onset levels, a
disease-count profile matching the Mendelian arm, mostly one gene per disease,
and a dominant share that rises with onset, which is the confound the real data
carries. Conservation loses `gamma` AUROC-equivalent points of discrimination
across the onset range while constraint holds its own, so a correct pipeline
recovers a positive Gamma of about that size and a null design recovers zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

ONSET_LEVELS = ["perinatal", "infancy", "childhood", "adolescent", "adult"]
DISEASES_PER_LEVEL = [350, 100, 120, 40, 80]
DOMINANT_SHARE = [0.32, 0.26, 0.41, 0.57, 0.71]

CONSERVATION_COLUMN = "phylop_241way_fraction_above"
CONSTRAINT_COLUMN = "s_het_log"


def auroc_to_d(auroc: float) -> float:
    return float(np.sqrt(2.0) * norm.ppf(auroc))


def generate(
    gamma: float = 0.10,
    base_auroc: float = 0.70,
    score_correlation: float = 0.6,
    genes_per_disease: int = 1,
    n_background_genes: int = 4000,
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (positives, features) with a known generating Gamma."""
    rng = np.random.default_rng(seed)

    n_positive_genes = sum(DISEASES_PER_LEVEL) * genes_per_disease
    n_genes = n_positive_genes + n_background_genes
    gene_ids = np.array([f"ENSG{index:08d}" for index in range(n_genes)])

    covariance = np.array([[1.0, score_correlation], [score_correlation, 1.0]])
    noise = rng.multivariate_normal(np.zeros(2), covariance, size=n_genes)

    rows = []
    conservation_shift = np.zeros(n_genes)
    constraint_shift = np.zeros(n_genes)

    cursor = 0
    span = max(len(ONSET_LEVELS) - 1, 1)
    for level, (name, count, share) in enumerate(
        zip(ONSET_LEVELS, DISEASES_PER_LEVEL, DOMINANT_SHARE, strict=True)
    ):
        position = level / span
        conservation_auroc = np.clip(base_auroc - gamma * position, 0.05, 0.95)
        for index in range(count):
            disease = f"SYN_{name}_{index:04d}"
            mode = "dominant" if rng.random() < share else "recessive"
            for _ in range(genes_per_disease):
                gene = gene_ids[cursor]
                conservation_shift[cursor] = auroc_to_d(conservation_auroc)
                constraint_shift[cursor] = auroc_to_d(base_auroc)
                rows.append(
                    {
                        "disease_id": disease,
                        "gene_id": gene,
                        "evidence_source": "synthetic",
                        "onset_bin": name,
                        "inheritance_mode": mode,
                    }
                )
                cursor += 1

    features = pd.DataFrame(
        {
            "ensembl_gene_id": gene_ids,
            CONSERVATION_COLUMN: noise[:, 0] + conservation_shift,
            CONSTRAINT_COLUMN: noise[:, 1] + constraint_shift,
            "cds_length": rng.lognormal(7.0, 0.8, size=n_genes),
            "gc_content": rng.beta(5, 5, size=n_genes),
            "expected_lof": rng.gamma(2.0, 20.0, size=n_genes),
        }
    )
    features["cds_length_decile"] = np.ceil(features["cds_length"].rank(pct=True) * 10).clip(1, 10)
    features["expected_lof_decile"] = np.ceil(features["expected_lof"].rank(pct=True) * 10).clip(
        1, 10
    )

    return pd.DataFrame(rows), features
