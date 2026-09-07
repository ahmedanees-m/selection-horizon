"""Mode of inheritance.

s_het is the selection coefficient on heterozygotes. A recessive disease gene
carries almost no heterozygote fitness cost, so the primary constraint metric is
close to blind to it by construction. Pooling dominant and recessive diseases
therefore dilutes the constraint arm of the contrast, and it does so unevenly
along the onset axis: the dominant share of Orphanet diseases rises from roughly
three in ten in the perinatal bin to seven in ten in the adult bin.

That makes inheritance mode a confounder of the H1 contrast, and one that biases
Gamma away from zero rather than toward it. It is treated as a stratifier from
the outset rather than as a robustness arm.

Orphanet records inheritance as a pipe-separated list, since a disease may be
reported under more than one mode. A disease carrying both dominant and
recessive reports is labelled mixed rather than forced into one.
"""

from __future__ import annotations

import pandas as pd

DOMINANT = "dominant"
RECESSIVE = "recessive"
MIXED = "mixed"
OTHER = "other or unknown"

ORDER = [DOMINANT, RECESSIVE, MIXED, OTHER]


def classify(inheritance: pd.Series) -> pd.Series:
    """Collapse the Orphanet inheritance list to one label per disease."""
    text = inheritance.fillna("").str.lower()
    has_dominant = text.str.contains("dominant")
    has_recessive = text.str.contains("recessive")

    labels = pd.Series(OTHER, index=inheritance.index, dtype=object)
    labels[has_dominant & ~has_recessive] = DOMINANT
    labels[has_recessive & ~has_dominant] = RECESSIVE
    labels[has_dominant & has_recessive] = MIXED
    return labels


def by_onset(diseases: pd.DataFrame, onset_order: list[str]) -> pd.DataFrame:
    """Composition table: inheritance mode against collapsed onset bin.

    The dominant share by bin is the quantity that matters, because it is the
    part of any apparent constraint trend that inheritance alone would produce.
    """
    table = pd.crosstab(diseases["onset_bin"], diseases["inheritance_mode"])
    for column in ORDER:
        if column not in table.columns:
            table[column] = 0
    table = table[ORDER]
    table = table.reindex([b for b in onset_order if b in table.index])
    table["total"] = table.sum(axis=1)
    resolved = table[DOMINANT] + table[RECESSIVE]
    table["dominant_share"] = (table[DOMINANT] / resolved.where(resolved > 0)).round(3)
    return table.reset_index()
