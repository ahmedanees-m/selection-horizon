"""Cross-check the complex-disease onset axis against published onset clusters.

Dönertaş et al. clustered 116 UK Biobank diseases by their age-of-onset profile
into four clusters. If the T2 axis is measuring onset, its median age at first
event should order those clusters. Material disagreement means the onset
variable is wrong, not that theirs was.

The two disease vocabularies do not share identifiers. UK Biobank self-reported
conditions are free text ("heart/cardiac problem") and FinnGen endpoints are
curated composites with their own names ("Major coronary heart disease event").
Matching is therefore by normalised token overlap, reported with its score, and
restricted to matches above a threshold. Every accepted match is written out so
that it can be read and disputed rather than trusted.

The cluster numbering carries no guaranteed direction, so the correlation is
reported with its sign and the direction is read off the data rather than
assumed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config

MATCH_THRESHOLD = 0.5

STOPWORDS = frozenset(
    {
        "and",
        "or",
        "of",
        "the",
        "a",
        "an",
        "other",
        "unspecified",
        "nos",
        "not",
        "otherwise",
        "specified",
        "disease",
        "diseases",
        "disorder",
        "disorders",
        "problem",
        "problems",
        "event",
        "events",
        "major",
        "malignant",
        "neoplasm",
        "combined",
        "definitions",
        "definition",
        "strict",
        "wide",
        "all",
    }
)


# Tokens that change what a disease *is*, not merely how it is worded. A match
# is vetoed when one name carries one of these and the other does not, because
# token overlap alone will happily pair "stomach disorder" with "malignant
# neoplasm of stomach", which is not the same condition and would drag an onset
# comparison toward cancer ages.
KIND_TOKENS = {
    "neoplasm": {
        "neoplasm",
        "neoplasms",
        "cancer",
        "malignant",
        "carcinoma",
        "tumour",
        "tumor",
        "sarcoma",
        "lymphoma",
        "leukemia",
        "leukaemia",
    },
    "infection": {
        "infection",
        "infections",
        "infectious",
        "sepsis",
        "tuberculosis",
        "hepatitis",
        "pneumonia",
    },
    "injury": {"injury", "injuries", "fracture", "trauma", "wound", "poisoning"},
    "congenital": {"congenital", "malformation", "malformations"},
    "pregnancy": {"pregnancy", "obstetric", "puerperium", "delivery"},
}


# A numeral that subtypes a disease is stripped by the tokeniser, which drops
# anything shorter than three characters, so type 1 and type 2 diabetes look
# identical to token overlap. They are different diseases with onsets four
# decades apart, so the numeral is recovered separately and a disagreement
# between two stated subtypes vetoes the match.
SUBTYPE = re.compile(r"\btype\s*([0-9]+|i{1,3}v?|iv|vi{0,3})\b", re.IGNORECASE)

ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6"}


def subtypes(name: str) -> set[str]:
    """Stated disease subtypes, with roman numerals folded onto arabic ones."""
    found = set()
    for match in SUBTYPE.finditer(str(name)):
        value = match.group(1).lower()
        found.add(ROMAN.get(value, value))
    return found


def kinds(name: str) -> set[str]:
    lowered = str(name).lower()
    return {
        kind
        for kind, markers in KIND_TOKENS.items()
        if any(marker in lowered for marker in markers)
    }


def _tokens(name: str) -> set[str]:
    parts = re.split(r"[^a-z0-9]+", str(name).lower())
    return {part for part in parts if part and part not in STOPWORDS and len(part) > 2}


def similarity(query: str, candidate: str) -> float:
    """Jaccard overlap of informative tokens, under two vetoes.

    The argument order carries meaning. The query is the condition being placed
    and the candidate is the registry endpoint offered for it, and the second
    veto is asymmetric between them.
    """
    if kinds(query) != kinds(candidate):
        return 0.0

    left, right = subtypes(query), subtypes(candidate)
    if left and right and left != right:
        return 0.0

    a, b = _tokens(query), _tokens(candidate)
    if not a or not b:
        return 0.0

    # An endpoint whose informative tokens are a strict subset of the
    # condition's is a broader category than the condition, and its age
    # distribution is the broader category's. Type 1 diabetes matched to
    # diabetes mellitus takes on the age of type 2, postcoital bleeding matched
    # to bleeding takes on the age of every cause of bleeding, and motor neurone
    # disease matched to motor disorders takes on the age of a category it is a
    # small part of. The veto is one directional on purpose: a registry endpoint
    # routinely carries qualifying tokens the condition does not, and vetoing
    # that direction would reject the matches that are working.
    if b < a:
        return 0.0

    return len(a & b) / len(a | b)


def load_clusters(path: Path | None = None) -> pd.DataFrame:
    """The 116 diseases with their age-of-onset cluster."""
    source = path or config.raw_dir() / "donertas" / "ukbbmetadata.tsv"
    frame = pd.read_csv(source, sep="\t")
    frame = frame.rename(
        columns={
            "Disease": "disease",
            "Age-of-onset Cluster": "cluster",
            "Number of Cases": "n_cases",
            "Disease Category": "category",
        }
    )
    return frame[["disease", "cluster", "n_cases", "category"]].dropna(subset=["cluster"])


def match(
    clusters: pd.DataFrame, t2: pd.DataFrame, threshold: float = MATCH_THRESHOLD
) -> pd.DataFrame:
    """Best-scoring Risteys endpoint for each clustered disease, above threshold."""
    usable = t2[t2["usable_for_onset"] & t2["longname"].notna()]
    if usable.empty:
        return pd.DataFrame(
            columns=[
                "disease",
                "cluster",
                "endpoint",
                "longname",
                "score",
                "median_age_first_event",
            ]
        )

    endpoints = list(
        zip(usable["endpoint"], usable["longname"], usable["median_age_first_event"], strict=True)
    )

    rows = []
    for _, entry in clusters.iterrows():
        best = max(endpoints, key=lambda item: similarity(entry["disease"], item[1]))
        score = similarity(entry["disease"], best[1])
        if score < threshold:
            continue
        rows.append(
            {
                "disease": entry["disease"],
                "cluster": int(entry["cluster"]),
                "endpoint": best[0],
                "longname": best[1],
                "score": round(score, 3),
                "median_age_first_event": best[2],
            }
        )
    return pd.DataFrame(rows).drop_duplicates("endpoint").reset_index(drop=True)


# Cluster 4 holds diseases whose incidence is roughly flat with age: nine of its
# ten members are infections or respiratory infections. It is not part of the
# age-of-onset gradient the other three describe, so the correlation is reported
# both across all four clusters and across the three that are age related.
AGE_INDEPENDENT_CLUSTER = 4


def crosscheck(matched: pd.DataFrame) -> dict:
    """Whether the onset axis recovers the published cluster structure.

    Two questions, because they are not the same. Whether individual diseases
    rank together, which a Spearman correlation answers and which free-text
    matching across two different cohorts will attenuate. And whether the
    cluster medians order correctly, which is what recovering the structure
    actually means and which the plan is really asking about.
    """
    usable = matched.dropna(subset=["median_age_first_event", "cluster"])
    if len(usable) < 10:
        return {"n_matched": int(len(usable)), "spearman": None, "verdict": "too few matches"}

    result = spearmanr(usable["cluster"], usable["median_age_first_event"])
    rho = float(result.statistic)

    age_related = usable[usable["cluster"] != AGE_INDEPENDENT_CLUSTER]
    restricted = spearmanr(age_related["cluster"], age_related["median_age_first_event"])

    summary = (
        age_related.groupby("cluster")["median_age_first_event"]
        .agg(["count", "median"])
        .reset_index()
        .sort_values("cluster")
    )
    medians = summary["median"].to_numpy()
    # Cluster numbering carries no guaranteed direction, so monotone either way
    # counts as recovering the ordering.
    ordered = bool(len(medians) > 1 and (all(np.diff(medians) < 0) or all(np.diff(medians) > 0)))

    by_cluster = (
        usable.groupby("cluster")["median_age_first_event"]
        .agg(["count", "median"])
        .reset_index()
        .to_dict(orient="records")
    )

    magnitude = abs(float(restricted.statistic))
    if ordered and magnitude >= 0.4:
        verdict = "recovers the published ordering"
    elif ordered:
        verdict = "orders correctly but the disease-level correlation is weak"
    elif magnitude >= 0.4:
        verdict = "correlates but the cluster medians do not order"
    else:
        verdict = "disagrees"

    return {
        "n_matched": int(len(usable)),
        "spearman": round(rho, 4),
        "p_value": float(result.pvalue),
        "spearman_age_related_clusters": round(float(restricted.statistic), 4),
        "p_value_age_related_clusters": float(restricted.pvalue),
        "n_age_related": int(len(age_related)),
        "cluster_medians_ordered": ordered,
        "direction": (
            "later clusters have later onset" if rho > 0 else "later clusters have earlier onset"
        ),
        "by_cluster": by_cluster,
        "verdict": verdict,
    }


def run(threshold: float = MATCH_THRESHOLD) -> dict:
    from src.onset import complex_onset

    clusters = load_clusters()
    t2 = complex_onset.load()

    matched = match(clusters, t2, threshold)
    out_dir = config.derived_dir() / "analysis_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(out_dir / "donertas_matches.csv", index=False)

    report = crosscheck(matched)
    print(
        f"Donertas cross-check: {len(clusters)} clustered diseases, {report['n_matched']} matched"
    )
    if report["spearman"] is not None:
        rho, pvalue = report["spearman"], report["p_value"]
        print(f"   spearman across all clusters      = {rho:+.4f}, p = {pvalue:.4g}")
        print(
            f"   spearman across age-related ones  = "
            f"{report['spearman_age_related_clusters']:+.4f}, "
            f"p = {report['p_value_age_related_clusters']:.4g}, "
            f"n = {report['n_age_related']}"
        )
        print(f"   cluster medians order monotonically: {report['cluster_medians_ordered']}")
        print(f"   {report['direction']}")
        print(f"   verdict: {report['verdict']}")
        print()
        print("   median age at first event by cluster")
        for row in report["by_cluster"]:
            count, median = row["count"], row["median"]
            print(f"      cluster {row['cluster']}: n = {count:>3}, median = {median:.1f}")
        print()
        print("   matches, highest scoring first")
        for _, row in matched.sort_values("score", ascending=False).head(12).iterrows():
            print(
                f"      {row['disease'][:34]:<36} -> {row['longname'][:34]:<36} "
                f"score {row['score']:.2f}"
            )
    else:
        print(f"   {report['verdict']}")
    return report


if __name__ == "__main__":
    config.ensure_dirs()
    run()
