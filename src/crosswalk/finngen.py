"""FinnGen endpoints to Platform disease terms.

Each Risteys endpoint page carries an EFO identifier. Joining those straight
onto the Platform disease index reaches 79 of 674 endpoints, which looks like a
failure of the route and is not. It is the failure the execution plan predicted:
release 26.06 ships EFO 3.88.0, in which a block of EFO disease terms was
replaced by Mondo terms, and FinnGen's annotations are pinned to a release
before that replacement. Checked against EFO itself, the identifier for coronary
artery disease, for type II diabetes and for actinomycosis are all now marked
obsolete.

The Platform tracks its own replacements in an `obsoleteTerms` field, so an
obsolete EFO identifier resolves to the Mondo term that superseded it. Routing
through that lifts coverage from 79 to 479 of 674, and the replacements are
semantically right: actinomycosis to actinomycosis, anthrax to anthrax
infection.

A second problem does not dissolve. Some FinnGen annotations point at terms that
are simply wrong for the endpoint: benign prostate neoplasm is annotated
EFO_0021510, which EFO itself calls a linoleate measurement. Those are not
deprecated identifiers, they are incorrect ones. Every mapping is therefore
scored on agreement between the endpoint name and the target term name, and
disagreements are flagged rather than carried silently into the analysis.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd

from src import config
from src.ingest import open_targets

# Two views of name agreement, because they fail on different things. Token
# overlap misses spelling variants: amoebiasis and amebiasis share no tokens.
# Character similarity misses reorderings and abbreviations. A mapping passes on
# either.
TOKEN_THRESHOLD = 0.15
CHARACTER_THRESHOLD = 0.55

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
        "disease",
        "diseases",
        "disorder",
        "disorders",
        "infection",
        "infections",
        "icd",
        "icd9",
        "strict",
        "wide",
        "finngen",
        "including",
        "with",
    }
)

# Terms whose name marks them as a measurement rather than a disease. A FinnGen
# endpoint resolving to one of these is a mis-annotation, not a rare phenotype.
MEASUREMENT_MARKERS = ("measurement", "ratio", "level", "count", "intensity")


def _tokens(name: str) -> set[str]:
    parts = re.split(r"[^a-z0-9]+", str(name).lower())
    return {part for part in parts if part and part not in STOPWORDS and len(part) > 2}


def token_agreement(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def character_agreement(left: str, right: str) -> float:
    a = " ".join(sorted(_tokens(left)))
    b = " ".join(sorted(_tokens(right)))
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def replacement_index() -> pd.DataFrame:
    """Obsolete Platform term to the term that superseded it."""
    disease = open_targets.read("disease", columns=["id", "name", "obsoleteTerms"])
    exploded = disease.explode("obsoleteTerms").dropna(subset=["obsoleteTerms"])
    return (
        exploded.rename(
            columns={"obsoleteTerms": "old_id", "id": "disease_id", "name": "disease_name"}
        )[["old_id", "disease_id", "disease_name"]]
        .drop_duplicates("old_id")
        .reset_index(drop=True)
    )


def build(t2: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per endpoint carrying an EFO identifier, with how it resolved."""
    from src.onset import complex_onset

    t2 = t2 if t2 is not None else complex_onset.load()
    have = (
        t2[t2["efo_id"].notna()][
            [
                "endpoint",
                "longname",
                "efo_id",
                "median_age_first_event",
                "n_cases",
                "usable_for_onset",
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    current = open_targets.read("disease", columns=["id", "name"]).rename(
        columns={"id": "disease_id", "name": "disease_name"}
    )
    replacements = replacement_index()

    # Both right-hand keys are unique, so a left merge preserves row order and
    # count and the two frames stay positionally aligned.
    direct = have.merge(
        current.drop_duplicates("disease_id"), left_on="efo_id", right_on="disease_id", how="left"
    )
    resolved = direct["disease_id"].notna().to_numpy()
    direct["route"] = "direct"

    via = have.loc[~resolved].merge(replacements, left_on="efo_id", right_on="old_id", how="left")
    via["route"] = "obsolete_replacement"
    via = via.drop(columns=["old_id"], errors="ignore")

    table = pd.concat([direct.loc[resolved], via], ignore_index=True)
    table.loc[table["disease_id"].isna(), "route"] = "unresolved"

    pairs = list(zip(table["longname"], table["disease_name"], strict=True))
    table["token_agreement"] = [
        token_agreement(left, right) if isinstance(right, str) else float("nan")
        for left, right in pairs
    ]
    table["character_agreement"] = [
        character_agreement(left, right) if isinstance(right, str) else float("nan")
        for left, right in pairs
    ]
    table["target_is_measurement"] = (
        table["disease_name"]
        .fillna("")
        .str.lower()
        .str.contains("|".join(MEASUREMENT_MARKERS), regex=True)
    )
    table["resolved"] = table["disease_id"].notna()

    names_agree = (table["token_agreement"] >= TOKEN_THRESHOLD) | (
        table["character_agreement"] >= CHARACTER_THRESHOLD
    )

    # Three tiers, not two. A mapping to a broader parent term is neither
    # obviously right nor obviously wrong, and the execution plan already
    # requires manual adjudication of any FinnGen endpoint entering the primary
    # analysis. That queue is what the review tier is.
    table["tier"] = "rejected"
    table.loc[table["resolved"] & ~table["target_is_measurement"], "tier"] = "review"
    table.loc[table["resolved"] & ~table["target_is_measurement"] & names_agree, "tier"] = (
        "confident"
    )
    table["confident"] = table["tier"] == "confident"
    return table.sort_values("endpoint").reset_index(drop=True)


def coverage(table: pd.DataFrame) -> dict:
    return {
        "endpoints_with_efo": int(len(table)),
        "resolved_direct": int((table["route"] == "direct").sum()),
        "resolved_via_replacement": int((table["route"] == "obsolete_replacement").sum()),
        "unresolved": int((~table["resolved"]).sum()),
        "target_is_a_measurement": int(table["target_is_measurement"].sum()),
        "needing_adjudication": int((table["tier"] == "review").sum()),
        "confident": int(table["confident"].sum()),
        "confident_and_usable_for_onset": int(
            (table["confident"] & table["usable_for_onset"]).sum()
        ),
    }


def run() -> pd.DataFrame:
    table = build()
    out_dir = config.interim_dir() / "crosswalk"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / "finngen_to_platform.parquet", index=False)

    counts = coverage(table)
    print("FinnGen endpoints to Platform disease terms")
    for key, value in counts.items():
        print(f"   {key.replace('_', ' ')}: {value:,}")

    rejected = table[(table["tier"] == "rejected") & table["resolved"]]
    if not rejected.empty:
        print()
        print("   rejected: the target is a measurement, not a disease")
        for _, row in rejected.head(8).iterrows():
            print(
                f"      {str(row['longname'])[:38]:<40} {row['efo_id']:<14} -> "
                f"{str(row['disease_name'])[:36]}"
            )

    review = table[table["tier"] == "review"]
    if not review.empty:
        print()
        print("   queued for adjudication: resolved, but the names do not agree")
        for _, row in review.head(8).iterrows():
            print(
                f"      {str(row['longname'])[:38]:<40} {row['efo_id']:<14} -> "
                f"{str(row['disease_name'])[:36]}"
            )
    return table


def load() -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / "crosswalk" / "finngen_to_platform.parquet")


if __name__ == "__main__":
    config.ensure_dirs()
    run()
