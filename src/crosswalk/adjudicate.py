"""Adjudication of the queued FinnGen to Platform mappings.

The crosswalk sorts every mapping into three tiers. Confident mappings pass a
name-agreement check against the identifier route that produced them, rejected
mappings point at something that is not a disease, and the rest are queued. The
queue is 428 mappings, 350 of which carry an onset, so leaving it queued costs
roughly a third of the complex arm.

The plan calls for these to be adjudicated by hand. Reading them shows that most
do not need a judgement so much as a lookup. The queue holds two kinds of thing
in roughly equal measure, and they are told apart by the disease ontology rather
than by an opinion.

The first kind is correct and merely differently named. Cholera against vibrio
infectious disease, rabies against Rhabdoviridae infectious disease, erysipeloid
against Erysipelothrix rhusiopathiae infectious disease: the registry names the
condition and the Platform names the pathogen, they share no words, and one is
an ancestor of the other in Mondo.

The second kind is wrong. Dermatophytosis against response to statin, protozoal
diseases against myeloproliferative neoplasm. These share no words either, and
in the ontology they share nothing else: neither is an ancestor of the other and
the branches they sit in do not meet anywhere useful.

So the adjudication is a relatedness test in Mondo, and what it cannot decide it
leaves queued rather than guessing. Three outcomes, all recorded with a reason:

    accepted    the target is an ancestor or a descendant of the condition
    rejected    the target is not a disease, or is unrelated to the condition
    queued      the condition does not resolve, so nothing can be said

The name resolution is deliberately strict. Only a Mondo primary name or an
exact synonym counts, and a name shared by two terms resolves to neither. A
fuzzy resolution here would accept mappings on the strength of a guess, which is
the failure this whole tier exists to avoid.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src import config, ontology
from src.ingest import fetch

MONDO_PREFIX = "MONDO"

# The root of the Mondo disease hierarchy. Its immediate children are the
# top-level categories a disease belongs to, and two terms in disjoint
# categories are about different things.
MONDO_ROOT = "MONDO:0000001"

# Namespaces in the Platform disease index that do not hold diseases. OBA is the
# ontology of biological attributes and GO is a process ontology; a mapping onto
# either is a mapping onto something a person cannot be diagnosed with.
NON_DISEASE_NAMESPACES = ("OBA", "GO")

# How far apart in the hierarchy a condition and its mapped target may sit. A
# registry endpoint and its ontology counterpart should be the same concept or
# one step of generalisation from it. Retained intraocular foreign body mapped
# to eye disorder is five steps, and accepting it would attach the gene
# annotations of every eye disorder to one endpoint's onset.
MAX_STEPS = 2

ACCEPTED = "accepted"
REJECTED = "rejected"
QUEUED = "queued"

BRACKETED = re.compile(r"[\[(][^\])]*[\])]")
TRAILING_QUALIFIER = re.compile(
    r",?\s*(not otherwise specified|unspecified|nos|other|and related \w+)\s*$", re.IGNORECASE
)
LEADING_QUALIFIER = re.compile(r"^(other|unspecified|certain)\s+", re.IGNORECASE)


def mondo_path() -> Path:
    spec = config.sources()["mondo"]
    entry = spec["files"]["ontology"]
    return fetch.download(
        entry["url"],
        config.raw_dir() / "mondo" / Path(entry["url"]).name,
        source="mondo",
        version=spec["version"],
        licence=spec["licence"],
        redistribute=spec["redistribute"],
    )


def normalise(name: str) -> list[str]:
    """Candidate lookup forms for a registry condition name, most specific first.

    FinnGen longnames are ICD phrasing, which carries qualifiers an ontology
    name does not: bracketed abbreviations, a leading other, a trailing
    unspecified. Each is stripped in turn and every form is tried, so a name
    resolves on the most specific form that matches rather than on the shortest.
    """
    text = str(name or "").strip()
    forms = [text]
    without_brackets = BRACKETED.sub(" ", text)
    forms.append(without_brackets)
    forms.append(TRAILING_QUALIFIER.sub("", without_brackets))
    forms.append(LEADING_QUALIFIER.sub("", TRAILING_QUALIFIER.sub("", without_brackets)))

    seen: list[str] = []
    for form in forms:
        cleaned = re.sub(r"\s+", " ", form).strip().strip(",").lower()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def _term(identifier: str) -> str:
    """Platform identifiers use an underscore; OBO files use a colon."""
    return str(identifier).replace("_", ":", 1)


def branches(mondo: ontology.Ontology, term: str) -> set[str]:
    """The top-level Mondo categories a term belongs to."""
    top = set(mondo.children.get(MONDO_ROOT, ()))
    return mondo.ancestors(term) & top


def steps_between(mondo: ontology.Ontology, lower: str, upper: str, limit: int) -> int | None:
    """Shortest subsumption path from a term up to one of its ancestors."""
    frontier = {lower}
    for distance in range(limit + 1):
        if upper in frontier:
            return distance
        frontier = {parent for term in frontier for parent in mondo.parents.get(term, ())}
        if not frontier:
            return None
    return None


def adjudicate_row(
    row, mondo: ontology.Ontology, lookup: dict[str, str], by_doid: dict[str, str]
) -> dict:
    target = str(row.get("disease_id") or "")
    namespace = target.split("_")[0]

    if namespace in NON_DISEASE_NAMESPACES:
        return {
            "verdict": REJECTED,
            "reason": f"the target is a {namespace} term, which is not a disease",
            "condition_term": None,
            "route": None,
        }

    # The Disease Ontology identifier Risteys publishes for the endpoint is
    # tried first. It is an identifier rather than a name, so it resolves
    # endpoints whose ICD phrasing has no ontology equivalent, which is most of
    # them. The name route remains as the fallback.
    resolved = by_doid.get(str(row.get("doid") or ""))
    route = "doid" if resolved else None
    if resolved is None:
        for form in normalise(row.get("longname")):
            if form in lookup:
                resolved = lookup[form]
                route = "name"
                break

    if resolved is None:
        return {
            "verdict": QUEUED,
            "reason": "the condition name does not resolve to a single Mondo term",
            "condition_term": None,
            "route": None,
        }

    if namespace != MONDO_PREFIX:
        return {
            "verdict": QUEUED,
            "reason": f"the target is a {namespace} term and is not in the Mondo release",
            "condition_term": resolved,
            "route": route,
        }

    term = _term(target)
    if term not in mondo.name:
        return {
            "verdict": QUEUED,
            "reason": "the target is not in this Mondo release",
            "condition_term": resolved,
            "route": route,
        }

    if term == resolved:
        return {
            "verdict": ACCEPTED,
            "reason": f"the target is {mondo.label(resolved)}",
            "condition_term": resolved,
            "route": route,
        }

    upward = steps_between(mondo, resolved, term, MAX_STEPS)
    if upward is not None:
        return {
            "verdict": ACCEPTED,
            "reason": f"the target is {upward} step(s) above {mondo.label(resolved)}",
            "condition_term": resolved,
            "route": route,
        }
    downward = steps_between(mondo, term, resolved, MAX_STEPS)
    if downward is not None:
        return {
            "verdict": ACCEPTED,
            "reason": f"the target is {downward} step(s) below {mondo.label(resolved)}",
            "condition_term": resolved,
            "route": route,
        }

    if term in mondo.ancestors(resolved) or term in mondo.descendants(resolved):
        return {
            "verdict": QUEUED,
            "reason": (
                f"the target is related to {mondo.label(resolved)} but more than "
                f"{MAX_STEPS} steps away"
            ),
            "condition_term": resolved,
            "route": route,
        }

    # Absence of a subsumption path is not evidence of unrelatedness. Mondo
    # relates an infection to its pathogen through predicates this reader does
    # not follow, so rabies and Rhabdoviridae infectious disease have no
    # ancestry between them and are plainly the same subject. A mapping is
    # rejected only when the two terms sit in disjoint top-level categories,
    # which is a positive statement that they are about different things, and
    # is queued otherwise.
    left, right = branches(mondo, resolved), branches(mondo, term)
    if left and right and not (left & right):
        return {
            "verdict": REJECTED,
            "reason": (
                f"{mondo.label(resolved)} and the target sit in disjoint Mondo categories: "
                f"{sorted(mondo.label(item) for item in left)[0]} against "
                f"{sorted(mondo.label(item) for item in right)[0]}"
            ),
            "condition_term": resolved,
            "route": route,
        }
    return {
        "verdict": QUEUED,
        "reason": f"no subsumption path to {mondo.label(resolved)}, and the categories overlap",
        "condition_term": resolved,
    }


def run() -> pd.DataFrame:
    crosswalk = pd.read_parquet(config.interim_dir() / "crosswalk" / "finngen_to_platform.parquet")
    queued = crosswalk[crosswalk["tier"] == "review"].copy()
    print(f"queued mappings to adjudicate: {len(queued):,}")

    onset = pd.read_parquet(config.interim_dir() / "onset_t2.parquet")[["endpoint", "doid"]]
    queued = queued.merge(onset, on="endpoint", how="left")

    mondo = ontology.load(str(mondo_path()), MONDO_PREFIX)
    lookup = mondo.by_any_name()
    by_doid = mondo.by_xref("DOID")
    print(
        f"Mondo release: {len(mondo.name):,} terms, {len(lookup):,} unambiguous names, "
        f"{len(by_doid):,} unambiguous Disease Ontology cross-references"
    )
    with_doid = int(queued["doid"].notna().sum())
    print(f"   queued mappings carrying a Disease Ontology identifier: {with_doid:,}")

    verdicts = [adjudicate_row(row, mondo, lookup, by_doid) for _, row in queued.iterrows()]
    queued["verdict"] = [item["verdict"] for item in verdicts]
    queued["verdict_reason"] = [item["reason"] for item in verdicts]
    queued["condition_term"] = [item["condition_term"] for item in verdicts]
    queued["resolution_route"] = [item.get("route") for item in verdicts]

    print()
    print(queued["verdict"].value_counts().to_string())
    print()
    print("   with a usable onset, by verdict")
    print(queued.groupby("verdict")["usable_for_onset"].sum().to_string())
    print()
    print("   why, in the words of the adjudication")
    print(
        queued["verdict_reason"].str.replace(r" of .*$", "", regex=True).value_counts().to_string()
    )

    for verdict in (ACCEPTED, REJECTED):
        sample = queued[queued["verdict"] == verdict].head(6)
        if sample.empty:
            continue
        print()
        print(f"   {verdict}, first few")
        for _, row in sample.iterrows():
            print(f"      {str(row['longname'])[:44]:<44} -> {str(row['disease_name'])[:34]:<34}")

    out_dir = config.derived_dir() / "crosswalk"
    out_dir.mkdir(parents=True, exist_ok=True)
    queued.to_parquet(out_dir / "adjudicated.parquet", index=False)
    with (out_dir / "adjudicated.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_queued": int(len(queued)),
                "counts": queued["verdict"].value_counts().to_dict(),
                "usable_for_onset": queued.groupby("verdict")["usable_for_onset"].sum().to_dict(),
            },
            handle,
            indent=2,
            default=int,
        )
        handle.write("\n")

    crosswalk = apply_to_crosswalk(queued)
    print()
    print("   crosswalk after adjudication")
    print(crosswalk["tier"].value_counts().to_string())
    usable = int((crosswalk["confident"] & crosswalk["usable_for_onset"]).sum())
    print(f"   confident and usable on the onset axis: {usable:,}")
    return queued


def apply_to_crosswalk(queued: pd.DataFrame) -> pd.DataFrame:
    """Fold the verdicts back into the crosswalk the rest of the pipeline reads.

    An accepted mapping joins the confident tier and a rejected one joins the
    rejected tier. What stays queued stays queued: it is not promoted on the
    grounds that the adjudication could not decide it. The column recording
    which rows were adjudicated is kept, so a confident mapping that arrived
    this way can always be told from one that passed the name check.
    """
    path = config.interim_dir() / "crosswalk" / "finngen_to_platform.parquet"
    table = pd.read_parquet(path)

    verdicts = queued.set_index("endpoint")["verdict"]
    table["adjudicated"] = table["endpoint"].map(verdicts)

    accepted = table["adjudicated"] == ACCEPTED
    rejected = table["adjudicated"] == REJECTED
    table.loc[accepted, "tier"] = "confident"
    table.loc[rejected, "tier"] = "rejected"
    table["confident"] = table["tier"] == "confident"

    table.to_parquet(path, index=False)
    return table


def load() -> pd.DataFrame:
    return pd.read_parquet(config.derived_dir() / "crosswalk" / "adjudicated.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adjudicate the queued crosswalk mappings")
    parser.parse_args()
    config.ensure_dirs()
    run()


if __name__ == "__main__":
    main()
