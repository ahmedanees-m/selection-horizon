"""A generic OBO reader, so that ontology terms are resolved rather than recalled.

A wrong ontology identifier changes a result by a plausible amount and fails no
downstream assertion. One was already found in this project, where HP:0011712
was taken for sudden cardiac death when it is complete right bundle branch
block, and a second where a concept was written under a name the ontology does
not use.

The response is not to be more careful with identifiers, it is to stop writing
them down. Term sets are declared by concept name, resolved against the released
ontology file at load time, and expanded over the ontology's own subclass
relations. The resolved identifier is carried into the output so that a reader
can check what was actually used.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

IS_OBSOLETE = re.compile(r"^is_obsolete: true$", re.M)
NAME = re.compile(r"^name: (.+)$", re.M)
# Cross-references to other vocabularies. Only the identifier is kept; the
# trailing provenance an OBO xref may carry is not needed to match on.
XREF = re.compile(r"^xref: ([A-Za-z0-9]+:[A-Za-z0-9.]+)", re.M)
# Only exact synonyms are kept. Broad and related synonyms in Mondo name a
# different concept than the term does, so treating them as names for it
# would resolve a condition onto its parent or onto a sibling.
SYNONYM = re.compile(r"^synonym: \"(.+?)\" EXACT", re.M)


@dataclass(frozen=True)
class Ontology:
    """Term names and subclass relations for one ontology release."""

    name: dict[str, str]
    parents: dict[str, tuple[str, ...]]
    children: dict[str, tuple[str, ...]]
    prefix: str
    synonyms: dict[str, tuple[str, ...]] = field(default_factory=dict)
    xrefs: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def label(self, term: str) -> str:
        if term not in self.name:
            raise KeyError(f"{term} is not a term in this {self.prefix} release")
        return self.name[term]

    def resolve(self, label: str) -> str:
        """The single term carrying this exact name, case-insensitively."""
        wanted = label.strip().lower()
        matches = [term for term, name in self.name.items() if name.lower() == wanted]
        if not matches:
            raise KeyError(f"no {self.prefix} term is named {label!r}")
        if len(matches) > 1:
            raise KeyError(f"{label!r} is ambiguous in {self.prefix}: {sorted(matches)}")
        return matches[0]

    def descendants(self, term: str, include_self: bool = True) -> set[str]:
        if term not in self.name:
            raise KeyError(f"{term} is not a term in this {self.prefix} release")
        seen = {term} if include_self else set()
        frontier = [term]
        while frontier:
            current = frontier.pop()
            for child in self.children.get(current, ()):
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        return seen

    def ancestors(self, term: str, include_self: bool = True) -> set[str]:
        if term not in self.name:
            raise KeyError(f"{term} is not a term in this {self.prefix} release")
        seen = {term} if include_self else set()
        frontier = [term]
        while frontier:
            current = frontier.pop()
            for parent in self.parents.get(current, ()):
                if parent not in seen:
                    seen.add(parent)
                    frontier.append(parent)
        return seen

    def by_any_name(self) -> dict[str, str]:
        """Lower-cased primary names and synonyms, to the term they name.

        A name shared by two terms is dropped rather than assigned to one of
        them, because a lookup that silently picks a winner is worse than a
        lookup that reports a miss.
        """
        counts: dict[str, set[str]] = {}
        for term, label in self.name.items():
            counts.setdefault(label.strip().lower(), set()).add(term)
        for term, values in self.synonyms.items():
            for value in values:
                counts.setdefault(value.strip().lower(), set()).add(term)
        return {label: next(iter(terms)) for label, terms in counts.items() if len(terms) == 1}

    def by_xref(self, prefix: str) -> dict[str, str]:
        """Cross-references of one vocabulary, to the term carrying them.

        A cross-reference claimed by two terms is dropped, on the same rule the
        name index uses: a lookup that picks a winner silently is worse than one
        that reports a miss.
        """
        counts: dict[str, set[str]] = {}
        for term, references in self.xrefs.items():
            for reference in references:
                if reference.startswith(f"{prefix}:"):
                    counts.setdefault(reference, set()).add(term)
        return {key: next(iter(terms)) for key, terms in counts.items() if len(terms) == 1}

    def subtree_by_name(self, label: str) -> tuple[str, set[str]]:
        term = self.resolve(label)
        return term, self.descendants(term)


@lru_cache(maxsize=8)
def load(path: str, prefix: str, follow_part_of: bool = True) -> Ontology:
    """Parse an OBO release, keeping only terms of the given identifier prefix.

    Subsumption follows `is_a` and, by default, `part_of`. Gene Ontology relies
    on `part_of` heavily for developmental processes: heart development is
    part_of anatomical structure development rather than a subclass of it, so a
    reader that follows only `is_a` returns a subtree an order of magnitude too
    small and silently loses most of the class it was asked for.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    term_block = re.compile(rf"^id: ({re.escape(prefix)}:\d+)$(.*?)(?=^\[|\Z)", re.M | re.S)
    is_a = re.compile(rf"^is_a: ({re.escape(prefix)}:\d+)", re.M)
    part_of = re.compile(rf"^relationship: part_of ({re.escape(prefix)}:\d+)", re.M)

    name: dict[str, str] = {}
    parents: dict[str, tuple[str, ...]] = {}
    children: dict[str, list[str]] = defaultdict(list)
    synonyms: dict[str, tuple[str, ...]] = {}
    xrefs: dict[str, tuple[str, ...]] = {}

    for term, body in term_block.findall(text):
        if IS_OBSOLETE.search(body):
            continue
        label = NAME.search(body)
        if label is None:
            continue
        name[term] = label.group(1).strip()
        above = tuple(is_a.findall(body))
        if follow_part_of:
            above = tuple(dict.fromkeys([*above, *part_of.findall(body)]))
        parents[term] = above
        for parent in above:
            children[parent].append(term)
        found = SYNONYM.findall(body)
        if found:
            synonyms[term] = tuple(dict.fromkeys(found))
        references = XREF.findall(body)
        if references:
            xrefs[term] = tuple(dict.fromkeys(references))

    return Ontology(
        name=name,
        parents=parents,
        children={key: tuple(value) for key, value in children.items()},
        prefix=prefix,
        synonyms=synonyms,
        xrefs=xrefs,
    )


def provenance(ontology: Ontology, labels: list[str]) -> list[dict]:
    """What each concept name resolved to, for the record."""
    rows = []
    for label in labels:
        term, subtree = ontology.subtree_by_name(label)
        rows.append(
            {
                "concept": label,
                "term_id": term,
                "ontology_label": ontology.label(term),
                "n_terms_in_subtree": len(subtree),
            }
        )
    return rows
