"""Per-case severity from the HPO clinical course sub-ontology.

Orphanet publishes an age of onset interval but no age of death, so `kappa_d`
in the fitness-cost model takes its interval from the HPO Age of death branch
(HP:0011420) instead.

Term identifiers and labels are checked against `hp.obo` at load time rather
than trusted from this file, because a mislabelled identifier here would place
deaths in the wrong decade and would not show up in any downstream assertion.
HP:0005268 is Miscarriage, not a young-adult death term, which is the specific
mistake the check exists to catch.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src import config

AGE_OF_DEATH_ROOT = "HP:0011420"

# Age at death interval in years, one entry per term under HP:0011420 that
# places death in an interval. Labels are asserted against hp.obo.
DEATH_INTERVALS = {
    "HP:0003811": ("Neonatal death", 0.0, 0.0767),
    "HP:0001522": ("Death in infancy", 0.0767, 2.0),
    "HP:0003819": ("Death in childhood", 2.0, 10.0),
    "HP:0011421": ("Death in adolescence", 10.0, 20.0),
    "HP:0100613": ("Death in early adulthood", 20.0, 40.0),
    "HP:0033763": ("Death in adulthood", 20.0, 65.0),
    "HP:0033764": ("Death in middle age", 40.0, 65.0),
    "HP:0033765": ("Death in late adulthood", 65.0, 85.0),
}

# Loss before birth. Reproduction is removed entirely, so kappa_d is 1 by
# construction and no interval is needed.
PRENATAL_LOSS = {
    "HP:0003826": "Stillbirth",
    "HP:0005268": "Miscarriage",
    "HP:0034241": "Prenatal death",
}

# Lethality without an age. These widen the sensitivity band and never set a
# point estimate.
UNLOCATED_SEVERITY = {
    "HP:0001699": "Sudden death",
    "HP:0001645": "Sudden cardiac death",
    "HP:0033258": "Sudden unexpected death in epilepsy",
}

TERM_PATTERN = re.compile(r"^id: (HP:\d+)\nname: (.+)$", re.M)


def ontology_labels(obo_path: Path | None = None) -> dict[str, str]:
    path = obo_path or config.raw_dir() / "hpo" / "hp.obo"
    text = path.read_text(encoding="utf-8", errors="replace")
    return dict(TERM_PATTERN.findall(text))


def verify_labels(obo_path: Path | None = None) -> None:
    """Fail loudly if a term identifier no longer carries the expected label."""
    labels = ontology_labels(obo_path)
    expected = {term: label for term, (label, _, _) in DEATH_INTERVALS.items()}
    expected.update(PRENATAL_LOSS)
    expected.update(UNLOCATED_SEVERITY)

    mismatches = [
        f"{term}: expected {label!r}, ontology says {labels.get(term)!r}"
        for term, label in expected.items()
        if labels.get(term) != label
    ]
    if mismatches:
        raise ValueError("HPO term labels have drifted:\n  " + "\n  ".join(mismatches))


def death_intervals(hpoa: pd.DataFrame) -> pd.DataFrame:
    """One row per disease and annotated age-of-death term."""
    frame = hpoa[hpoa["hpo_id"].isin(DEATH_INTERVALS)].copy()
    frame["death_label"] = frame["hpo_id"].map(lambda term: DEATH_INTERVALS[term][0])
    frame["death_lower"] = frame["hpo_id"].map(lambda term: DEATH_INTERVALS[term][1])
    frame["death_upper"] = frame["hpo_id"].map(lambda term: DEATH_INTERVALS[term][2])
    frame["death_midpoint"] = (frame["death_lower"] + frame["death_upper"]) / 2.0
    return frame[
        ["database_id", "hpo_id", "death_label", "death_lower", "death_upper", "death_midpoint"]
    ].reset_index(drop=True)


def per_disease(hpoa: pd.DataFrame) -> pd.DataFrame:
    """One row per disease.

    A disease annotated with several mortality terms describes a range of
    outcomes. The earliest midpoint is the point estimate, matching the earliest
    interval rule used for onset, and the widest span across annotated terms
    sets the sensitivity band. Prenatal loss is flagged separately, since it
    removes reproduction entirely and needs no interval.
    """
    intervals = death_intervals(hpoa)
    prenatal = set(hpoa.loc[hpoa["hpo_id"].isin(PRENATAL_LOSS), "database_id"])
    unlocated = set(hpoa.loc[hpoa["hpo_id"].isin(UNLOCATED_SEVERITY), "database_id"])

    if intervals.empty:
        table = pd.DataFrame(
            columns=["database_id", "death_lower", "death_upper", "death_midpoint", "n_terms"]
        )
    else:
        grouped = intervals.groupby("database_id")
        table = pd.DataFrame(
            {
                "database_id": grouped.size().index,
                "death_lower": grouped["death_lower"].min().to_numpy(),
                "death_upper": grouped["death_upper"].max().to_numpy(),
                "death_midpoint": grouped["death_midpoint"].min().to_numpy(),
                "n_terms": grouped.size().to_numpy(),
            }
        )

    extra = sorted((prenatal | unlocated) - set(table["database_id"]))
    if extra:
        blanks = pd.DataFrame(
            {
                "database_id": extra,
                "death_lower": pd.Series([float("nan")] * len(extra), dtype="float64"),
                "death_upper": pd.Series([float("nan")] * len(extra), dtype="float64"),
                "death_midpoint": pd.Series([float("nan")] * len(extra), dtype="float64"),
                "n_terms": pd.Series([0] * len(extra), dtype="int64"),
            }
        )
        table = pd.concat([table, blanks], ignore_index=True)

    table["prenatal_loss"] = table["database_id"].isin(prenatal)
    table["unlocated_severity"] = table["database_id"].isin(unlocated)
    table["orpha_code"] = (
        table["database_id"].str.extract(r"^ORPHA:(\d+)$", expand=False).astype("Int64")
    )
    return table


def coverage(hpoa: pd.DataFrame) -> dict[str, int]:
    table = per_disease(hpoa)
    located = table["death_midpoint"].notna()
    return {
        "diseases_in_hpoa": int(hpoa["database_id"].nunique()),
        "with_age_of_death_interval": int(located.sum()),
        "with_prenatal_loss_only": int((~located & table["prenatal_loss"]).sum()),
        "with_unlocated_severity_only": int(
            (~located & ~table["prenatal_loss"] & table["unlocated_severity"]).sum()
        ),
        "orpha_keyed_with_interval": int(table.loc[located, "orpha_code"].notna().sum()),
    }


def load() -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / "hpo_annotations.parquet")
