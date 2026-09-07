"""kappa_d, the per-case reduction in residual reproduction.

The execution plan takes this from an Orphanet age-of-death interval that
Orphanet does not publish, and the HPO age-of-death branch reaches under one
percent of the onset-annotated diseases. Carrying it as a constant instead would
make

    s_d = kappa * sum_a h_d(a) W(a) / W(0)

a monotone transform of onset, so s_d would hold no information beyond onset and
H4 would be untestable by construction rather than by data.

It is therefore built as an index over the three things that actually remove
residual reproduction.

    Lethality           dying before reproduction ends removes what remains
    Fertility           an allele causing infertility has a selection
                        coefficient near one whatever the lifespan
    Care burden         a disease touching many organ systems, or involving
                        intellectual disability, reduces realised reproduction
                        without being lethal or directly gonadal

For a selection model the fertility component matters more than age of death. An
allele causing death at seventy carries a selection coefficient near zero however
certain that death is.

Each component is bounded in [0, 1] and they combine as a bounded sum, because
they remove overlapping rather than additive shares of the same remaining
reproduction. The 439 diseases carrying an observed age of death are held out as
a calibration set rather than used to fit.

Every HPO concept is named, resolved against the ontology at load, and its
resolved identifier reported. See src/onset/hpo_ontology.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.onset import clinical_course, hpo_ontology


def _spec() -> dict:
    return config.analysis()["severity_index"]


def frequency_weights(frequencies: pd.Series) -> pd.Series:
    """Orphanet frequency labels mapped to the midpoint of the interval named."""
    weights = _spec()["frequency_weights"]
    mapped = frequencies.map(weights)
    unknown = set(frequencies.dropna().unique()) - set(weights)
    if unknown:
        raise ValueError(f"Orphanet frequency labels absent from config: {sorted(unknown)}")
    # An unannotated frequency is treated as the modal category rather than as
    # absent, since Orphanet leaves it blank when it is not assessed.
    return mapped.fillna(weights["Frequent (79-30%)"])


def fertility_component(phenotypes: pd.DataFrame) -> pd.DataFrame:
    """Expected reduction in reproduction from the fertility annotations.

    For each annotated fertility term the reduction is the term's severity
    weight times the frequency with which the disease shows it. A disease is
    scored at its worst annotated term rather than by summing them, because the
    terms describe overlapping presentations of the same loss.
    """
    ontology = hpo_ontology.load()
    terms = _spec()["fertility_terms"]

    weight_of: dict[str, float] = {}
    resolved = []
    for label, weight in terms.items():
        term, subtree = ontology.subtree_by_name(label)
        resolved.append(
            {
                "concept": label,
                "hpo_id": term,
                "weight": weight,
                "n_terms_in_subtree": len(subtree),
            }
        )
        for member in subtree:
            weight_of[member] = max(weight_of.get(member, 0.0), weight)

    frame = phenotypes[phenotypes["hpo_id"].isin(weight_of)].copy()
    if frame.empty:
        out = pd.DataFrame(columns=["orpha_code", "fertility_loss", "n_fertility_terms"])
    else:
        frame["severity"] = frame["hpo_id"].map(weight_of)
        frame["expected"] = frame["severity"] * frequency_weights(frame["frequency"])
        grouped = frame.groupby("orpha_code")
        out = pd.DataFrame(
            {
                "orpha_code": grouped.size().index,
                "fertility_loss": grouped["expected"].max().to_numpy(),
                "n_fertility_terms": grouped.size().to_numpy(),
            }
        )
    out.attrs["resolved_terms"] = resolved
    return out


def rarefied_richness(counts: np.ndarray, sample: int) -> float:
    """Expected distinct categories when drawing `sample` annotations without replacement.

    A raw count of organ systems is not a measure of how many systems a disease
    affects; it is largely a measure of how thoroughly the disease has been
    curated. Measured on this release, the raw count correlates 0.79 with the
    number of annotations a disease carries.

    Rarefaction removes that. Every disease is compared at the same annotation
    depth, so a well-curated disease no longer looks broader than a sparsely
    curated one for that reason alone. This is the standard estimator:

        E[S_m] = sum_s ( 1 - C(n - n_s, m) / C(n, m) )

    computed in log space because the binomial coefficients overflow.
    """
    total = int(counts.sum())
    if total == 0:
        return float("nan")
    if total <= sample:
        return float((counts > 0).sum())

    from scipy.special import gammaln

    def log_choose(n: int, k: int) -> float:
        if k > n or k < 0:
            return -np.inf
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    denominator = log_choose(total, sample)
    expected = 0.0
    for count in counts:
        if count <= 0:
            continue
        numerator = log_choose(total - int(count), sample)
        expected += 1.0 - np.exp(numerator - denominator)
    return float(expected)


def care_burden_component(phenotypes: pd.DataFrame) -> pd.DataFrame:
    """Breadth of organ-system involvement, and intellectual disability.

    Breadth is rarefied to a fixed annotation depth so that it reflects how
    broadly a disease presents rather than how thoroughly it has been curated.
    It is then scaled by the number of systems the ontology defines, so the
    component is bounded and comparable across releases.
    """
    ontology = hpo_ontology.load()
    spec = _spec()
    depth = spec["rarefaction_depth"]

    root = ontology.resolve(spec["organ_system_root"])
    systems = ontology.children.get(root, ())
    system_index = {system: position for position, system in enumerate(systems)}
    system_of: dict[str, set[int]] = {}
    for system, position in system_index.items():
        for member in ontology.descendants(system):
            system_of.setdefault(member, set()).add(position)

    disability_root, disability_terms = ontology.subtree_by_name(
        spec["intellectual_disability_term"]
    )

    frame = phenotypes.dropna(subset=["hpo_id"]).copy()
    frame["systems"] = frame["hpo_id"].map(lambda term: system_of.get(term, set()))

    rows = []
    for code, group in frame.groupby("orpha_code"):
        counts = np.zeros(len(systems), dtype=float)
        for entry in group["systems"]:
            for position in entry:
                counts[position] += 1.0

        observed = int((counts > 0).sum())
        rarefied = rarefied_richness(counts, depth)
        breadth = rarefied / len(systems) if systems and np.isfinite(rarefied) else np.nan
        disability = bool(set(group["hpo_id"]) & disability_terms)
        rows.append(
            {
                "orpha_code": code,
                "n_annotations": int(len(group)),
                "n_organ_systems": observed,
                "rarefied_organ_systems": rarefied,
                "organ_system_breadth": breadth,
                "intellectual_disability": disability,
                "care_burden": (
                    spec["care_burden_weight"] * (breadth if np.isfinite(breadth) else 0.0)
                    + spec["intellectual_disability_weight"] * disability
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["n_systems"] = len(systems)
    out.attrs["rarefaction_depth"] = depth
    out.attrs["intellectual_disability_id"] = disability_root
    return out


def lethality_component(hpoa: pd.DataFrame, onset_midpoint: pd.Series) -> pd.DataFrame:
    """Reduction implied by an observed age at death, where one is annotated.

    Available for a small minority of diseases. It is the calibration set for
    the index rather than a component that can carry it.
    """
    table = clinical_course.per_disease(hpoa)
    table = table[table["orpha_code"].notna()].copy()
    table["orpha_code"] = table["orpha_code"].astype(int)

    onset = onset_midpoint.reindex(table["orpha_code"]).to_numpy()
    death = table["death_midpoint"].to_numpy(dtype=float)

    with np.errstate(invalid="ignore"):
        span = np.where(np.isfinite(onset) & np.isfinite(death), death - onset, np.nan)
    table["observed_years_after_onset"] = span
    table["prenatal_loss"] = table["prenatal_loss"].fillna(False)
    return table[
        [
            "orpha_code",
            "death_lower",
            "death_upper",
            "death_midpoint",
            "observed_years_after_onset",
            "prenatal_loss",
        ]
    ].reset_index(drop=True)


def build() -> pd.DataFrame:
    """kappa_d per ORPHAcode, with its components and the calibration column."""
    interim = config.interim_dir()
    phenotypes = pd.read_parquet(interim / "orphanet" / "orphanet_phenotypes.parquet")
    onset = pd.read_parquet(interim / "orphanet" / "orphanet_onset.parquet")

    from src.onset import tiers

    t1a = tiers.build_t1a(onset).set_index("orpha_code")["onset_midpoint"]

    fertility = fertility_component(phenotypes)
    care = care_burden_component(phenotypes)
    lethality = lethality_component(clinical_course.load(), t1a)

    table = (
        care.merge(fertility, on="orpha_code", how="outer")
        .merge(lethality, on="orpha_code", how="left")
        .fillna({"fertility_loss": 0.0, "n_fertility_terms": 0, "care_burden": 0.0})
    )

    # Bounded sum: the components remove overlapping shares of the same
    # remaining reproduction, so they cannot simply add.
    table["kappa_d"] = 1.0 - (1.0 - table["fertility_loss"]) * (1.0 - table["care_burden"])
    table.loc[table["prenatal_loss"].fillna(False), "kappa_d"] = 1.0

    table.attrs["fertility_terms"] = fertility.attrs.get("resolved_terms", [])
    table.attrs["n_organ_systems"] = care.attrs.get("n_systems", 0)
    table.attrs["rarefaction_depth"] = care.attrs.get("rarefaction_depth", 0)
    return table.sort_values("orpha_code").reset_index(drop=True)


def run() -> pd.DataFrame:
    table = build()
    target = config.interim_dir() / "severity_index.parquet"
    table.to_parquet(target, index=False)

    with_death = int(table["death_midpoint"].notna().sum())
    print(f"severity_index: {len(table):,} diseases")
    print(f"   organ systems in the ontology: {table.attrs['n_organ_systems']}")
    print(f"   breadth rarefied to {table.attrs['rarefaction_depth']} annotations")
    print(f"   with a fertility annotation: {int((table['n_fertility_terms'] > 0).sum()):,}")
    print(f"   with an observed age of death, held out for calibration: {with_death:,}")
    print()
    print("   resolved fertility concepts")
    for row in table.attrs["fertility_terms"]:
        print(
            f"      {row['concept']:<32} {row['hpo_id']}  "
            f"weight {row['weight']}  subtree {row['n_terms_in_subtree']}"
        )
    print()
    print(table[["fertility_loss", "care_burden", "kappa_d"]].describe().to_string())
    return table


def load() -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / "severity_index.parquet")


if __name__ == "__main__":
    config.ensure_dirs()
    run()
