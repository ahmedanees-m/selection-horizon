"""Source coverage checks and the identifiability crosstab.

Runs the coverage checks, assembles the disease by onset crosstab for each
evidence source, applies the minimum sizes in config/analysis.yaml, and writes
the record the reported counts are read from.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from src import config
from src.crosswalk import build as crosswalk
from src.diagnostics import crosstab
from src.ingest import open_targets
from src.onset import clinical_course, inheritance, tiers

L2G_THRESHOLD = 0.5


def _orphanet(name: str) -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / "orphanet" / f"{name}.parquet")


def check_onset_coverage() -> dict:
    """Triage check 1. Orphanet onset coverage after joining to gene associations."""
    onset_long = _orphanet("orphanet_onset")
    genes = _orphanet("orphanet_genes")

    coverage = tiers.coverage(onset_long)
    t1a = tiers.build_t1a(onset_long)

    with_genes = set(genes["orpha_code"].dropna().astype(int))
    joined = t1a[t1a["orpha_code"].isin(with_genes)]

    threshold = config.analysis()["thresholds"]["analysis_set"]["min_diseases_with_onset_and_gene"]
    result = {
        **coverage,
        "orphacodes_with_gene_association": len(with_genes),
        "with_ordinal_onset_and_gene": int(joined["orpha_code"].nunique()),
        "threshold": threshold,
        "pass": bool(joined["orpha_code"].nunique() >= threshold),
    }
    result["by_interval"] = joined["onset_earliest"].value_counts().sort_index().to_dict()
    return result


def check_age_of_death() -> dict:
    """Triage check 2. Age of death coverage in the natural-history product.

    The execution plan expects an interval average age of death alongside age of
    onset. The free product carries onset intervals and inheritance only, and
    the Orphanet RDcode API exposes no natural-history endpoint. The named
    fallback applies: the per-case severity term is taken from the HPO clinical
    course sub-ontology instead, and the sensitivity band widens.
    """
    raw = config.raw_dir() / "orphanet" / "en_product9_ages.xml"
    text = raw.read_text(encoding="utf-8", errors="replace") if raw.exists() else ""
    present = "AgeOfDeath" in text

    result: dict = {
        "age_of_death_field_present": present,
        "threshold": config.analysis()["thresholds"]["analysis_set"]["min_age_of_death_coverage"],
        "fallback": "HPO Age of death branch HP:0011420",
    }

    try:
        hpoa = clinical_course.load()
        clinical_course.verify_labels()
        result["fallback_coverage"] = clinical_course.coverage(hpoa)
    except (FileNotFoundError, ValueError) as error:
        result["fallback_coverage"] = {"error": str(error)}

    onset_long = _orphanet("orphanet_onset")
    annotated = tiers.build_t1a(onset_long)["orpha_code"].nunique()
    located = result["fallback_coverage"].get("orpha_keyed_with_interval", 0)
    result["fraction_of_onset_annotated"] = round(located / annotated, 4) if annotated else 0.0
    result["pass"] = bool(result["fraction_of_onset_annotated"] >= result["threshold"])
    return result


def check_platform_integrity() -> dict:
    """Triage check 3. Mondo resolution across the diseases that carry evidence."""
    report = crosswalk.platform_vocabulary_report()

    sources = {
        "a_mendelian_via_platform": "evidence_orphanet",
        "b_gwas": "evidence_gwas_credible_sets",
    }
    resolution: dict[str, dict[str, float] | None] = {}
    for label, dataset in sources.items():
        try:
            evidence = open_targets.read(dataset, columns=["diseaseId"])
        except FileNotFoundError:
            resolution[label] = None
            continue
        ids = evidence["diseaseId"].dropna().unique()
        mondo = sum(1 for value in ids if str(value).startswith("MONDO_"))
        resolution[label] = {
            "n_diseases": int(len(ids)),
            "mondo_fraction": round(mondo / len(ids), 4) if len(ids) else 0.0,
        }

    threshold = config.analysis()["thresholds"]["analysis_set"]["min_mondo_resolution"]
    observed = [value["mondo_fraction"] for value in resolution.values() if isinstance(value, dict)]
    return {
        "release": config.sources()["open_targets"]["release"],
        "vocabulary_counts": report.set_index("prefix")["n_terms"].to_dict(),
        "evidence_resolution": resolution,
        "threshold": threshold,
        "pass": bool(observed) and min(observed) >= threshold,
    }


def check_evo2_hardware() -> dict:
    """Triage check 4. Recorded rather than computed."""
    return {
        "gpu": "NVIDIA RTX A4000",
        "compute_capability": 8.6,
        "memory_gb": 16,
        "required_compute_capability": 8.9,
        "plan_primary_model": "evo2_20b",
        "pass": False,
        "resolution": "7B in bf16 without the FP8 path, cross-checked against the 1B base "
        "checkpoint; the model size is recorded wherever the arm appears",
    }


def build_positives() -> pd.DataFrame:
    """Gene-disease positives per evidence source with an onset bin attached.

    The Mendelian arm is assembled in ORPHA space directly from Orphanet, which
    needs no crosswalk. The Platform arms are assembled in Mondo space and take
    onset through the ORPHA cross-reference where one exists. Diseases with no
    onset assignment are retained under an "unassigned" bin: the complex-disease
    onset axis is built from Risteys and Kuan, and until those land the
    gap has to stay visible.
    """
    onset_long = _orphanet("orphanet_onset")
    t1a = tiers.build_t1a(onset_long)[["orpha_code", "onset_earliest"]]

    modes = (
        onset_long[["orpha_code", "inheritance"]]
        .drop_duplicates("orpha_code")
        .assign(inheritance_mode=lambda frame: inheritance.classify(frame["inheritance"]))
    )

    genes = _orphanet("orphanet_genes")
    causal = genes[genes["association_type"].str.contains("Disease-causing", case=False, na=False)]
    mendelian = causal.merge(t1a, on="orpha_code", how="left").merge(
        modes[["orpha_code", "inheritance_mode"]], on="orpha_code", how="left"
    )
    mendelian = mendelian.assign(
        disease_id=mendelian["orpha_code"].astype("Int64").astype(str).radd("Orphanet_"),
        gene_id=mendelian["ensembl_gene_id"],
        evidence_source="a_mendelian",
        onset_bin=mendelian["onset_earliest"].str.lower(),
    )

    mapping = crosswalk.orpha_to_platform().merge(t1a, on="orpha_code", how="left")
    platform_onset = (
        mapping.dropna(subset=["onset_earliest"])
        .groupby("disease_id")["onset_earliest"]
        .first()
        .str.lower()
    )
    complex_onset_map = platform_complex_onset()

    columns = ["disease_id", "gene_id", "evidence_source", "onset_bin", "inheritance_mode"]
    frames = [mendelian[columns].assign(onset_decade=pd.NA)]

    platform_arms = {
        "b_gwas": ("evidence_gwas_credible_sets", L2G_THRESHOLD),
        "c_clinvar": ("evidence_eva", None),
        "d_gene_burden": ("evidence_gene_burden", None),
    }
    for source, (dataset, threshold) in platform_arms.items():
        columns = ["diseaseId", "targetId"]
        if threshold is not None:
            columns.append("resourceScore")
        try:
            evidence = open_targets.read(dataset, columns=columns)
        except FileNotFoundError:
            continue
        if threshold is not None:
            evidence = evidence[evidence["resourceScore"] >= threshold]
        frames.append(
            pd.DataFrame(
                {
                    "disease_id": evidence["diseaseId"],
                    "gene_id": evidence["targetId"],
                    "evidence_source": source,
                    "onset_bin": evidence["diseaseId"].map(platform_onset),
                    "onset_decade": evidence["diseaseId"].map(complex_onset_map),
                }
            )
        )

    positives = pd.concat(frames, ignore_index=True)
    positives = positives.dropna(subset=["disease_id", "gene_id"])
    positives["onset_bin_raw"] = positives["onset_bin"]
    positives["onset_bin"] = tiers.collapse(positives["onset_bin"])
    return apply_namespace_policy(positives)


def platform_complex_onset() -> pd.Series:
    """Platform disease identifier to its T2 decade bin, where one exists.

    Only the confident tier of the FinnGen crosswalk is used. Mappings queued for
    adjudication and mappings whose target is a measurement rather than a disease
    are left out, so an endpoint contributes an onset only where the identifier
    it resolved to is the intended one.
    """
    from src.crosswalk import finngen

    try:
        crosswalk_table = finngen.load()
    except FileNotFoundError:
        return pd.Series(dtype=object)

    confident = crosswalk_table[
        crosswalk_table["confident"] & crosswalk_table["usable_for_onset"]
    ].dropna(subset=["disease_id", "median_age_first_event"])
    if confident.empty:
        return pd.Series(dtype=object)

    binned = tiers.bin_continuous(confident["median_age_first_event"]).astype(object)
    return (
        pd.Series(binned.to_numpy(), index=confident["disease_id"].to_numpy())
        .groupby(level=0)
        .first()
    )


def apply_namespace_policy(positives: pd.DataFrame) -> pd.DataFrame:
    """Drop namespaces for which age of onset is undefined.

    The Ontology of Biological Attributes holds measurements: lipid levels,
    blood cell counts, lung function. A measurement has no age of onset, so
    those terms cannot enter an onset-stratified analysis. They are retained
    under a flag and used as a negative control, where no score should show an
    onset gradient because there is no onset to gradient against.
    """
    excluded = tuple(
        f"{prefix}_"
        for prefix in config.analysis()["namespace_policy"]["onset_analysis_exclude_prefixes"]
    )
    frame = positives.copy()
    frame["namespace"] = frame["disease_id"].astype(str).str.split("_").str[0]
    frame["onset_eligible"] = ~frame["disease_id"].astype(str).str.startswith(excluded)
    frame.loc[~frame["onset_eligible"], "onset_bin"] = None
    frame.loc[~frame["onset_eligible"], "onset_decade"] = None
    return frame


def namespace_breakdown(positives: pd.DataFrame) -> pd.DataFrame:
    """Diseases and pairs per namespace, per evidence source."""
    return (
        positives.groupby(["evidence_source", "namespace"])
        .agg(n_diseases=("disease_id", "nunique"), n_pairs=("gene_id", "size"))
        .reset_index()
        .sort_values(["evidence_source", "n_pairs"], ascending=[True, False])
    )


def adult_cell_composition(positives: pd.DataFrame) -> pd.DataFrame:
    """Disease-category breakdown of the adult Mendelian cell.

    Adult-onset Mendelian disease is not a random draw from the genome. It
    concentrates in hereditary cancer predisposition, adult neurodegeneration,
    and adult metabolic and cardiac disease. Cancer predisposition genes are DNA
    repair genes, constrained for reasons unrelated to when the cancer appears,
    which is exactly that account operating inside the cell that
    carries the entire late-onset signal. The breakdown is a main result, not a
    supplementary table.
    """
    adult = positives[
        (positives["evidence_source"] == "a_mendelian") & (positives["onset_bin"] == "adult")
    ]
    if adult.empty:
        return pd.DataFrame(columns=["category", "n_diseases", "n_pairs"])

    names = _orphanet("orphanet_genes")[["orpha_code", "disease_name"]].drop_duplicates()
    names["disease_id"] = "Orphanet_" + names["orpha_code"].astype("Int64").astype(str)
    joined = adult.merge(names[["disease_id", "disease_name"]], on="disease_id", how="left")
    joined["category"] = joined["disease_name"].map(classify_disease)

    return (
        joined.groupby("category")
        .agg(n_diseases=("disease_id", "nunique"), n_pairs=("gene_id", "size"))
        .reset_index()
        .sort_values("n_diseases", ascending=False)
    )


CATEGORY_PATTERNS = {
    "hereditary cancer predisposition": [
        "cancer",
        "carcinoma",
        "tumor",
        "tumour",
        "neoplas",
        "adenoma",
        "sarcoma",
        "leukemia",
        "leukaemia",
        "lymphoma",
        "melanoma",
        "polyposis",
        "blastoma",
        "predisposition to",
    ],
    "neurodegenerative": [
        "ataxia",
        "parkinson",
        "alzheimer",
        "dementia",
        "huntington",
        "amyotrophic",
        "spastic parapleg",
        "neuropathy",
        "leukodystroph",
        "prion",
        "spinocerebellar",
    ],
    "metabolic": [
        "deficiency",
        "storage disease",
        "aciduria",
        "acidemia",
        "porphyria",
        "amyloidosis",
        "hyperlipid",
        "diabetes",
        "mitochondrial",
    ],
    "cardiac or neuromuscular": [
        "cardiomyopathy",
        "arrhythm",
        "long qt",
        "brugada",
        "myopathy",
        "muscular dystroph",
        "myasthen",
        "aortic",
    ],
}


def autosomal_restriction(positives: pd.DataFrame) -> pd.DataFrame:
    """Positive rows lost to the autosomal restriction, by onset level.

    The gnomAD v4.1 constraint release covers the autosomes, and expected
    loss-of-function count is one of the matching covariates, so a gene on a sex
    chromosome carries no value for it and leaves the design. Counted over the
    positives whose gene carries both primary scores, which are the rows that
    would otherwise enter.
    """
    from src.features import evolutionary, gene_table

    scored = evolutionary.load().dropna(
        subset=[evolutionary.CONSERVATION_PRIMARY, evolutionary.CONSTRAINT_PRIMARY]
    )
    mendelian = positives[
        (positives["evidence_source"] == "a_mendelian")
        & positives["onset_bin"].notna()
        & positives["gene_id"].isin(set(scored["ensembl_gene_id"]))
    ]
    genes = gene_table.load()[["ensembl_gene_id", "chrom"]].drop_duplicates("ensembl_gene_id")
    joined = mendelian.merge(
        genes.rename(columns={"ensembl_gene_id": "gene_id"}), on="gene_id", how="left"
    )
    chromosome = joined["chrom"].astype(str).str.upper().str.replace("CHR", "", regex=False)
    joined["sex_chromosome"] = chromosome.isin(["X", "Y"])

    from src.onset import tiers

    order = [level for level in tiers.collapsed_order() if level in set(joined["onset_bin"])]
    counts = (
        pd.crosstab(joined["onset_bin"], joined["sex_chromosome"])
        .reindex(order)
        .reindex(columns=[False, True], fill_value=0)
        .fillna(0)
        .astype(int)
    )
    counts.columns = ["retained", "excluded"]
    counts["excluded_share"] = counts["excluded"] / (counts["retained"] + counts["excluded"])
    return counts.reset_index().rename(columns={"onset_bin": "onset"})


def classify_disease(name: str | None) -> str:
    """Keyword classification of the adult cell.

    Coarse by design and reported as such. Its purpose is to show whether the
    adult cell is dominated by one disease class, which a keyword pass answers
    adequately. The Orphanet linearisation replaces it once product3 is
    ingested.
    """
    if not isinstance(name, str):
        return "unclassified"
    lowered = name.lower()
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            return category
    return "other"


def main() -> None:
    config.ensure_dirs()
    out_dir = config.derived_dir() / "analysis_set"
    out_dir.mkdir(parents=True, exist_ok=True)

    triage = {
        "onset_coverage": check_onset_coverage(),
        "age_of_death": check_age_of_death(),
        "platform_integrity": check_platform_integrity(),
        "evo2_hardware": check_evo2_hardware(),
    }

    positives = build_positives()
    positives.to_parquet(out_dir / "positives_collapsed.parquet", index=False)

    cells = crosstab.build(positives)
    cells.to_parquet(out_dir / "crosstab.parquet", index=False)
    result = crosstab.assess(cells)

    # The complex arm runs on its own axis. Mendelian onset is an ordinal
    # interval and complex onset is a median age in years, so the two are binned
    # differently and their crosstabs are reported separately. Gamma is estimated
    # inside each arm and never pooled across them.
    complex_cells = crosstab.build(
        positives.drop(columns=["onset_bin"]).rename(columns={"onset_decade": "onset_bin"}),
        order=crosstab.decade_order(),
    )
    complex_cells.to_parquet(out_dir / "crosstab_complex.parquet", index=False)

    namespaces = namespace_breakdown(positives)
    namespaces.to_csv(out_dir / "namespace_breakdown.csv", index=False)

    composition = adult_cell_composition(positives)
    composition.to_csv(out_dir / "adult_cell_composition.csv", index=False)
    restriction = autosomal_restriction(positives)
    restriction.to_csv(out_dir / "autosomal_restriction.csv", index=False)

    mendelian_diseases = (
        positives[positives["evidence_source"] == "a_mendelian"]
        .dropna(subset=["onset_bin"])
        .drop_duplicates("disease_id")
    )
    inheritance_table = inheritance.by_onset(mendelian_diseases, tiers.collapsed_order())
    inheritance_table.to_csv(out_dir / "inheritance_by_onset.csv", index=False)

    mendelian_counts = cells[
        (cells["evidence_source"] == "a_mendelian") & (cells["onset_bin"] != "unassigned")
    ].set_index("onset_bin")["n_diseases"]
    collapse_warnings = tiers.check_collapse_counts(mendelian_counts)

    record = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "open_targets_release": config.sources()["open_targets"]["release"],
        "onset_bins": tiers.collapsed_order(),
        "triage": triage,
        "crosstab": cells.to_dict(orient="records"),
        "crosstab_complex": complex_cells.to_dict(orient="records"),
        "per_source_bins": result.per_source_bins.to_dict(orient="records"),
        "namespace_breakdown": namespaces.to_dict(orient="records"),
        "adult_cell_composition": composition.to_dict(orient="records"),
        "inheritance_by_onset": inheritance_table.to_dict(orient="records"),
        "collapse_warnings": collapse_warnings,
        "pivotal_cell_diseases": result.pivotal_cell_diseases,
        "verdict": result.verdict,
        "reasons": result.reasons,
    }
    with (out_dir / "analysis_set_record.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=str)
        handle.write("\n")

    print("Dependency triage")
    for name, values in triage.items():
        print(f"  {name}: {'pass' if values.get('pass') else 'fail'}")
    print()
    print("Diseases by collapsed onset bin and evidence source")
    print(crosstab.to_markdown(cells))
    print()
    print("Gene-disease pairs")
    print(crosstab.to_markdown(cells, value="n_pairs"))
    print()
    print("Complex arm, diseases by decade of median age at first event")
    print(crosstab.to_markdown(complex_cells, order=crosstab.decade_order()))
    print()
    print("GWAS evidence by namespace")
    gwas = namespaces[namespaces["evidence_source"] == "b_gwas"].head(8)
    print(gwas.to_string(index=False))
    print()
    print("Adult Mendelian cell composition")
    print(composition.to_string(index=False))
    print()
    print("Inheritance mode by onset bin, Mendelian arm")
    print(inheritance_table.to_string(index=False))
    if collapse_warnings:
        print()
        print("Collapse rule warnings")
        for warning in collapse_warnings:
            print(f"  {warning}")
    print()
    print(f"Analysis set verdict: {result.verdict}")
    for reason in result.reasons:
        print(f"  {reason}")


if __name__ == "__main__":
    main()
