"""Orphanet ingest.

Three free products are used.

product9_ages   Average age of onset intervals per ORPHAcode. Primary onset
                source for the Mendelian arm.
product6        Gene to disease associations with association type, validation
                status, and external gene identifiers.
product1        External disease cross-references to OMIM, ICD-10, MeSH,
                MedDRA, UMLS and GARD, each with a mapping relation.
product4        HPO phenotype annotations with a frequency label per
                association, which the severity index weights by.

The natural-history product carries age of onset but no age of death, so the
fitness-cost model takes its per-case severity term from the HPO clinical-course
sub-ontology instead.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src import config
from src.ingest import fetch

SOURCE = "orphanet"
NO_DATA = "No data available"


def _text(node: ET.Element | None, path: str) -> str | None:
    if node is None:
        return None
    found = node.find(path)
    return found.text if found is not None else None


def _download(name: str) -> Path:
    spec = config.sources()["orphanet"]
    entry = spec["files"][name]
    dest = config.raw_dir() / SOURCE / Path(entry["url"]).name
    return fetch.download(
        entry["url"],
        dest,
        source=SOURCE,
        version=spec["version"],
        licence=spec["licence"],
        redistribute=spec["redistribute"],
    )


def parse_natural_history(path: Path) -> pd.DataFrame:
    """One row per ORPHAcode and onset interval."""
    root = ET.parse(path).getroot()
    rows = []
    for disorder in root.iter("Disorder"):
        code = _text(disorder, "OrphaCode")
        name = _text(disorder, "Name")
        dtype = _text(disorder, "DisorderType/Name")
        dgroup = _text(disorder, "DisorderGroup/Name")
        inheritance = [
            item.text
            for item in disorder.findall("TypeOfInheritanceList/TypeOfInheritance/Name")
            if item.text
        ]
        onsets: list[str | None] = [
            item.text
            for item in disorder.findall("AverageAgeOfOnsetList/AverageAgeOfOnset/Name")
            if item.text
        ]
        if not onsets:
            onsets = [None]
        for onset in onsets:
            rows.append(
                {
                    "orpha_code": int(code) if code else None,
                    "disease_name": name,
                    "disorder_type": dtype,
                    "disorder_group": dgroup,
                    "onset_interval": None if onset == NO_DATA else onset,
                    "inheritance": "|".join(inheritance) if inheritance else None,
                }
            )
    return pd.DataFrame(rows)


def parse_genes(path: Path) -> pd.DataFrame:
    """One row per ORPHAcode and gene association."""
    root = ET.parse(path).getroot()
    rows = []
    for disorder in root.iter("Disorder"):
        code = _text(disorder, "OrphaCode")
        name = _text(disorder, "Name")
        for assoc in disorder.findall("DisorderGeneAssociationList/DisorderGeneAssociation"):
            gene = assoc.find("Gene")
            if gene is None:
                continue
            refs = {
                ref.findtext("Source"): ref.findtext("Reference")
                for ref in gene.findall("ExternalReferenceList/ExternalReference")
            }
            rows.append(
                {
                    "orpha_code": int(code) if code else None,
                    "disease_name": name,
                    "gene_symbol": _text(gene, "Symbol"),
                    "gene_name": _text(gene, "Name"),
                    "gene_type": _text(gene, "GeneType/Name"),
                    "ensembl_gene_id": refs.get("Ensembl"),
                    "hgnc_id": refs.get("HGNC"),
                    "omim_gene": refs.get("OMIM"),
                    "uniprot": refs.get("SwissProt"),
                    "association_type": _text(assoc, "DisorderGeneAssociationType/Name"),
                    "association_status": _text(assoc, "DisorderGeneAssociationStatus/Name"),
                }
            )
    return pd.DataFrame(rows)


def parse_phenotypes(path: Path) -> pd.DataFrame:
    """One row per ORPHAcode and annotated HPO term, with its frequency label."""
    root = ET.parse(path).getroot()
    rows = []
    for disorder in root.iter("Disorder"):
        code = _text(disorder, "OrphaCode")
        for assoc in disorder.findall("HPODisorderAssociationList/HPODisorderAssociation"):
            rows.append(
                {
                    "orpha_code": int(code) if code else None,
                    "hpo_id": _text(assoc, "HPO/HPOId"),
                    "hpo_term": _text(assoc, "HPO/HPOTerm"),
                    "frequency": _text(assoc, "HPOFrequency/Name"),
                    "diagnostic_criteria": _text(assoc, "DiagnosticCriteria/Name"),
                }
            )
    return pd.DataFrame(rows)


def parse_cross_references(path: Path) -> pd.DataFrame:
    """One row per ORPHAcode and external disease reference."""
    root = ET.parse(path).getroot()
    rows = []
    for disorder in root.iter("Disorder"):
        code = _text(disorder, "OrphaCode")
        for ref in disorder.findall("ExternalReferenceList/ExternalReference"):
            rows.append(
                {
                    "orpha_code": int(code) if code else None,
                    "source": ref.findtext("Source"),
                    "reference": ref.findtext("Reference"),
                    "mapping_relation": ref.findtext("DisorderMappingRelation/Name"),
                    "mapping_icd_relation": ref.findtext("DisorderMappingICDRelation/Name"),
                    "mapping_validation": ref.findtext("DisorderMappingValidationStatus/Name"),
                }
            )
    return pd.DataFrame(rows)


def run() -> dict[str, Path]:
    out_dir = config.interim_dir() / SOURCE
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for key, parser, stem in (
        ("natural_history", parse_natural_history, "orphanet_onset"),
        ("gene_associations", parse_genes, "orphanet_genes"),
        ("cross_references", parse_cross_references, "orphanet_xrefs"),
        ("phenotypes", parse_phenotypes, "orphanet_phenotypes"),
    ):
        frame = parser(_download(key))
        target = out_dir / f"{stem}.parquet"
        frame.to_parquet(target, index=False)
        written[stem] = target
        print(f"{stem}: {len(frame):,} rows, {frame['orpha_code'].nunique():,} ORPHAcodes")
    return written


if __name__ == "__main__":
    config.ensure_dirs()
    run()
