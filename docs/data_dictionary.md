# Data dictionary

Every column in every derived table: meaning, units, source and missingness.
Written as each table is built.

## interim/orphanet/orphanet_onset.parquet

One row per ORPHAcode and onset interval. A disease with several intervals
contributes several rows.

| Column | Type | Meaning |
|---|---|---|
| orpha_code | int | Orphanet disorder identifier |
| disease_name | str | Orphanet preferred name, English |
| disorder_type | str | Disease, Malformation syndrome, Clinical subtype and so on |
| disorder_group | str | Disorder, Group of disorders, Subtype of disorder |
| onset_interval | str | Average age of onset interval. Null where Orphanet records no data |
| inheritance | str | Pipe-separated inheritance modes |

## interim/orphanet/orphanet_genes.parquet

One row per ORPHAcode and gene association.

| Column | Type | Meaning |
|---|---|---|
| orpha_code | int | Orphanet disorder identifier |
| gene_symbol | str | HGNC-approved symbol |
| ensembl_gene_id | str | Ensembl gene identifier, the join key for the feature matrices |
| hgnc_id | str | HGNC numeric identifier |
| omim_gene | str | OMIM gene identifier |
| uniprot | str | UniProtKB accession |
| association_type | str | Disease-causing germline, somatic, susceptibility factor and so on |
| association_status | str | Assessed or Not yet assessed |

## interim/orphanet/orphanet_xrefs.parquet

One row per ORPHAcode and external disease reference.

| Column | Type | Meaning |
|---|---|---|
| orpha_code | int | Orphanet disorder identifier |
| source | str | OMIM, ICD-10, ICD-11, MeSH, MedDRA, UMLS, GARD |
| reference | str | Identifier in the target vocabulary |
| mapping_relation | str | Exact, NTBT, BTNT or ND, as Orphanet records it |
| mapping_icd_relation | str | ICD-specific relation where applicable |
| mapping_validation | str | Validation status |

## interim/crosswalk/orpha_to_platform.parquet

| Column | Type | Meaning |
|---|---|---|
| orpha_code | int | Orphanet disorder identifier |
| disease_id | str | Open Targets Platform disease identifier, mixed namespace |
| disease_name | str | Platform preferred label |
| therapeuticAreas | list | Platform therapeutic area identifiers |

## derived/analysis_set/positives.parquet

Gene-disease positives with an onset bin, one row per pair per evidence source.

| Column | Type | Meaning |
|---|---|---|
| disease_id | str | Orphanet_ prefixed code for the Mendelian arm, Platform identifier otherwise |
| gene_id | str | Ensembl gene identifier |
| evidence_source | str | a_mendelian, b_gwas, c_clinvar or d_gene_burden |
| onset_bin | str | Orphanet interval, lower case, or null where no onset source covers the disease |

## derived/analysis_set/crosstab.parquet

| Column | Type | Meaning |
|---|---|---|
| evidence_source | str | As above |
| onset_bin | str | As above, with unassigned in place of null |
| n_diseases | int | Distinct diseases in the cell |
| n_pairs | int | Distinct gene-disease pairs in the cell |
| median_genes_per_disease | float | Median across the diseases in the cell |
