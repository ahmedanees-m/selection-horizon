# selection-horizon

[![test](https://github.com/ahmedanees-m/selection-horizon/actions/workflows/test.yml/badge.svg)](https://github.com/ahmedanees-m/selection-horizon/actions/workflows/test.yml)
[![lint](https://github.com/ahmedanees-m/selection-horizon/actions/workflows/lint.yml/badge.svg)](https://github.com/ahmedanees-m/selection-horizon/actions/workflows/lint.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Analysis code for an onset-stratified comparison of gene-level evolutionary
scores in Mendelian and complex disease. The pipeline assembles disease-gene
sets with age-of-onset annotation, builds evolutionary and non-evolutionary
feature matrices, and fits mixed-effects models estimating how the
discriminative value of each score family varies with age of onset.

## Requirements

- Docker, or Python 3.11
- R 4.2 for the relative-success comparison in `R/`
- Approximately 60 GB of disk for source and derived data
- No GPU is required. The Evo 2 scoring module calls a hosted endpoint

## Installation

```bash
git clone https://github.com/ahmedanees-m/selection-horizon.git
cd selection-horizon
docker build -t selection-horizon .
```

Or without Docker:

```bash
pip install -e ".[dev]"
```

## Data

All sources are public. `config/sources.yaml` lists each with its version tag and
expected checksum, and `data/MANIFEST.json` records what was retrieved.

```bash
make fetch
```

Downloads are verified against the manifest and the pipeline stops on a checksum
mismatch. UMLS requires a licence and an API key supplied through `UMLS_API_KEY`.
FinnGen registry values are retrieved by `src/ingest/risteys.py` rather than
redistributed here.

## Running

```bash
make all
```

Individual stages:

```bash
make crosswalk      # ontology mapping
make onset          # onset tiers and concordance
make features       # evolutionary and baseline matrices
make models         # estimands, controls, mediation, robustness
```

Outputs are written under `data/derived/`. A complete run takes about two hours
on 32 cores excluding downloads and Evo 2 scoring.

## Repository layout

| Path | Contents |
|---|---|
| `config/` | Source versions, ontology mapping rules, model specifications |
| `src/ingest/` | One module per data source |
| `src/crosswalk/` | Disease ontology mapping |
| `src/onset/` | Onset tier construction and the registry measurement model |
| `src/features/` | Evolutionary and non-evolutionary feature matrices |
| `src/evo2/` | Sequence likelihood scoring against a hosted endpoint |
| `src/models/` | Estimators, mediation, robustness arms |
| `src/diagnostics/` | Crosstabs, power simulation, source verification |
| `R/` | Relative-success comparison |
| `docs/` | Methods, data dictionary, causal diagram, environment |

## Tests

```bash
make test
```

238 tests covering ingest schemas, ontology mapping, onset construction,
estimator behaviour on synthetic data with known parameters, and regression
tests on the analysis set definition.

## Citation

Ahmed A. Evolutionary constraint does not differentially decay with age of onset
in Mendelian disease genes. In preparation.

Analysis plan: https://doi.org/10.17605/OSF.IO/CBS8Q

## License

MIT for code. Derived data tables are CC-BY-4.0, subject to the upstream terms
recorded in `data/MANIFEST.json` and `LICENSE-DATA`.
