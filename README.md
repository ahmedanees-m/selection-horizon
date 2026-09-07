# selection-horizon

[![test](https://github.com/ahmedanees-m/selection-horizon/actions/workflows/test.yml/badge.svg)](https://github.com/ahmedanees-m/selection-horizon/actions/workflows/test.yml)
[![lint](https://github.com/ahmedanees-m/selection-horizon/actions/workflows/lint.yml/badge.svg)](https://github.com/ahmedanees-m/selection-horizon/actions/workflows/lint.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Analysis code for an onset-stratified comparison of gene-level evolutionary
scores in Mendelian and complex disease.

## The question

Gene-level constraint and conservation scores are among the most widely used
filters in human disease genetics, and they rest on the premise that selection
has recorded which genes matter. That premise has a stated limit: selection acts
weakly on phenotypes appearing after reproduction, so these scores should carry
less information about late-onset disease, and a gene may look tolerant of
inactivation despite causing severe late-onset disease. The limit is established
at the level of allele frequencies. Whether it reaches the gene-level scores that
pipelines actually filter on is a separate question, and it is the one that bears
on practice.

Measuring it directly is hard, because any single score's apparent decay along an
onset axis is confounded by label quality, which is itself worse at the late end.
The estimand here is therefore a contrast rather than a level: the difference
between the onset slope of a cross-species conservation score and that of a
human-population constraint score, fitted jointly on the same genes and the same
diseases, so that a bias common to both arms partly cancels.

## The design

An age-of-onset axis is built for Mendelian disease from the Orphanet natural
history product, and for complex disease from registry age at first recorded
event. Disease genes are matched to controls on coding sequence length, expected
loss-of-function count and base composition. The model is a logistic regression
of positive status on the two standardised scores, onset, each score by onset and
each score by mode of inheritance, with a two-way cluster-robust covariance over
disease and over gene, because genes recur across diseases and diseases cluster
their genes. The contrast of the two score-by-onset coefficients is the estimand.

Around that sit seventeen robustness arms swapping the score, the control ratio,
the onset encoding, the row set and the label-quality restriction; a permutation
null with onset shuffled within therapeutic area; negative and positive controls;
and a power calculation read from the realised standard error of each fitted
contrast rather than from the design-stage simulation.

## What it reports

The two families do not decay at detectably different rates, in a design that
resolves 0.049 AUROC units. The gene-class explanation, that early-onset disease
genes are developmental, pleiotropic and haploinsufficient by construction, does
not account for the contrast. The answer depends on which constraint metric is
used, which is a finding about the metrics rather than about onset. And against
controls matched on coding length and expected loss-of-function count, no
constraint metric identifies recessive disease genes at all.

The analysis was registered before the data were fitted, and the deviations from
that plan are listed in the manuscript's additional file.

## Requirements

- Docker, or Python 3.11
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
mismatch. FinnGen registry values are retrieved by `src/ingest/risteys.py` rather
than redistributed here.

A companion data record holds the inputs the analysis reads together with the
per-gene aggregates standing in for sources that may not be redistributed, so
that `make models` reproduces every figure in the manuscript without fetching
anything. Sources that can be retrieved again at the version used are pointed at
rather than re-hosted, each with its URL, release and checksum.

Data record: https://doi.org/10.5281/zenodo.23125599

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
| `config/` | Source versions and analysis settings |
| `src/ingest/` | One module per data source |
| `src/crosswalk/` | Disease ontology mapping |
| `src/onset/` | Onset tier construction and the registry measurement model |
| `src/features/` | Evolutionary and non-evolutionary feature matrices |
| `src/evo2/` | Sequence likelihood scoring against a hosted endpoint |
| `src/models/` | Estimators, mediation, robustness arms |
| `src/diagnostics/` | Crosstabs, power simulation, source verification |
| `docs/` | Methods, data dictionary, causal diagram, environment |

## Tests

```bash
make test
```

258 tests covering ingest schemas, ontology mapping, onset construction,
estimator behaviour on synthetic data with known parameters, the matched control
draw, the dinucleotide-preserving scramble, and the analysis set definition.

## Citation

Mahaboob Ali AA, Nelson EJR. Gene-level evolutionary conservation and human
constraint decline at comparable rates across the age of onset of Mendelian
disease.

Analysis plan: https://doi.org/10.17605/OSF.IO/CBS8Q

## License

MIT for code. Derived data tables are CC-BY-4.0, subject to the upstream terms
recorded in `data/MANIFEST.json` and `LICENSE-DATA`.
