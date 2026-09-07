SHELL := /bin/bash

IMAGE      := selection-horizon:latest
R_IMAGE    := selection-horizon-r:4.2

REPO       := $(shell pwd)
DATA_ROOT  ?= $(REPO)/data
UID_GID    := $(shell id -u):$(shell id -g)

# Evo 2 scoring calls a hosted endpoint, so it needs an API key rather than a
# GPU. The key is read from a file outside the repository.
KEY_DIR    ?= $(HOME)/.config/selection-horizon
KEY_FILE   ?= /keys/nvidia_api_key

RUN := docker run --rm \
	-u $(UID_GID) \
	-v $(REPO):/work \
	-v $(DATA_ROOT):/work/data \
	-e SH_DATA_ROOT=/work/data \
	-e HOME=/tmp \
	-w /work $(IMAGE)

RUN_R := docker run --rm \
	-u $(UID_GID) \
	-v $(REPO):/work \
	-v $(DATA_ROOT):/work/data \
	-w /work $(R_IMAGE)

RUN_KEYED := docker run --rm \
	-u $(UID_GID) \
	-v $(REPO):/work \
	-v $(DATA_ROOT):/work/data \
	-v $(KEY_DIR):/keys:ro \
	-e NVIDIA_API_KEY_FILE=$(KEY_FILE) \
	-e SH_DATA_ROOT=/work/data \
	-e HOME=/tmp \
	-w /work $(IMAGE)

.PHONY: all image r-image fetch verify manifest crosswalk onset features \
        analysis-set power models estimands controls mediation robustness balancing \
        demography evo2 evo2-full evo2-validate evo2-deposit \
        test lint shell clean-interim

all: fetch analysis-set crosswalk onset features models

image:
	docker build -t $(IMAGE) .

r-image:
	docker build -t $(R_IMAGE) -f R/Dockerfile R

fetch:
	$(RUN) python -m src.ingest.run --all

manifest:
	$(RUN) python -m src.ingest.manifest --verify

# Checks every source against the manifest: URLs, DOIs through Crossref, Zenodo
# records, version tags read off the retrieved files, identifier shapes, value
# ranges and cross-source overlaps.
verify:
	$(RUN) python -m src.diagnostics.verify_sources

analysis-set:
	$(RUN) python -m src.diagnostics.analysis_set

power:
	$(RUN) python -m src.diagnostics.power

# T1a is Orphanet onset intervals, T2 the Finnish registry, T3 the English one.
onset:
	$(RUN) python -m src.onset.severity
	$(RUN) python -m src.onset.complex_onset
	$(RUN) python -m src.onset.kuan
	$(RUN) python -m src.onset.donertas

# Adjudication rewrites the FinnGen crosswalk in place, so it runs after the
# crosswalk and before anything that reads it.
crosswalk:
	$(RUN) python -m src.crosswalk.build
	$(RUN) python -m src.crosswalk.finngen
	$(RUN) python -m src.crosswalk.adjudicate

features:
	$(RUN) python -m src.features.gene_table
	$(RUN) python -m src.ingest.conservation_tracks
	$(RUN) python -m src.features.evolutionary
	$(RUN) python -m src.features.baseline
	$(RUN) python -m src.features.pleiotropy
	$(RUN) python -m src.features.gene_class

models: estimands controls mediation robustness balancing

estimands:
	$(RUN) python -m src.models.primary
	$(RUN) python -m src.models.log_r
	$(RUN) python -m src.models.discrimination

controls:
	$(RUN) python -m src.models.controls --draws 1000

mediation:
	$(RUN) python -m src.models.mediation
	$(RUN) python -m src.models.metric_mechanism

robustness:
	$(RUN) python -m src.models.robustness

balancing:
	$(RUN) python -m src.models.balancing

demography:
	$(RUN) python -m src.models.validate_demography

# Scoring runs on a subset first and the full set follows only if the subset
# reproduces the published benchmark.
evo2:
	$(RUN_KEYED) python -m src.evo2.score --limit 2000
	$(RUN_KEYED) python -m src.evo2.validation

evo2-full:
	$(RUN_KEYED) python -m src.evo2.score

evo2-validate:
	$(RUN) python -m src.evo2.validation

evo2-deposit:
	$(RUN) python -m src.evo2.deposit

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check src tests
	$(RUN) black --check src tests
	$(RUN) mypy src

shell:
	docker run --rm -it -u $(UID_GID) -v $(REPO):/work -v $(DATA_ROOT):/work/data \
		-e SH_DATA_ROOT=/work/data -e HOME=/tmp -w /work $(IMAGE) bash

clean-interim:
	rm -rf $(DATA_ROOT)/interim/*
