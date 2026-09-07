# Evo 2 scoring

This directory is isolated from the analysis environment. It has its own image,
its own CUDA stack, and no dependency on anything under `src/models`.

## Method

For each protein-coding gene on the GENCODE canonical transcript:

1. Extract an 8 kb genomic window centred on the CDS start.
2. Compute the wild-type sequence log-likelihood.
3. Insert a premature stop codon at a fixed early-CDS position following the
   published protocol, and record the exact rule in the output header.
4. Compute the mutant log-likelihood.
5. Score the length-normalised difference in log-likelihood.
6. Repeat at three insertion positions and average, to reduce sensitivity to
   the choice of position.

Roughly 19,000 genes at two conditions and three positions is on the order of
114,000 forward passes at 8 kb. The job checkpoints every 500 genes and is
resumable.

## Hardware

The plan makes `evo2_20b` primary on a GPU of compute capability 8.9 or above.
The available card is an RTX A4000: compute capability 8.6, 16 GB. The FP8 path
that Vortex uses is not available at 8.6, and a 20B checkpoint does not fit in
16 GB.

This arm therefore runs `evo2_7b` in bfloat16 as primary and cross-checks
against `evo2_1b_base` on a 2,000-gene random subset, reporting the rank
correlation between them. The model size against the published 40B
result is disclosed wherever the arm appears.

## Benchmark comparison

Before any score is used, the DepMap human essentiality AUROC is reproduced in
the same logistic setup as the published evaluation, against GC content and
transcript length baselines. Landing within 0.03 of the published 0.66 releases
the arm. Materially below that, the stop-codon rule and the context window are
debugged once. Still wrong after one retry, the arm is dropped and H1 proceeds
on PhyloP-241way, PhyloP-100way and phastCons.

The score table is released either way. It has standalone value and costs
nothing extra to publish.
