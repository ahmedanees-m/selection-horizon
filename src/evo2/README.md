# Evo 2 scoring

Per-gene scores from the Evo 2 40B model, taken as the change in sequence
likelihood when a premature stop codon is inserted early in the coding region.
The model is called through a hosted endpoint that returns vocabulary logits at
every position of a supplied sequence, so no GPU is needed.

## Rule

For each protein-coding gene on its GENCODE canonical transcript:

1. Extract an 8 kb genomic window centred on the coding start, reverse
   complemented for genes on the minus strand.
2. Compute the log likelihood of the wild-type window.
3. Replace the codon at nucleotide offsets 15, 30 and 45 from the coding start
   with TGA, one position at a time, and compute the log likelihood of each
   mutant window.
4. Score the mean difference across the three positions, divided by the window
   length.

A gene is scored only if its first coding exon holds the furthest insertion.
The rule was specified for this analysis. The published benchmark uses a
different perturbation, described under Evaluation.

## Evaluation

`validation.py` measures how well the score separates DepMap common-essential
genes, alongside coding length and GC content. The published Evo 2 evaluation
reports an AUROC of 0.66 on DepMap human gene essentiality, obtained by scoring
each gene's coding sequence against a scramble of it that preserves dinucleotide
composition. Evo 2 enters the conservation family only if the AUROC measured here
lies within 0.03 of that figure.

`benchmark.py` applies the published perturbation on the same genes and labels,
so the two rules can be compared without the perturbation differing between
them.

## Running

`score.py` checkpoints one gene at a time and can be resumed;
`scripts/evo2_supervisor.sh` keeps a long run going across network failures.
`deposit.py` writes the per-gene table with a header describing the rule.
