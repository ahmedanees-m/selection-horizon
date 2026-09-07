# Causal graph and covariate roles

Every covariate carries one role, and the role determines whether it is matched
on, adjusted for, or held out of the primary analysis.

## Graph

```
                     evidence source
                    /       |        \
                   v        v         v
            label quality  pleiotropy  discovery power
                   |        |    \         |
                   |        v     \        v
   onset --------> discrimination <-- constraint / conservation
     ^                  ^                  ^
     |                  |                  |
 gene class ------------+                  |
     |                                     |
     +-------------------------------------+

  CDS length, expected LoF count, GC content -> constraint estimability
  CDS length, expected LoF count, GC content -> discovery power
  disease-gene status -> citation count  (collider)
```

## Roles

| Variable | Role | Handling |
|---|---|---|
| CDS length, expected LoF count, GC content | Confounder. Affects both how well constraint can be estimated and how likely a gene is to be discovered | Matched on and adjusted for |
| Pleiotropy: effective number of traits across UK Biobank quantitative traits | Confounder for the gene-class account, with a published mechanism | Adjusted for, and central to the mediation analysis |
| Disease breadth: therapeutic areas a gene carries evidence for | Counts the outcome among its inputs, because the evidence includes the associations that define the positives | Reported on its own. Never adjusted for |
| Gene class: developmental annotation share, haploinsufficiency (pLI) | Confounder for the gene-class account. Causes both early onset and constraint | Adjusted for in the mediation analysis |
| Expression breadth (tau), PPI degree | Ambiguous. Mediator of constraint and also a marker of gene class | Enter only the non-evolutionary baseline of the incremental-value estimand. Never matched on or adjusted for in the contrast |
| DepMap gene effect | Mediator, with collider risk | Enters only the non-evolutionary baseline of the incremental-value estimand |
| PubMed citation count | Collider. Disease-gene status causes fame | Not used, since conditioning on a collider opens a path |
| Evidence source | Confounder twice over, through label quality and through pleiotropy | Each source is fitted on its own. Sources are never pooled |
| **Mode of inheritance** | **Confounder.** s_het is a coefficient on heterozygotes, so it is close to blind to recessive disease genes, and the dominant share of Orphanet diseases rises with onset | Entered as a stratifying interaction with each score. The onset slope by mode is a robustness arm |
| L2G score | Label quality for GWAS-sourced positives | Positives require an L2G score of 0.5 or more, and the score's gradient with onset is reported |

## Why mode of inheritance is a confounder and not a detail

s_het is the selection coefficient on heterozygotes. A recessive disease gene
carries almost no heterozygote fitness cost, so the primary constraint metric
cannot see it. Against the pool of scored genes, s_het discriminates dominant
Mendelian disease genes at an AUROC of 0.723 and recessive ones at 0.487.

The pool carries the matching covariates inside it. Against controls matched on
coding sequence length, expected loss-of-function count and base composition, the
same figures are 0.674 and 0.457, and LOEUF, which is above s_het on recessive
genes against the pool, falls below it. Both reference sets are reported: the
pooled figure is what a prioritisation using one metric would see, the matched
one what the metrics are worth with the covariates held fixed.

Inheritance mode is also distributed unevenly along the onset axis. The dominant
share of Orphanet diseases rises from 0.319 in the perinatal bin, through 0.261,
0.410 and 0.568, to 0.714 in the adult bin.

Put together: constraint discriminates dominant disease genes better, and later
bins hold more dominant disease. Constraint's apparent discrimination would
therefore rise with onset from composition alone, which inflates Gamma. The
confound runs in the direction that would manufacture the hypothesised result.

This is the gene-class account in a further form, and it is handled by a
stratifying interaction.

## Why pleiotropy sits at the centre

Spence et al. (2025) show that genome-wide association studies prioritise genes
near trait-specific variants, and can therefore surface highly pleiotropic
genes, while burden tests prioritise trait-specific genes and generally cannot.
Pleiotropy drives constraint. The path from evidence source through pleiotropy
to constraint is therefore a published mechanism, not a conjecture, and it is
exactly the gene-class account. The mediation separates it from the causal
hypothesis, and it is the reason the evidence sources are never pooled.

## Why the primary estimands are contrasts

Ascertainment of disease genes is not uniform along the onset axis, and the
direction of the resulting bias is not knowable in advance. Both estimands are
differences taken inside a single gene set: Gamma differences two score families
against each other, and the log R slope differences a model with evolutionary
features against a model without them. Multiplicative label noise attenuates
both arms of a contrast and cancels to first order, which is why the design is
stated as a contrast rather than as a level.

## Why the registry measurement model biases toward the null

For the complex-disease tiers, what is observed is age at first recorded event:

```
age_recorded = age_onset + diagnostic_delay + registry_left_truncation
```

Finnish registers begin at fixed calendar years, so a participant born in 1940
cannot contribute an event at age 20. FinnGen is hospital-enriched with a high
median participant age, so early-onset events among older participants are
systematically missed. Both inflate apparent onset for genuinely early-onset
conditions, which compresses the onset axis and biases the contrast toward zero.
Individual birth years are not published, so the observation window is bounded
per endpoint from the calendar-year distribution of first events, and endpoints
whose events pile up at the start of the recording window are flagged.
