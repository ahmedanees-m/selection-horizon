# Causal graph and covariate roles

The graph is frozen at pre-registration and is not revisited after data
inspection. Every covariate carries one role, and the role determines whether it
is matched on, adjusted for, or held out of the primary analysis.

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
| Pleiotropy: Spence trait-specificity, count of distinct Mondo parents | Confounder for C2, with a published mechanism | Adjusted for, and central to the mediation analysis |
| Gene class: developmental expression, haploinsufficiency | Confounder for C2. Causes both early onset and constraint | Adjusted for, primary mediator in the decomposition |
| Expression breadth (tau), PPI degree | Ambiguous. Mediator of constraint and also a marker of gene class | Reported adjusted and unadjusted. Never used in primary matching |
| DepMap gene effect | Mediator, with collider risk | Sensitivity only |
| PubMed citation count | Collider. Disease-gene status causes fame | Sensitivity only. Never in primary matching |
| Evidence source | Confounder twice over, through label quality and through pleiotropy | Stratified from the outset. Sources are never pooled in the primary estimand |
| **Mode of inheritance** | **Confounder.** s_het is a coefficient on heterozygotes, so it is close to blind to recessive disease genes, and the dominant share of Orphanet diseases rises with onset | Stratified from the outset. Autosomal dominant is the primary Mendelian stratum; all-inheritance results are reported adjusted for mode |
| L2G SHAP contributions | Label-quality covariate for GWAS-sourced positives | Adjusted for within the GWAS arm |

## Why mode of inheritance is a confounder and not a detail

s_het is the selection coefficient on heterozygotes. A recessive disease gene
carries almost no heterozygote fitness cost, so the primary constraint metric
cannot see it. Measured on the realised gene sets, pooled discrimination for
Orphanet disease genes is 0.600 by s_het across all inheritance modes, 0.672
restricted to autosomal dominant, and 0.547 restricted to autosomal recessive.

Inheritance mode is also distributed unevenly along the onset axis. The dominant
share of Orphanet diseases rises from 0.319 in the perinatal bin, through 0.261,
0.410 and 0.568, to 0.714 in the adult bin.

Put together: constraint discriminates dominant disease genes better, and later
bins hold more dominant disease. Constraint's apparent discrimination would
therefore rise with onset from composition alone, which inflates
`Gamma = gamma_conservation - gamma_constraint` upward. The confound runs in the
direction that would manufacture the hypothesised result.

This is the gene-class account in a further form. It is handled by
stratification from the outset rather than by a robustness arm.

## Why pleiotropy sits at the centre

Spence et al. (2025) show that genome-wide association studies prioritise genes
near trait-specific variants, and can therefore surface highly pleiotropic
genes, while burden tests prioritise trait-specific genes and generally cannot.
Pleiotropy drives constraint. The path from evidence source through pleiotropy
to constraint is therefore a published mechanism, not a conjecture, and it is
exactly the gene-class account. The mediation separates it from the causal
hypothesis, and stratifying by evidence source from the outset is the reason the
sources are never pooled.

## Why the primary estimands are contrasts

Ascertainment of disease genes is not uniform along the onset axis, and the
direction of the resulting bias is not knowable in advance. Both estimands are
differences taken inside a single gene set: Gamma differences two score families
against each other, and the log R slope differences a model with evolutionary
features against a model without them. Multiplicative label noise attenuates
both arms of a contrast and cancels to first order. This is the reason the
design is stated as a contrast rather than as a level, and it is the main
methodological argument the manuscript makes.

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
If Gamma survives, it survives despite a conservative measurement. The primary
analysis restricts to birth cohorts with full coverage across the relevant age
range and reports the full sample as a sensitivity arm.
