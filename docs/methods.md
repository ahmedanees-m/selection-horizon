# Methods

How the analysis is specified and what each stage of the pipeline computes.

## Design

The unit of analysis is a gene-disease pair. For gene g and disease d, y is 1 if
g is a positive for d under a given evidence source. One logistic model is
fitted per evidence source over that source's pairs:

```
logit P(y[g,d] = 1) = alpha
    + sum_s ( beta_s z[s,g] + gamma_s z[s,g] x onset[d] )
    + delta onset[d] + theta' x[g] + log(n_genes[d])
    + sum_m ( lambda_m mode[m,d] + sum_s nu[s,m] z[s,g] x mode[m,d] )
```

`z[s,g]` is score s standardised within the analysis set, onset is the disease's
onset level scaled to the unit interval and centred, and `x[g]` holds the
confounders: CDS length decile, expected loss-of-function count decile, and GC
content. `mode[m,d]` are indicators for dominant and for recessive inheritance,
with mixed or unknown as the reference.

Genes recur across diseases through pleiotropy and diseases cluster their genes,
so rows are correlated in two crossed directions. The covariance is a two-way
cluster-robust sandwich over disease and gene, in the Cameron, Gelbach and
Miller construction, which is valid under arbitrary correlation within a disease
and within a gene, assumes no distribution for either, and is fast enough to
carry a permutation null of several thousand replicates. Its coefficients are
population-averaged, which is what a contrast of two interaction terms calls
for, and it is the specification the power simulation is calibrated on. Onset
enters linearly, with the midpoint of the annotated span as the sensitivity
encoding. Evidence sources are never pooled, because label quality and
pleiotropy differ across them.

A one-stage model is used rather than per-disease AUROC followed by a regression
on onset. Per-disease AUROC is bounded, heteroskedastic, and noisy at ten
positives, which both attenuates a second-stage slope and forces the exclusion
of small diseases. Excluding small diseases would remove most of the Mendelian
arm, where the median disease has one causal gene.

### The onset contrast

The decay of a score is how much discriminative power it loses across the onset
range, the negative of its score-by-onset interaction coefficient, so

```
Gamma = decay_conservation - decay_constraint = gamma_constraint - gamma_conservation
```

estimated as a contrast of the two interaction coefficients on the fitted
cluster-robust covariance. Gamma above zero means deep-time conservation loses
discriminative power faster than recent human constraint as onset rises. Its
interval is a Wald interval from that covariance at alpha 0.025, the error rate
the design is powered at, and so has 97.5% coverage.

### The incremental-value slope

Per disease, cross-validated discrimination of nested models. M0 holds the
non-evolutionary features; M1 adds the evolutionary scores.

```
R(d) = ( AUC_M1(d) - 0.5 ) / ( AUC_M0(d) - 0.5 )
estimand = slope of log R(d) on onset
```

The ratio rather than the difference, because multiplicative attenuation from
label noise cancels in a ratio and label noise is the confound that varies
systematically along the onset axis. Diseases with `AUC_M0(d) - 0.5` at or below
0.05 are excluded for stability, and the difference version is reported as a
sensitivity arm.

## Onset

### T1a, Orphanet, primary Mendelian axis

Average age of onset intervals from the Orphanet natural-history product:
antenatal, neonatal, infancy, childhood, adolescent, adult, elderly, all ages.
A disease may carry several intervals. The primary encoding is the earliest,
which is the quantity selection acts on, since selection registers the earliest
fitness-relevant manifestation rather than an average one. The sensitivity
encoding is the midpoint of the span between the earliest and latest interval.
Diseases annotated only as spanning all ages carry no ordinal position and are
excluded from the continuous axis while remaining in the coverage counts.

### T1b, HPO, secondary

The onset sub-ontology HP:0003674 through `phenotype.hpoa`. Coverage is roughly
a quarter of Orphanet's, so this tier is used only for the concordance figure
against T1a, never as a primary axis.

### Which column each arm reads

Onset is not one column, and the choice is made from the evidence source rather
than defaulted. The Mendelian arm reads the Orphanet interval; the three Platform
arms read the decade bin on the registry median age. Reading the Mendelian column
for a Platform arm keeps only the diseases carrying both labels, which is about a
fifth of the arm, and places those on an axis they were not measured against.
Both vocabularies scale onto the unit interval, which is what makes a Gamma from
one arm the same quantity as a Gamma from the other. A set of positives mixing
the two families raises rather than picking one.

### T2 and T3, complex disease

T2 is the Risteys median age at first event together with the Aalen-Johansen
competing-risks cumulative incidence, which already treats death as a competing
event and is stratified by sex with age as the timescale. T3 is the Kuan et al.
median age at first record across 308 conditions, independent of FinnGen, and
the T2 to T3 concordance figure is the external check on the complex-disease
axis.

### The registry measurement model

What a register records is age at first recorded event:

```
age_recorded = age_onset + diagnostic_delay + registry_left_truncation
```

Finnish registers begin at fixed calendar years. Individual birth years are not
published, so the observation window is bounded per endpoint from the
calendar-year distribution of first events, and endpoints whose events pile up
at the start of the recording window are flagged. FinnGen is hospital-enriched with a high median participant
age, so early-onset events among older participants are systematically missed.
Risteys distribution bars are aggregated to include at least five individuals,
so endpoints whose cumulative incidence is too coarse to convert to a hazard are
excluded. T2 is also run as an ordinal rank in a sensitivity arm, because
electronic health record timestamps reflect adult onset poorly.

The direction of this bias is conservative. Left truncation inflates apparent
onset for genuinely early-onset conditions, which
compresses the onset axis and biases the contrast toward zero.

## Fitness cost

Hamilton's selection gradient rather than reproductive value alone. Residual
reproductive weight at age a:

```
W(a) = sum_{x >= a} exp(-r x) l(x) m(x)
```

with r the intrinsic growth rate solved from the Euler-Lotka equation. Fisher
reproductive value is the same tail renormalised by survivorship,
`v(a) = (exp(r a) / l(a)) W(a)`. W is monotone non-increasing by construction; v
rises through the pre-reproductive years and peaks near the age at first
reproduction. Both are reported.

Fitness cost of disease d:

```
s_d = kappa_d * sum_a h_d(a) W(a) / W(0)
```

`h_d(a)` is the age-specific hazard, obtained from the Aalen-Johansen cumulative
incidence. `kappa_d` is the per-case reduction in residual reproduction. Orphanet
does not publish an age of death, so `kappa_d` is built as an index over
lethality, fertility and care burden from the HPO annotations
(`src/onset/severity.py`).

Fitness cost of an allele is `s_allele = delta_pi * s_d`, with `delta_pi` the
genotype-specific increment in lifetime risk. Without it the complex-disease arm
would be flat for a trivial reason, since an allele with an odds ratio of 1.05
carries almost no fitness cost at any onset.

Ancestral re-weighting applies ancestral survivorship and fertility to the
hazard, never to a modern cumulative incidence. Multiplying modern cumulative
incidence by an ancestral life table double counts modern survival and forces
the modern and ancestral indices to agree; a correlation above 0.95 between them
is the diagnostic that this happened.

## Controls

Each control replaces both evolutionary scores with a single control score and
refits the onset slope, so no control is a contrast between score families.

| Control | Type | Expected |
|---|---|---|
| Coding sequence length | Negative | Flat onset slope |
| Tissue specificity | Negative | No decay |
| Developmental annotation | Positive | Steep decay |
| Onset permuted within therapeutic area | Placebo | Null centred on zero |

The positive control shows that the onset axis resolves developmental gene
class; a decay curve estimated on an axis that did not would not be
interpretable. What it establishes is narrower than it looks: developmental genes are also pleiotropic and haploinsufficient,
so passing shows the axis resolves gene class rather than fitness cost. It is a
floor, not a confirmation.

Tissue specificity is measured without conditioning on the affected tissue,
because no disease-to-tissue mapping exists in the sources used, and the
developmental control uses the developmental branch of the Gene Ontology,
because the fetal transcriptome is deposited as alignments rather than as a
joinable per-gene matrix. Measured values are reported in the paper.
