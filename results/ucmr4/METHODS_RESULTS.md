# UCMR4 HAA observed-grid analysis

## Methods

The analysis paired HAA5, HAA6Br and HAA9 only when PWSID, facility, sample point, collection date, SampleEventCode and SampleID matched exactly. All concentrations were reported as numeric equality results in the source extract. Same-event composition used every complete triplet. The finite-grid replay excluded every PWS with any incomplete three-total event, so the full reference grid did not silently discard unresolved panel results. HAA9 identity checks used a tolerance of 1e-08 µg/L.

The primary endpoint was a chemical signal: at least one selected concentration strictly greater than 60 µg/L. It is not a legal HAA9 standard or compliance determination. A0 was HAA5, A1 was HAA9, M0 was one SampleEventCode round selected uniformly without reference to its concentrations, and M1 was the full finite observed grid. All admissible M0 rounds were enumerated. Outcomes were averaged within PWS before PWS-level averaging. Thus systems with more rounds or sites did not receive more weight. The reference-positive population was defined by Y11=1. Alternative designs selected one fixed observed site across all its rounds, or one observed site-round, with the same within-system enumeration.

Temporal and spatial random-subset curves used exact hypergeometric probabilities. A budget larger than a system's observed universe used all of that system's observed units. The development/holdout label was set only from a salted SHA-256 hash of PWSID: the lowest rounded 20% of digests formed the holdout. Percentile 95% intervals came from 10,000 PWS-cluster resamples (seed 20260913); each resample preserved all candidate-round outcomes for a selected PWS.

Residual-disinfectant codes were joined exactly on PWSID, facility, sample point and SampleEventCode. Conflicting codes were retained as code sets. Source-water bromide and total organic carbon were joined only at PWSID–SampleEventCode for descriptive, non-hydraulically-paired associations. Nondetects were represented by 0–MRL bounds, with MRL/2 as a labeled midpoint sensitivity. Facility-water-type and residual strata use whole-system profiles; systems spanning values are labeled MULTIPLE, and partial residual coverage is labeled PARTIAL_COVERAGE.

## Results

The source contained 63,495 distinct HAA events in 4,924 systems. Exact complete triplets were available for 63,432 events. 63 events in 20 systems were incomplete; excluding those systems left 4,904 systems for finite-grid replay. No complete triplet violated HAA5≤HAA9, HAA6Br≤HAA9, or HAA9≤HAA5+HAA6Br at the stated tolerance.

Among 769 reference-positive systems at 60 µg/L, the system-weighted primary cell probabilities were p00=0.2087 (95% cluster-bootstrap CI 0.1930–0.2244), p10=0.4191 (0.4039–0.4348), p01=0.5774 (0.5424–0.6120), and p11=1 by reference definition. The panel gain at M0 was 0.2103, the grid gain at HAA5 was 0.3687, and the combined gain was 0.7913. The interaction was 0.2123 (0.1843–0.2409).

The mutually exclusive system-weighted repair fractions were already detected 0.2087, panel expansion sufficient while design expansion alone was insufficient 0.1516, design expansion sufficient while panel expansion alone was insufficient 0.3099, either change alone sufficient 0.0587, and both changes required 0.2710. These fractions sum to 1.0000 on the complete reference-positive denominator. The interaction identity error and attribution-sum error were both below 1.11e-16.

Across complete same-event triplets with positive HAA9, the event-weighted median fraction outside HAA5 was 0.2529 (IQR 0.1337–0.4002); after averaging events within PWS first, the median system value was 0.2867 (IQR 0.1561–0.4127). These are measured mass fractions, not event probabilities or health-risk fractions.

Threshold, holdout, one-site, one-site-round, temporal/spatial subset, stratum, geographic and indicator sensitivity results are in the accompanying tidy tables. Stratum estimates with fewer than 20 reference-positive systems and state estimates with fewer than 10 are flagged. The observed-grid design cannot identify subgrid excursions, continuous event duration, unobserved household exposure, legal violations, or national prevalence.
