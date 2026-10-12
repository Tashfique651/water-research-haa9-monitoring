# Retrospective HAA9 budget and policy audit, frozen 19 September 2026

This protocol is written before new empirical calculations for this extension. It is not a prospective preregistration: the investigators and assistant have seen earlier archive analyses, including HAA9 captures at 4, 8 and 12 physical samples and the warm-season means. The new full-curve family and paired procedures below are fixed before their calculation. All results, including null or negative contrasts, will be retained.

## Population and endpoint

Use the frozen released UCMR4 paired-event dataset without changing its completeness rules. Retain all events in the 4,904 complete-grid PWSs. The primary reference is full observed HAA9 maximum strictly greater than 60 micrograms/L, expected to identify 769 PWSs. This is an analytical benchmark, not an HAA9 compliance standard, and not an independently measured health endpoint. Use equal PWS weights.

## Exact physical-sample budgets

Audit the physical-sample key (PWSID, SampleID) against dates, sites, event codes and values. If a sample identifier maps to multiple physical records, stop and report the conflict instead of collapsing it silently. At each requested integer k from 1 through 16, select min(k,n) unique physical samples uniformly without replacement. Exact capture is 1 - choose(n-m,k)/choose(n,k), with saturation at n. Evaluate both panels on the identical selected support. Compute two paired contrast families: HAA9-k minus HAA5-k and HAA9-k minus the entire observed HAA5 record. All 32 contrasts form one simultaneous family.

Use 10,000 paired whole-PWS bootstrap samples with seed 20260919. Report exact means, percentile pointwise 95% intervals, and a 95% simultaneous band using the 95th percentile of the bootstrap maximum absolute standardized centered deviation over the declared 32 contrasts. Standardization uses each original system-contribution standard error; zero-variance contrasts have zero-width bands. Report the first integer with a positive point estimate, positive pointwise lower bound and positive simultaneous lower bound as retrospective curve summaries, not independently confirmed thresholds.

Report available and realized physical samples, systems saturated at k and strictly below k, exact expected distinct collection dates and sites, and their cohort means. Collection dates and sites are resource descriptors, not measured field visits or laboratory costs. Add the previously reported HAA9-k4 versus HAA5-k12 comparison as a separately labelled exploratory contrast, outside the primary family.

## Other sampling units

Repeat exact paired budgets 1 through 4 for uniformly selected administrative event-code groups and for distribution sites. A selected site includes its complete observed temporal record; a selected administrative group includes its complete observed spatial support. These are not equal physical-sample or equal-cost comparisons. Report expected physical samples and saturation.

## Panel by season

For administrative groups use median collection date, as in the prior archive policy. For actual collection-date units use the actual date. For physical samples use each actual collection date. Warm season is May through September inclusive. Compare uniform selection over all units with uniform selection over warm units, falling back to all units if no warm unit exists. Both panels always share the same selection rule. Primary policy contrast is (HAA9 warm - HAA5 warm) - (HAA9 uniform - HAA5 uniform). Retain the four paired cells, HAA5 and HAA9 seasonal gains, and the policy interaction with paired bootstrap intervals. For physical samples report k=1..4; warm-first selection randomly supplements with nonwarm samples if fewer than min(k,n) warm samples exist, keeping realized sample counts equal. Stratify the one-unit policy by source-water profile descriptively, without outcome-selected strata. Use all 4,904 PWSs and the fixed 769-PWS reference cohort, with conditional results primary.

## Audits

Recompute the original four cells and interaction across the reference cohort and all eligible systems. Audit the archived highest-HAA9 administrative-group oracle by choosing the HAA9-maximizing group once, uniformly resolving exact ties, and evaluating both panels on that group. Compare per-system archived results to corrected shared-selection results. Include a synthetic counterexample in which best HAA9 and best HAA5 differ.

Where archived alternating-direction data are available, compute direction-row-weighted and equal-PWS-weighted pooled interaction; bootstrap PWS clusters, preserving all directions within each cluster. These are different estimands. Retain source paths and input hashes. No claim of independent validation arises from reprocessing the same archive.

## Verification

Test exact capture against brute-force subset enumeration, zero/full-budget boundaries and saturation, expected distinct support against enumeration, matched warm-first sampling against enumeration, oracle shared selection, and synthetic nested examples with positive, zero and negative additive interaction. Save machine-readable outputs and a canonical summary with exact estimates reused by manuscript/figure production.
