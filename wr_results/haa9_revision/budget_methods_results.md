# Budget, season and implementation audit

## Methods and analysis history

This is a retrospective extension of an already inspected archive, not independent confirmation or a prospective preregistration. The dated protocol was saved before the new paired calculations. Earlier means at selected budgets and the administrative-group seasonal means were known. The full integer budget family, selection policies, resampling scheme and reporting rules were then frozen.

The sample key, PWSID plus SampleID, is unique across the 63,105 complete-grid eligible paired records. All 4,904 eligible systems were retained. The fixed reference cohort consists of 769 systems whose full observed HAA9 maximum exceeded 60 micrograms/L. The benchmark is analytical, not an HAA9 legal standard. Both panels were evaluated on the same sampled support. Capture was calculated exactly under uniform sampling without replacement, using min(k,n) physical samples. A 10,000-replicate paired whole-PWS bootstrap characterized between-system variation. The primary simultaneous band controls a declared family of 32 contrasts: the 16 HAA9-k minus HAA5-k differences and the 16 HAA9-k minus full-record HAA5 differences. It uses a bootstrap maximum absolute standardized centered deviation, with original system-level standard errors.

## Canonical findings

Up to eight HAA9 samples captured 79.8033% of reference-positive systems, compared with 43.2230% for the same HAA5 support and 57.7373% for the complete HAA5 record. The gain over the full HAA5 record was 22.0660 percentage points, with pointwise 95% interval 18.7026 to 25.4541 and simultaneous interval 17.8509 to 26.2810 points. The same-budget advantage was 36.5803 points, with simultaneous lower bound 33.3896 points. All 16 same-budget comparisons had positive simultaneous lower bounds.

The first integer budget with a positive simultaneous lower bound against the complete HAA5 record was five: the mean gain was 7.1732 points and the simultaneous interval was 3.0296 to 11.3168 points. This is a retrospective crossing summary, not an independently validated deployment threshold. At four samples the gain remained 0.1425 points with pointwise interval -3.1609 to 3.5383 and simultaneous interval -3.9998 to 4.2847 points. It does not demonstrate superiority or formal equivalence.

Reference-positive systems had a mean 18.4018 physical samples in the full record. The eight-sample policy realized 7.8661 samples on average; 27 systems had fewer than eight available and 143 were fully sampled at that budget. The expected selected support included 3.8001 distinct dates and 4.0335 sites. These are not observed field visits or monetary costs.

Two administrative event-code groups captured 69.0507% with HAA9 and 36.5626% with HAA5. Their HAA9 gain over the full HAA5 record was 11.3134 points, pointwise interval 7.7807 to 14.7594 points. This policy used 9.5815 physical samples on average. Two distribution sites, retaining their full observed temporal records, captured 78.9971% with HAA9; their gain over the full HAA5 record was 21.2597 points, interval 17.9405 to 24.6987 points. Site and administrative-group policies are different resource investments, not matched-cost policies.

## Season and population interpretation

Warm months were fixed as May through September. When no warm unit existed, the selector used the full record uniformly. Physical-sample policies randomly supplemented warm samples with nonwarm samples if needed to preserve the same realized sample budget. The policy interaction, defined as the warm HAA9-minus-HAA5 gap minus the uniform gap, was 7.1088 points for administrative groups (95% interval 5.1365 to 9.0811), 6.7392 for actual collection dates (4.8454 to 8.6266), and 4.3664 for one physical sample (3.3436 to 5.4011). At four matched physical samples it was 2.6268 points (0.7830 to 4.5151). These are secondary, pointwise retrospective comparisons, not proof of universal or future optimality.

The panel-by-full-grid interaction was 21.2289 points among the 769 reference-positive systems (18.3355 to 24.0683), but 3.3289 points across all 4,904 eligible systems (2.8259 to 3.8302). Neither is a national prevalence estimate. The finite-grid means are deterministic; bootstrap intervals do not quantify missed conditions between observations or analytical error.

## Oracle and directional weighting

The archived oracle implementation was correct: it selected each system's HAA9-maximizing administrative group once and evaluated both panels on that group. All 769 archived rows matched the independent recomputation. There were no maximum ties, and the selected group empirically captured every full-record HAA5 signal. Therefore the zero oracle interaction is genuine in this archive. It is not a mathematical identity, as the retained synthetic counterexample demonstrates.

The alternating-fold pooled analysis contains 1,021 direction rows from 765 systems, including 256 systems with two qualifying directions. Equal direction-row weighting gives a 4.8972-point interaction (PWS-cluster interval 3.3757 to 6.4004). Equal-PWS weighting after averaging each system's directions gives 3.2680 points (2.2541 to 4.3137). The latter should be primary if the manuscript claims equal system weights; the former is a labelled sensitivity. The chronology, reference definition and evaluation duration also differ from the original factorial, so attenuation cannot be attributed to conditioning alone.

## Verification and source files

The independent 24-test suite passed, including exhaustive small-grid capture enumeration, expected distinct support, warm-first selection, oracle tie handling, the oracle counterexample, and nested synthetic positive, zero and negative interactions. Machine-readable canonical outputs are in this directory: budget_curve, budget_systems, budget_cross_k_exploratory, support_budget_curve, support_budget_systems, season_summary, season_systems, season_source_strata, oracle_interaction_summary, oracle_archive_comparison and support_direction_weighting. Each table is supplied as CSV and Parquet. The input hash registry is budget_input_manifest.csv; protocol digest and crossing summaries are in budget_summary.json.
