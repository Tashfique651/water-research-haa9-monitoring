# Forward screening and common-support endpoint results

## Analysis status and support

The primary forward comparison is grouped internal validation under a frozen retrospective protocol, not independent confirmation. Prior drafts had already inspected the archive. Among 2,693 PWS with exactly four administrative event-code groups, 57 failed strict temporal separation because a date in the earlier pair overlapped or followed a date in the later pair. The resulting cohort contains 2,636 PWS, including 420 with a later HAA9 maximum above the 60 micrograms/L analytical benchmark and 2,216 without that later signal. The retained 24,211 predictor samples map one-to-one to species records on PWS/sample identity; collection dates and event codes also match. Two predictor samples retain previously documented species-sum versus official-total discrepancies, with a maximum difference of 0.5 micrograms/L. No original value was altered or system discarded to obtain agreement.

The first two chronological groups supply predictors, and the last two supply outcomes. All samples from a PWS remain in the same SHA256-assigned outer fold. Features use log1p concentrations in micrograms/L and training-only centering/scaling. Four fixed L2-penalized logistic models use HAA5 maximum, HAA9 maximum, five species maxima, or nine species maxima. Nondetect species are zero in the released lower-concentration realization. Calibrated thresholds target training follow-up fractions of 5%, 10%, 15%, 20%, and 30%; cutoff ties share a common fractional selection probability. Evaluation workload is measured, not assumed to equal its training target.

## Primary forward results

Early HAA9 maximum has AUROC 0.80925 compared with 0.78778 for early HAA5 maximum. The paired difference is 0.02146, with a 95% conditional PWS-bootstrap interval of 0.01319 to 0.03020. Average precision is 0.43002 versus 0.39922, with paired difference 0.03080 (0.01753 to 0.04428). These improvements concern later HAA9 signal prediction, not health outcomes or compliance violations.

At the 10% training workload target, HAA9 sensitivity is 32.62% versus HAA5 sensitivity 28.79%. Achieved follow-up fractions are 9.901% and 10.049%, respectively. The paired sensitivity difference is 3.83 percentage points (pointwise 95% interval 1.06 to 6.61 points; simultaneous five-workload band 0.08 to 7.58 points). At the 30% target, the difference is 5.00 points (simultaneous band 1.25 to 8.75 points). Simultaneous bands include zero at 5%, 15%, and 20%. The complete family must be shown; no universal workload-independent superiority is established.

The nine-species model has AUROC 0.80505 versus 0.79624 for the five-species model, and average precision 0.43915 versus 0.41566. The corresponding paired differences are 0.00881 (0.00046 to 0.01738) and 0.02349 (0.00545 to 0.04221). Brier scores are 0.11017 and 0.11255. Although overall discrimination and probability accuracy improve modestly, all five workload-specific simultaneous sensitivity bands include zero. The scalar HAA9 total and nine-species information vector are distinct predictors and should not be conflated.

Direct early HAA9 >60 produces sensitivity 46.43%, compared with 28.10% for HAA5 >60, but increases follow-up from 9.71% to 16.50%. It is not an equal-workload comparison. False-positive rates are 10.83% and 6.23%, showing that specificity is no longer structurally perfect against a disjoint later outcome. The hypothetical staged same-sample strategy gates HAA9 reporting after HAA5 screening; its analytical selection and follow-up fractions are reported separately. The archive does not establish specimen retention, feasible holding times, a second-visit workflow, or economic superiority.

## Post-protocol exact-capacity sensitivity

This sensitivity was added after inspecting the training-calibrated results, with that development history recorded separately. It allocates the same integer number of follow-up slots within each outer evaluation fold, using only out-of-fold scores and a fixed PWS-hash tie breaker. It is a batch-ranking allocation policy, not a deployable single-system threshold.

At 10% nominal capacity, both panels allocate 261 slots among 2,636 PWS. HAA9 captures 136 of 420 later-positive systems (32.38%), compared with 120 (28.57%) for HAA5. The paired gain is 3.81 points, with simultaneous band 0.07 to 7.54 points. At 30%, 789 slots produce a gain of 4.29 points (0.55 to 8.02 points). Other scalar capacities have simultaneous bands including zero. Nine-species versus five-species model sensitivity is slightly lower at 20% capacity, and all five simultaneous bands include zero.

All intervals use 5,000 paired whole-PWS resamples with fits, cutoffs, and allocations held fixed. They omit retraining, selection of the analysis family, temporal sampling, analytical uncertainty, and transport to new populations. Fixed-bin probability calibration, all operating points, fitted coefficients, training transforms, learned cutoffs, individual predictions, and exact-capacity allocations are supplied in machine-readable form.

## Common-support endpoint comparison

The same 2,487 PWS and 11,135 complete four-consecutive-quarter site windows support maximum, OEL-form, and LRAA-form endpoints. The maximum uses physical samples from those windows. OEL-form is the maximum of the two eligible three-quarter weighted means within each four-quarter window; LRAA-form is the four-quarter mean. PWS scores take the maximum across identical qualifying sites/windows. These are research analogues, not legal compliance determinations.

HAA9-positive counts are 610 for the sample maximum, 229 for OEL-form, and 136 for LRAA-form. Corresponding HAA5-positive counts are 365, 109, and 48. Thus endpoint-specific HAA5 capture falls from 59.84% to 47.60% to 35.29%. This does not mean that aggregation uniformly enlarges the absolute panel benefit. Across all 2,487 common-support PWS, the HAA9-minus-HAA5 positive-fraction gap falls from 9.85 to 4.83 to 3.54 percentage points. Among the fixed 610 maximum-HAA9-positive PWS, the gap falls from 40.16 to 19.67 to 14.43 points. Averaging reduces both panels' absolute positive yields while increasing the HAA5 missed fraction within the smaller endpoint-specific HAA9-positive groups.

## Audit coverage

Nine independent test functions cover tied workload calibration; fractional operating metrics; AUROC/AP agreement with expanded bootstrap records; temporal and PWS-fold separation; reconstruction of training-only scalar cutoffs and transformations; out-of-fold probabilities from archived fitted coefficients; common-window support and panel nesting; and exact label-free capacity allocation. Existing broader-cohort endpoint analyses remain intact.
