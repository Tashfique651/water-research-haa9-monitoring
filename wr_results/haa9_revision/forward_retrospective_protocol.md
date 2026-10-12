# Frozen retrospective forward-screening and common-support protocol

Recorded 19 September 2026 before fitting the new forward-screening comparisons. Earlier drafts and analyses already inspected this archive. This is grouped internal validation under a frozen retrospective protocol, not an untouched holdout or independent confirmation. The policy family and analytical settings below will not be selected for favorable HAA9 results.

## Cohort and temporal target

Start with the existing UCMR4 eligible complete-grid PWS cohort. Aggregate exact matched samples within PWS and administrative event-code groups. Retain PWS with exactly four groups, ordered by earliest collection date and event code. Predictor support is all samples in the first two groups; outcome support is all samples in the last two groups. Require the latest predictor date to be strictly earlier than the earliest outcome date. Report excluded PWS and temporal boundaries. Retain both positive and negative later outcomes. The common primary endpoint is any later HAA9 result strictly above 60 micrograms/L. This analytical benchmark is not an HAA9 regulatory limit.

## Frozen comparisons

Use five outer folds assigned by SHA256 of PWS identifier with a fixed salt. Entire systems stay in one fold. Continuous comparators are first-block HAA5 maximum and HAA9 maximum. Fixed threshold comparators apply >60 micrograms/L. Workload-calibrated scalar rules select thresholds using training scores only at the declared 5%, 10%, 15%, 20%, and 30% follow-up targets. At a cutoff tie, all tied scores receive the same selection probability needed to attain the training workload; do not select individual ties using outcomes. Evaluation follow-up fractions may differ from the nominal training targets and will be reported. No exact evaluation-workload equality will be claimed.

Probability models are fixed penalized logistic regressions (L2, C=1) with (1) early HAA5 maximum, (2) early HAA9 maximum, (3) the five HAA5 individual-species early maxima, and (4) all nine individual-species early maxima when the complete released species records map exactly to all eligible predictor samples. Features are log1p of concentrations in micrograms/L, centered and scaled using outer-training data only. Fixed model specification avoids outcome-selected hyperparameters. Model-score workload thresholds are learned only on outer-training predictions. These are training-calibrated operating rules, not internally optimized treatment rules.

A same-sample hypothetical staged strategy gates broader HAA9 reporting at the training-calibrated HAA5 threshold, then triggers follow-up only when early HAA9 >60. The fraction requiring broader analysis and the resulting follow-up workload are reported separately. Its specimen retention, holding times, feasibility, and incremental cost are not established by these archive records. No causal, compliance-replacement, or cost-effectiveness interpretation is permitted.

## Metrics and uncertainty

For every operating rule report sensitivity, missed-positive fraction and count, positive predictive value, false-positive rate, and achieved follow-up fraction. For continuous scores report AUROC and average precision. For out-of-fold logistic probabilities also report Brier score, log loss, and fixed-bin observed-versus-predicted calibration. Whole-PWS paired percentile bootstrap, 5,000 replicates and fixed seed, resamples the out-of-fold records with fitted models and learned cutoffs held fixed. Report paired HAA9 minus HAA5 differences at every workload and for the continuous/probability metrics. Intervals are conditional on these fitted models and folds and omit retraining, model-selection, transport, temporal sampling, and analytical measurement uncertainty. They are descriptive, not a post-selection confirmatory test. A simultaneous centered-bootstrap band across the five workload contrasts will also be provided for sensitivity differences.

## Common-support endpoint analysis

Use identical systems, sites, and complete four-consecutive-calendar-quarter windows for all three endpoints. Quarterly concentrations are arithmetic means of eligible samples at the site. Within each qualifying window compute the maximum physical-sample concentration, the maximum of the two available three-quarter OEL-form values (Q1+Q2+2Q3)/4 and (Q2+Q3+2Q4)/4, and the four-quarter mean. Aggregate each endpoint by maximum across the same eligible windows/sites for a PWS. Report endpoint-specific HAA9-positive capture, fixed-reference capture among common-window maximum-HAA9-positive systems, and positivity among all common-support PWS. Preserve broader-cohort baseline endpoint analyses. OEL-form and LRAA-form results are research analogues, not regulatory determinations.

## Audit and reporting

Verify exact sample mapping, unique PWS, fold partition nonoverlap, strict date separation, training-only transformation/cutoff fitting, probability bounds, cohort denominators, nested panel positivity, and complete consecutive-quarter support. Publish all operating points, unfavorable and null contrasts, training cutoffs, out-of-fold predictions, individual common-window scores, and software settings. Do not relabel any retrospective result as prespecified or independently confirmed.
