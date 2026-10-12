# Cross-archive validation: methods and results

## Scope and non-interpretive guardrails

This analysis validates data structure, paired concentration patterns, and finite-grid sampling capture. It does **not** estimate legal compliance, individual exposure, health effects, or causal effects. A concentration equal to the stated threshold is classified as not above; the event definition is strictly `>60 µg/L` for HAA5 (and the non-regulatory HAA9 comparison) and `>80 µg/L` for TTHM.

## SYR4 pairing and censoring

HAA5 and TTHM were inner-joined on the exact composite key `PWSID + SAMPLE_ID`. Missing sample IDs were never matched. Within each analyte, duplicate keys were collapsed only when all analysis-critical fields agreed; conflicting keys were excluded. The retained primary cohort additionally required the HAA5 and TTHM collection dates to agree. Sampling-point database IDs were audited but not required because analyte-specific IDs can differ despite a matching sample key, date, and facility.

Detected values were used as point concentrations. Non-detects were represented as `[0, detection limit]`; positive limits reported in mg/L were multiplied by 1,000 and limits with a missing unit were assumed to be µg/L because all detected values in these analyte files were reported in µg/L. Zero, missing, negative, or unrecognized-unit limits have an unknown upper bound. A result was definitely above when its lower bound exceeded the threshold, definitely below when its upper bound was at or below the threshold, and otherwise indeterminate. Midpoint (`DL/2`), lower-bound, and upper-bound summaries are provided as sensitivity analyses, not replacements for the interval classification.

The two source files contained 1,005,235 HAA5 and 1,089,557 TTHM rows. Conflict-resolved exact-key pairing produced 527,319 pairs; 527,296 same-date pairs from 26,540 systems entered primary analyses. There were 436,150 pairs with both analytes detected and 510,728 pairs with both threshold statuses classifiable. Threshold-status agreement was 0.932 (95% system-cluster bootstrap CI 0.930–0.935). Detected-pair Spearman rho was 0.619 (95% fixed-rank system-cluster bootstrap CI 0.609–0.628). Source-water, population-size, system-type, and seasonal-quarter summaries are descriptive and retain their event and system denominators.

For the SYR4 finite-grid analysis, a system-year required at least three distinct sampled quarters and complete interval classification in every included quarter. A quarter was positive if any same-date exact pair was definitely above the analyte threshold. Within each 3- or 4-quarter grid and conditional on at least one positive quarter, capture for sampling `k` quarters without replacement was calculated exactly as `1 - C(Q-m,k)/C(Q,k)`.

## ICR extraction, identities, and event cohorts

The Access tables `TUXDBP`, `TUXSAMPLE`, `TUXUTMON`, `TUXWQP`, and `TUXDISFRES` were extracted without value substitution and written to CSV and Parquet. The ICR user guide's general rule GR012000140 defines `-999` in analytical results as below the minimum reporting level (MRL); it is never interpreted as a negative concentration. The WQP rules add a special averaged-sample convention for TOC, TOX, and UV-254: `-999` is retained when at least one initial/duplicate result is below MRL and the prescribed MRL-substitution average equals the MRL. For DBP totals, below-MRL constituents are replaced by zero during summation, while a total is coded `-999` when every constituent is below MRL (DBP011000071 for HAA5 and DBP011000131 for HAA9). Because analyte-specific MRLs are absent from these extracts, `-999` totals are excluded from numeric threshold support rather than silently set to zero.

The primary ICR distribution cohort is exactly `AVG, AVG1, AVG2, MAX, NFC`. The prespecified all-distribution-system sensitivity is exactly `ENT, DSE, AVG, AVG1, AVG2, MAX, SDS, NFC`. The primary cohort contains 8,084 DBP events and the all-DS sensitivity contains 13,282.

The HAA5 identity was evaluable for 15,207 events and matched the five-component rule in 15,207, with 0 mismatches. HAA9 was evaluable for 3,461 events and matched the nine-component rule in 3,461, with 0 mismatches. HAA9 availability is reported explicitly by event type, sampling period, quarter, and cohort; it is not treated as if universally measured.

## ICR temporal capture

For each system and sampling period, the maximum nonnegative numeric analyte result within the specified event scope represented that period. Quarters were similarly represented by their maximum supported period result. A full-window system had numeric support in each of ICR quarters 1–6; missing and `-999` results did not count as support. Capture curves are conditional on a full-window system having at least one observed value strictly above 60 µg/L. For each system's finite supported grid, random-subset capture is the exact hypergeometric complement `1 - C(N-M,k)/C(N,k)`. Curves stop at the minimum supported grid size shared by the cohort. D50/D90 are reported only if the empirical curve reaches 0.50/0.90 on that grid; otherwise the output says `not reached on supported grid`.

- HAA5 monthly: 151 systems had numeric support in all six quarters; 42 had at least one value >60 µg/L.
- HAA5 quarterly: 151 systems had numeric support in all six quarters; 42 had at least one value >60 µg/L.
- HAA9 monthly: 8 systems had numeric support in all six quarters; 2 had at least one value >60 µg/L.
- HAA9 quarterly: 8 systems had numeric support in all six quarters; 2 had at least one value >60 µg/L.

## Matched HAA5 × HAA9 factorial addendum

The compact ICR factorial uses only primary-distribution DBP events with paired nonnegative HAA5 and HAA9 at the same `EVENT_ID`. Within each system, events are collapsed to the maximum at each observed sampling period (monthly grid) or quarter (quarterly grid); both panels therefore share an identical finite grid. `M0` selects one grid period uniformly, evaluated by exact enumeration, and `M1` uses the complete observed grid. `p00`, `p10`, `p01`, and `p11` respectively denote HAA5/M0, HAA9/M0, HAA5/M1, and HAA9/M1 signals at the common research benchmark B=60 µg/L. The primary addendum is the available-period monthly cohort with at least two paired periods; available-quarter results and a strict paired six-quarter sensitivity are also reported. Conditional results use systems positive under HAA9/M1, so `p11=1` by definition and is not a claim of perfect physical or regulatory detection. The five exhaustive fractions are already detected, panel-only repair, design-only repair, either-change repair, and both-changes required; their sum is audited on the common reference-positive denominator. Interaction is `p11-p10-p01+p00`, a detection-scale interaction rather than biological synergy.

## Uncertainty and holdout

All reported bootstrap intervals use 10,000 fixed-seed (`20260913`) nonparametric whole-system resamples. For observation-level Spearman rho, global midranks were fixed before resampling and cluster sufficient statistics were resampled; the system-median analysis provides a complementary system-level check. SYR4's deterministic 20% system holdout was assigned by a stable cryptographic hash, preventing samples from the same system from crossing splits. An ICR holdout was not imposed because the full-window positive cohorts are small; exact finite-grid probabilities and explicit support counts are more informative there.

## Reproducibility

All cohort rules, thresholds, event-type sets, censoring decisions, bootstrap settings, and output checksums are recorded in `analysis_metadata.json`, `validation_summary.json`, and `output_manifest.csv`. Results are descriptive validation of these public monitoring data and should not be generalized beyond the observed systems, periods, locations, and reporting conventions.
