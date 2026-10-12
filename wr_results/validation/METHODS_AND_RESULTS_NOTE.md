# Water Research validation extensions

## Scope

This analysis is isolated from the frozen core outputs. It creates only files under `wr_results/validation`. All thresholds are analytical signal benchmarks, not compliance determinations.

## Exact-site SYR4 pairing

HAA5 and TTHM records were first linked on exact PWSID and SAMPLE_ID, as in the prior analysis. The strict-site primary cohort additionally required identical collection date, WATER_FACILITY_ID and SAMPLING_POINT_ID. The same-date all-pairs cohort was retained as a sensitivity. This resolved 124 same-date links with nonmatching sampling points by excluding them from the strict-site primary cohort. The mismatches were retained in a row-level audit, not silently relabelled.

Detected HAA5-TTHM association used Spearman correlation. Threshold agreement used only pairs whose censoring intervals were classifiable at 60 and 80 ug/L. Ninety-five percent intervals used 10,000 whole-system resamples. The strict-site cohort contained 527,172 pairs in 26,535 systems; its detected-pair rho was 0.618474 and threshold agreement was 0.932222. The all-pairs sensitivity contained 527,296 pairs, with rho 0.618521 and agreement 0.932224.

## UCMR4-SYR4 overlap

Complete UCMR4 HAA triplets were joined to all valid SYR4 HAA5 records on exact PWSID and collection date. Concentration differences were evaluated with one-to-one minimum-cost matching at 0, 0.001, 0.01, 0.05, 0.1, 0.5 and 1.0 ug/L. The primary candidate definition used <=0.05 ug/L, which accommodates rounding to the nearest 0.1 ug/L without asserting record identity. Same-day concurrence and concentration similarity are candidate overlap, not proof of identical provenance.

There were 11,837 UCMR4 events in 1,000 systems with at least one SYR4 HAA5 record on the same date. At the primary tolerance, one-to-one matching identified 1,846 candidate event pairs in 388 systems. Robustness analyses removed matched event identifiers, all concurrent dates, matched systems, and all systems with a concurrent date. The exact paired-association and UCMR4 factorial results for each exclusion are in the output tables.

The original UCMR4 interaction was 0.212289. It was 0.213773 after removing primary candidate events, 0.203162 after removing their systems, 0.190776 after removing every same-date UCMR4 event, and 0.202699 after removing every system with any same-date record.

## Full-record SYR4 quarter capture

Quarter-capture curves were rebuilt separately for HAA5 and TTHM from every valid single-analyte record keyed by the publisher-unique SIX_YEAR_ID, including records without SAMPLE_ID. Detected values were point observations. Nondetects were bounded from zero to a positive unit-normalized detection limit; unknown or unsupported bounds were indeterminate. A quarter was above threshold if any record was above, below if all records were definitely below, and otherwise indeterminate. System-years required at least three observed, classifiable quarters and at least one above-threshold quarter. Capture for k of N quarters was enumerated exactly without replacement and averaged with equal weight per system-year. The prior paired-only design was retained in both same-date and exact-site forms.

The all-record inputs contained 1,005,235 valid HAA5 records and 1,089,557 valid TTHM records. Curves and target crossings are reported without replacing the full-record analysis with the paired subset.

For four-quarter, reference-positive system-years, one randomly selected quarter captured 0.365296 of HAA5 signals and 0.402079 of TTHM signals. Two quarters increased those probabilities to 0.628353 and 0.658202, respectively.

## ICR -999 recovery

The ICR guide states that -999 denotes a result below the minimum reporting level, that below-MRL components are replaced by zero only when calculating totals, and that HAA5 or HAA9 equals -999 when all named components are below their MRLs. MRLs were read directly from TUXANLYT in the supplied Access database. The HAA5 component MRLs sum to 6.0 ug/L and the HAA9 component MRLs sum to 14.0 ug/L, both below 60 ug/L. Every recovered total also had all required components coded -999.

For threshold classification only, verified -999 totals were coded with a binary helper value that cannot exceed 60 ug/L. This is not a concentration imputation and is never used in concentration summaries. The analysis recovered 1,837 HAA5 and 492 HAA9 event states. Original exclusion and MRL-bounded recovery results are supplied side by side for temporal support, capture and the matched factorial.

In the primary ICR scope, complete six-quarter support increased from 151 to 167 systems for HAA5 and from 8 to 11 for HAA9. The complete-six-quarter matched factorial increased from 8 to 11 eligible systems and from 2 to 3 HAA9-reference-positive systems.

## Reproducibility

All tables are supplied as CSV and Parquet. The JSON summary records software, thresholds, rules and exact headline counts. The manifest records SHA-256 hashes for every generated output. Random seeds are fixed; finite-grid probabilities and one-to-one overlap assignments are deterministic.
