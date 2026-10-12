# Reproduction routes

## Current calculations

Run commands from the repository root after installing `requirements.txt`.

| Command | Inputs and outputs |
| --- | --- |
| `python -m pytest tests -q` | Historical scientific tests and current reconstruction checks. |
| `python src/current_monitoring.py --bootstrap 10000` | Compact paired observations; all 16 budgets, later-site augmentation and 17 benchmarks. Outputs: `verification/results/`. |
| `python src/current_screening.py --bootstrap 2000` | Released early predictors; weighted training, fixed folds, refitting and paired resampling. Outputs: `verification/results/screening/`. |
| `python src/check_screening_inputs.py` | Six-field reconstruction of all early scalar and species predictors from retained Parquet inputs. Outputs: `verification/inputs/`. |

The paired input contains 63,432 complete chemical events; replay retains 63,105 events in 4,904 systems. The primary reference contains 769 systems. The date-separated cohort contains 2,636 systems and 420 later positives. All 24,211 early samples and all scalar/species predictors match the retained source-linked features exactly.

The screening run completed 2,000 refitted resamples. All 30 rows in the reported accuracy and paired-gain tables agree at their displayed precision. Exact monitoring point estimates agree with the reported results. New Monte Carlo intervals are retained separately; small endpoint differences from earlier resampling runs do not change the comparison family or interpretation. The 17-benchmark targeting band includes zero at the 60 microgram/L endpoint.

## Retained analyses

`results/`, `wr_results/` and `wr_figures/source_data/` preserve historical outputs and source tables. Their filenames are historical, not a current figure-numbering scheme. `data/reported_tables/` contains 54 machine-readable reported analytical tables, each identified by an index and hash. Table transcription is not numerical regeneration.

The supplied historical release does not contain the original later-extension scripts for within-system bromination, matched incubation, confounding, signal timing, margin and sharp-censoring analyses, jurisdiction sensitivity, and prospective scenarios. Their reported tables and methods remain distinct from the executed monitoring and screening reconstructions. The current release does not claim independent regeneration of those extensions or of every historical model.

## Recorded execution

`verification/results/test_run.json` records 83 passed tests: 64 historical scientific tests and 19 current reconstruction checks. The recorded Linux run used the included flat-Parquet I/O adapter because PyArrow was unavailable. The adapter was checked against five independent CSV mirrors, with no mismatched columns. It supports the specific flat files in this archive, not arbitrary Parquet datasets. The scientific tests were unchanged.

The ordinary cross-platform route uses PyArrow from `requirements.txt`. To repeat the recorded fallback on Linux, install `verification/tools/requirements-fallback.txt`, provide `libsnappy.so.1` and `libzstd.so.1`, and run `python verification/tools/run_checked_tests.py`. A fresh dependency installation and CI execution were not performed here.

## Source integrity

Existing retained data files are unchanged. Compact inputs and reconstructed features have separate provenance records. `FILE_MANIFEST.csv` identifies packaged bytes. Raw publisher archives are acquired separately using the locations and recorded hashes in `data/manifests/` and `THIRD_PARTY_DATA.md`; they were not downloaded again during this verification. Raw ICR Access extraction has a separate driver dependency.
