# California SDWIS4 HAA5/TTHM validation methods

## Scope and provenance

The source is the official California Division of Drinking Water SDWIS4 flat-file archive at `.\data\raw\additional\california\sdwis4\SDWIS4.tab`. Its SHA-256 digest is `9aa519a26257efaaf3a7a91e87ce45917b61cd4a26b7b5eec7802770cc8cb0df`. The primary analysis includes calendar years 2023-2025 only, analyte codes 2456 (HAA5 total) and 2950 (TTHM total), routine samples (`RT`), distribution-system facilities (`DS`), and records reported in `UG/L`.

The archive uses Windows-1252 encoding. A byte-level line-shape audit found 14 rows with a nonstandard tab-field count among more than 7.1 million data rows, but none was an HAA5 or TTHM row. DuckDB therefore used its relaxed structural parser for the archive-wide scan; target-analyte counts were checked against the independent byte-level counts before analysis.

## Record validity and censoring

A record required a parseable sample date, nonempty public-water-system and sampling-point identifiers, and either a nonnegative quantified result or a less-than flag with a nonnegative reporting limit. Quantified values were used as reported. Censored values were represented as intervals from zero to the reporting limit. The primary continuous value used one-half the reporting limit; zero and full-limit substitutions were analyzed separately. These substitutions are numerical sensitivity conventions and are not new measurements.

## Duplicate event-analyte keys

An event-analyte key was defined by analyte, PWS, sampling point, sample date, and clock time. Multiple source rows at a key were collapsed with the median record-level value. The minimum and maximum values at each duplicated key were retained for sensitivity analysis, and analyses excluding discordant duplicate keys were also reported. The audit distinguishes exact duplicate rows from discordant replicate values.

## Temporal summaries

Annual, month-of-year, and calendar-season summaries use all valid analyte-specific events, not only paired samples. Seasons were Winter (December-February), Spring (March-May), Summer (June-August), and Autumn (September-November). Concentration summaries report lower-bound, half-limit, and upper-bound values. Counts at 60 micrograms per litre for HAA5 and 80 micrograms per litre for TTHM are descriptive concentration-level comparisons only. They are not determinations of compliance, which depends on regulatory calculations and context not reconstructed here.

## Paired concordance

The primary paired analysis required exact agreement on PWS, sampling point, sample date, and nonmissing clock time. It contained 37,894 pairs across 2,538 PWS. Primary rank concordance was estimated among pairs in which both totals were quantified, avoiding dependence on a censor-substitution convention; lower-bound, half-limit, and upper-bound analyses used all pairs as sensitivities. A same-date analysis that ignored clock time was retained only as a sensitivity analysis. Concordance was summarized with Spearman rho and Kendall tau-b. The 95% interval for the primary rho used 10,000 percentile bootstrap replicates resampled by PWS, with all observations from each sampled PWS carried together and ranks recomputed in every replicate. PWS-level median concentrations among detected pairs were also analyzed to reduce domination by frequently sampled systems.

## Threshold agreement

The paired values were dichotomized over a prespecified grid of HAA5 thresholds from 20 to 80 micrograms per litre and TTHM thresholds from 40 to 100 micrograms per litre. Results include the four cell counts, overall agreement, high-set Jaccard agreement, the phi coefficient, and both conditional high-concentration probabilities. Because high concentrations are uncommon, overall agreement is reported together with prevalence-sensitive measures and must not be interpreted alone.

## Interpretation limits

This California analysis validates only the reproducibility and external transport of HAA5/TTHM co-occurrence patterns in a separate state monitoring archive. It does not measure HAA9, does not prove that the archive is statistically independent of every other public data product, does not establish causality, and does not estimate health risk. Censor substitutions, unresolved provenance of discordant same-key records, unequal sampling intensity, and the inability to reconstruct regulatory locational running annual averages remain explicit limitations.
