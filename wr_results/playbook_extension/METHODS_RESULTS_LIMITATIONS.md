# Water Research playbook analysis extension

> Legacy analysis note, retained for provenance. Its editorial recommendations are superseded by the 19 September 2026 HAA9 revision in `wr_results/haa9_revision/`. In particular, the historical phrase "four-sample equivalence" is not a formal equivalence conclusion. The paired difference at four samples remains uncertain; five is the first tested integer budget with a positive simultaneous lower bound in the revised comparison family. Recommendations to move figures do not apply to the current main-text layout. Numerical baseline files are preserved unchanged.

## Scope and reproducibility

This additive analysis uses only frozen public data already present in the repository. The official EPA UCMR4 HAA5, HAA6Br, and HAA9 totals remain authoritative. The frozen Peterson et al. FOIA records supply simultaneous individual-species values for BCAA and the exact same-size panel sensitivity. The EPA SYR4 system table supplies a secondary, incomplete population-served sensitivity. All random procedures use seed 20260914. Cluster bootstrap intervals use the public water system as the resampling unit.

## Feasibility decisions

| analysis | status | recommendation | qualification |
| --- | --- | --- | --- |
| A1 interaction scales and inference | implemented | KEEP IN MAIN TEXT | Report additive and multiplicative scales together. Wild sign-flip p-values are secondary and require sign exchangeability. |
| A2 operational round policies | implemented | KEEP WARM-SEASON RESULT IN MAIN; MOVE TOC AND ORACLE TO SI | Warm season is prespecified; TOC has partial, non-hydraulic event-code linkage; HAA9 oracle is nonprospective. |
| A3 mechanism and heterogeneity | implemented | KEEP CONCISE BROMIDE/TOC AND COMPOSITION RESULT IN MAIN; FULL MODELS IN SI | Bromide and TOC models are complete-case associations; composition predictors are descriptive. |
| A4 equivalence and cost-ratio frontier | implemented | KEEP FOUR-SAMPLE EQUIVALENCE IN MAIN; KEEP COST-RATIO FRONTIER AS SCENARIO SENSITIVITY | Uniform physical-sample selection and hypothetical relative costs, not observed dollars. |
| A5 LRAA and OEL analogues plus panel candidates | implemented | KEEP ONE COMPACT MAIN-TEXT PANEL; PUT FULL WINDOWS AND DISCLAIMERS IN SI | Research analogues only; consecutive-quarter support is incomplete; HAA5+BCAA uses frozen species data. |
| A6 population weighting | implemented as sensitivity | KEEP IN SI AND MENTION DIRECTION IN MAIN | EPA SYR4 population coverage is incomplete for UCMR4 systems. |
| A7 map | data implemented | DROP FROM MAIN UNLESS SPACE ALLOWS; USE IN SI | State-level values are supplied for plotting; small state counts require visual suppression or annotation. |
| A8 fixed-B60 cohort threshold sensitivity | implemented | KEEP FIXED-COHORT CURVE IN MAIN ROBUSTNESS FIGURE | Reported beside threshold-specific cohorts to distinguish estimand drift. |
| A9 negative controls | implemented | KEEP NESTED NULL IN SI; USE SAME-SIZE PANEL ENUMERATION AS SI SENSITIVITY | Duplicate-panel null is exact; same-size species enumeration is a sensitivity, not a formal exchangeability test. |
| Highest-temperature round policy | not implemented | DROP | No co-located temperature variable exists; calendar season is used without fabricated climate data. |
| National population-representative inference | not supportable | DROP NATIONAL CLAIM | The available population join covers only a subset of eligible systems. |

## Main findings

The complete-observed-grid estimand retained an additive interaction of 0.2123 (95% bootstrap interval 0.1844 to 0.2424). Interaction depended on scale. The multiplicative interaction ratio was 0.863 (0.830 to 0.898), while the relative excess risk due to interaction was 1.017. The chronological late-condition, early-evaluation estimate was 0.0436 (0.0212 to 0.0660). The wild sign-flip results are reported separately and require sign-exchangeability under the null.

The prespecified May-to-September selector produced HAA9 one-round detection of 0.5630, compared with 0.4191 under uniform round selection. The highest-TOC policy produced 0.5963 among 541 TOC-complete benchmark-positive systems. The TOC result is not hydraulically paired because UCMR4 indicators were joined only by system and event code.

At the physical-sample scale, 4 uniformly selected HAA9 samples achieved conditional capture 0.5788, compared with 0.5774 for the entire observed HAA5 record. At exactly four samples, the paired difference was 0.0014 (whole-system bootstrap interval -0.0304 to 0.0356). The benchmark-positive systems had a mean of 18.40 physical samples. Under the explicitly hypothetical equal-expected-budget calculation, the estimated HAA9-to-HAA5 per-sample cost-ratio crossing was 3.74 (bootstrap interval 3.39 to 4.14). This is a design frontier, not a dollar cost-effectiveness estimate.

The adjusted-population complete-case estimate used 587 benchmark-positive systems and 49,850,261 covered population. Its additive interaction was 0.1847 (0.1022 to 0.2958). Coverage is incomplete, so this result cannot be described as nationally population representative.

Across all complete finite-grid systems, the single-sample-maximum endpoint identified 444 HAA5-positive systems and 769 HAA9-positive systems. The rolling research analogues retained 2683 systems for the three-quarter operational-evaluation formula and 2487 systems for four consecutive quarters. These are not compliance determinations.

The species audit mapped 63,432 complete nine-species events. Exact enumeration of 126 five-species panels placed regulatory HAA5 at descending interaction rank 98; this is a composition sensitivity, not a species-exchangeability test. The duplicate-panel nested-null control returned additive interaction 0.000000.

## Interpretation limits

1. The complete-grid cohort is conditioned on an observed HAA9 maximum above 60 ug/L. It estimates conditional detection, not national prevalence or compliance.
2. UCMR4 event codes are administrative groups and can span more than one collection date. Warm-season selection uses the median collection date within each event-code group.
3. TOC and bromide are source-side indicators joined by PWSID and event code. They are not asserted to be hydraulically paired with a distribution-system HAA sample.
4. Population weighting covers only systems found in the SYR4 water-system table and is a complete-case sensitivity.
5. LRAA and operational-evaluation calculations require consecutive calendar quarters at a site, use quarterly arithmetic means, and are research analogues only.
6. Same-size panel enumeration preserves measured species but does not make species exchangeable. It supplies a design sensitivity rather than a formal causal or randomization null.
7. The highest-HAA9 round is an outcome-informed oracle and cannot be implemented prospectively.
