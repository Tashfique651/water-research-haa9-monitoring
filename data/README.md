# Data

Processed CSV and Parquet files retain the environmental identifiers needed for linkage. `reported_tables/` contains machine-readable analytical result tables; these are not substitutes for independently regenerated outputs.

`verification/inputs/` at the repository root contains compact replay inputs, a reconstructed screening-feature table, source hashes and a six-field linkage audit. Current calculations do not overwrite historical outputs.

Original publisher archives are not bundled. Their recorded locations, hashes and attribution are in `data/manifests/` and `THIRD_PARTY_DATA.md`. Reading the raw ICR Access database requires a compatible Access driver; the included processed ICR tables avoid that dependency. The California acquisition script documents the separate large-archive route.
