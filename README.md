# HAA9 monitoring analysis

Processed data and reproducible analyses of haloacetic-acid monitoring, sample allocation and supplemental reporting.

## Run

Use Python 3.12 or later from the repository root.

```sh
python -m pip install -r requirements.txt
python -m pytest tests -q
python src/current_monitoring.py --bootstrap 10000
python src/current_screening.py --bootstrap 2000
```

`data/` contains processed observations and reported tables; `src/` contains analysis scripts; `tests/` contains checks. Current reconstruction outputs are in `verification/`; historical outputs remain in `results/` and `wr_results/`.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for execution scope and [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md) for source attribution. Original publisher archives are acquired separately.
