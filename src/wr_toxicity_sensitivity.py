"""In-vitro cytotoxicity-weighted HAA composition sensitivity.

This is a compositional sensitivity analysis. It is not a human-health risk,
exposure, regulatory-compliance or causal analysis. Two coefficient tables
distributed with Peterson et al.'s public supporting archive are kept separate
because their LC50 values are not identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPECIES_PATH = ROOT / "results" / "additional" / "ucmr4_species_event_panel.parquet"
R_COEFFICIENTS = (
    ROOT
    / "data"
    / "raw"
    / "additional"
    / "peterson_haa"
    / "r"
    / "Supporting Files"
    / "DBP Index.csv"
)
DIMENSION_COEFFICIENTS = (
    ROOT
    / "data"
    / "raw"
    / "additional"
    / "peterson_haa"
    / "python"
    / "data"
    / "haa9_dimension_table.csv"
)
OUT = ROOT / "wr_results" / "toxicity_sensitivity"
SPECIES = ["MCAA", "MBAA", "DCAA", "DBAA", "TCAA", "BCAA", "BDCAA", "CDBAA", "TBAA"]
HAA5 = {"MCAA", "MBAA", "DCAA", "DBAA", "TCAA"}
EXTRA4 = set(SPECIES) - HAA5
BOOTSTRAPS = 10_000
SEED = 20260914


def save_table(frame: pd.DataFrame, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"{stem}.csv", index=False)
    frame.to_parquet(OUT / f"{stem}.parquet", index=False)


def load_coefficients() -> pd.DataFrame:
    r_table = pd.read_csv(R_COEFFICIENTS)
    r_table = r_table.loc[r_table["class"].eq("HAA")].copy()
    r_table["species"] = r_table["DBP"].replace(
        {"CAA": "MCAA", "BAA": "MBAA", "DBCAA": "CDBAA"}
    )
    r_table = r_table.rename(columns={"MW": "molecular_weight_g_mol", "LC50": "lc50_mol_l"})
    r_table["coefficient_source"] = "Peterson_public_R_supporting_table"

    dimension = pd.read_csv(DIMENSION_COEFFICIENTS)
    dimension = dimension.rename(
        columns={
            "Abbreviation": "species",
            "MolecularWeight(g/mol)": "molecular_weight_g_mol",
            "LC50(mol/L)": "lc50_mol_l",
        }
    )
    dimension["coefficient_source"] = "Peterson_public_dimension_table"
    columns = ["coefficient_source", "species", "molecular_weight_g_mol", "lc50_mol_l"]
    combined = pd.concat([r_table[columns], dimension[columns]], ignore_index=True)
    combined["panel"] = np.where(combined["species"].isin(HAA5), "HAA5", "additional_HAA9_four")
    combined["inverse_lc50_l_mol"] = 1.0 / combined["lc50_mol_l"]
    if set(combined.groupby("coefficient_source")["species"].nunique()) != {9}:
        raise ValueError("Each coefficient source must cover exactly nine HAA species")
    return combined.sort_values(["coefficient_source", "panel", "species"]).reset_index(drop=True)


def weighted_events(species: pd.DataFrame, coefficients: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        "PWSID",
        "SampleID",
        "SampleEventCode",
        "CollectionDate",
        "State",
        "FacilityWaterType",
        "HAA5_official_ug_L",
        "HAA9_official_ug_L",
        "Extra4_fraction_official",
    ]
    output: list[pd.DataFrame] = []
    for source, coefficient_group in coefficients.groupby("coefficient_source"):
        coefficient_group = coefficient_group.set_index("species")
        for substitution in ["lower", "mid", "upper"]:
            values = species[id_columns].copy()
            total = np.zeros(len(species), dtype=float)
            haa5 = np.zeros(len(species), dtype=float)
            extra = np.zeros(len(species), dtype=float)
            for analyte in SPECIES:
                concentration_ug_l = species[f"{analyte}_{substitution}_ug_L"].to_numpy(float)
                molecular_weight = float(coefficient_group.loc[analyte, "molecular_weight_g_mol"])
                lc50 = float(coefficient_group.loc[analyte, "lc50_mol_l"])
                toxicity_units = concentration_ug_l * 1e-6 / molecular_weight / lc50
                total += toxicity_units
                if analyte in HAA5:
                    haa5 += toxicity_units
                else:
                    extra += toxicity_units
            values["coefficient_source"] = source
            values["nondetect_substitution"] = substitution
            values["HAA5_in_vitro_cytotoxicity_units"] = haa5
            values["additional_four_in_vitro_cytotoxicity_units"] = extra
            values["HAA9_in_vitro_cytotoxicity_units"] = total
            values["additional_four_weighted_fraction"] = np.divide(
                extra,
                total,
                out=np.full(len(total), np.nan),
                where=total > 0,
            )
            output.append(values)
    return pd.concat(output, ignore_index=True)


def cluster_bootstrap_system_means(frame: pd.DataFrame, value: str, seed: int) -> tuple[float, float, float]:
    system_values = frame.groupby("PWSID")[value].mean().dropna().to_numpy(float)
    point = float(system_values.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAPS, dtype=float)
    for start in range(0, BOOTSTRAPS, 250):
        stop = min(start + 250, BOOTSTRAPS)
        idx = rng.integers(0, len(system_values), size=(stop - start, len(system_values)))
        draws[start:stop] = system_values[idx].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return point, float(lower), float(upper)


def summarize(weighted: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, ((source, substitution), group) in enumerate(
        weighted.groupby(["coefficient_source", "nondetect_substitution"], sort=True)
    ):
        fraction = group["additional_four_weighted_fraction"].dropna()
        mean, lower, upper = cluster_bootstrap_system_means(
            group, "additional_four_weighted_fraction", SEED + index
        )
        rows.append(
            {
                "coefficient_source": source,
                "nondetect_substitution": substitution,
                "events": int(len(group)),
                "systems": int(group["PWSID"].nunique()),
                "system_equal_mean_additional_four_weighted_fraction": mean,
                "cluster_bootstrap_ci_lower": lower,
                "cluster_bootstrap_ci_upper": upper,
                "event_median_additional_four_weighted_fraction": float(fraction.median()),
                "event_q25_additional_four_weighted_fraction": float(fraction.quantile(0.25)),
                "event_q75_additional_four_weighted_fraction": float(fraction.quantile(0.75)),
                "event_mean_mass_fraction_additional_four": float(group["Extra4_fraction_official"].mean()),
                "interpretation": "in-vitro CHO-cell cytotoxicity-weighted composition sensitivity; not health risk",
            }
        )
    return pd.DataFrame(rows)


def coefficient_comparison(weighted: pd.DataFrame) -> pd.DataFrame:
    midpoint = weighted.loc[weighted["nondetect_substitution"].eq("mid")]
    event_key = ["PWSID", "SampleID"]
    wide = midpoint.pivot_table(
        index=event_key,
        columns="coefficient_source",
        values=["HAA9_in_vitro_cytotoxicity_units", "additional_four_weighted_fraction"],
        aggfunc="first",
    )
    sources = midpoint["coefficient_source"].drop_duplicates().tolist()
    if len(sources) != 2:
        raise ValueError("The coefficient comparison requires exactly two sources")
    rows: list[dict[str, object]] = []
    for metric in ["HAA9_in_vitro_cytotoxicity_units", "additional_four_weighted_fraction"]:
        paired = wide[metric][sources].dropna()
        rows.append(
            {
                "metric": metric,
                "coefficient_source_1": sources[0],
                "coefficient_source_2": sources[1],
                "paired_events": int(len(paired)),
                "spearman_rank_correlation": float(paired[sources[0]].corr(paired[sources[1]], method="spearman")),
                "median_source_1": float(paired[sources[0]].median()),
                "median_source_2": float(paired[sources[1]].median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    species = pd.read_parquet(SPECIES_PATH)
    coefficients = load_coefficients()
    weighted = weighted_events(species, coefficients)
    summary = summarize(weighted)
    comparison = coefficient_comparison(weighted)
    save_table(coefficients, "coefficient_sets")
    save_table(weighted, "event_weighted_composition")
    save_table(summary, "weighted_composition_summary")
    save_table(comparison, "coefficient_set_comparison")
    methods = """# In-vitro cytotoxicity-weighted composition sensitivity

Species concentrations were converted from micrograms per litre to molar concentration and divided by Chinese hamster ovary cell LC50 values. HAA5 and the four additional HAA9 species were summed separately. Nondetects were evaluated at zero, one-half the reported minimum reporting level, and the full reporting level. Two public Peterson supporting-data coefficient tables were analysed independently because several LC50 values differ, particularly for tribromoacetic acid. The R-supporting table is the prespecified coefficient source and the dimension table is a coefficient sensitivity. All intervals resample public water systems and preserve records within systems.

The resulting dimensionless index is used only to test whether mass fraction understates composition under one in-vitro cytotoxicity weighting scheme. It is not an estimate of human dose, toxicity in vivo, health risk, disease burden, regulatory compliance, or benefit from a monitoring change.
"""
    (OUT / "METHODS_LIMITATIONS.md").write_text(methods, encoding="utf-8")
    payload = {
        "events": int(species.shape[0]),
        "systems": int(species["PWSID"].nunique()),
        "coefficient_sources": coefficients["coefficient_source"].unique().tolist(),
        "summary": summary.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
        "bootstrap_replicates": BOOTSTRAPS,
        "seed": SEED,
    }
    (OUT / "toxicity_sensitivity_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
