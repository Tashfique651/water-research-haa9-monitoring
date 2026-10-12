"""Frozen retrospective matched-budget, season and shared-oracle audit.

Protocol: wr_results/haa9_revision/budget_retrospective_protocol.md.
All finite-grid selection probabilities are exact. Bootstrap intervals describe
between-PWS resampling variation, not uncertainty about the observed arithmetic.
"""
from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "wr_results" / "haa9_revision"
SEED = 20260919
REPS = 10000
THRESHOLD = 60.0


def capture(n: int, m: int, k: int) -> float:
    """Capture under simple random sampling without replacement, saturated at n."""
    if not (0 <= m <= n) or k < 0:
        raise ValueError("Require 0 <= positives <= available and nonnegative budget")
    k = min(k, n)
    if not k or not m:
        return 0.0
    return 1.0 if n - m < k else 1.0 - math.comb(n - m, k) / math.comb(n, k)


def expected_distinct(counts: list[int] | np.ndarray, k: int) -> float:
    counts = np.asarray(counts, dtype=int)
    return sum(capture(int(counts.sum()), int(m), k) for m in counts)


def warm_first_capture(warm: np.ndarray, cold: np.ndarray, k: int) -> float:
    """Select warm first, supplement cold to preserve a matched physical budget."""
    k = min(k, len(warm) + len(cold))
    a = min(k, len(warm))
    b = k - a
    return 1 - (1 - capture(len(warm), int(np.sum(warm)), a)) * (
        1 - capture(len(cold), int(np.sum(cold)), b)
    )


def shared_oracle(group: pd.DataFrame) -> tuple[float, float, int]:
    selected = group.loc[group.HAA9.eq(group.HAA9.max())]
    return float(selected.HAA5.gt(THRESHOLD).mean()), float(selected.HAA9.gt(THRESHOLD).mean()), len(selected)


def interaction_from_units(haa5: np.ndarray, haa9: np.ndarray) -> float:
    a = np.asarray(haa5, bool)
    b = np.asarray(haa9, bool)
    if len(a) != len(b) or np.any(a & ~b):
        raise ValueError("Require matched nested binary observations")
    return float(b.any()) - b.mean() - float(a.any()) + a.mean()


def bootstrap(values: np.ndarray, seed: int = SEED, reps: int = REPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    n, w = values.shape
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, w))
    for start in range(0, reps, 100):
        take = min(100, reps - start)
        idx = rng.integers(0, n, size=(take, n))
        draws[start:start + take] = values[idx].mean(axis=1)
    return values.mean(axis=0), np.quantile(draws, [0.025, 0.975], axis=0), draws


def save(frame: pd.DataFrame, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"{stem}.csv", index=False, lineterminator="\n")
    frame.to_parquet(OUT / f"{stem}.parquet", index=False)


def load_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_parquet(ROOT / "data/processed/ucmr4/paired_haa_events.parquet")
    events = events.loc[events.finite_grid_system_eligible.fillna(False)].copy()
    events.CollectionDate = pd.to_datetime(events.CollectionDate, errors="raise")
    full = pd.read_parquet(ROOT / "data/processed/ucmr4/system_full_grid_panel_maxima.parquet")
    assert len(events) == 63105 and events.PWSID.nunique() == 4904
    assert not events.duplicated(["PWSID", "SampleID"]).any(), "Physical sample key is not unique"
    assert events[["CollectionDate", "SampleID", "site_id", "SampleEventCode"]].notna().all().all()
    assert not (events.HAA5 > events.HAA9 + 1e-9).any()
    assert full.HAA9_full_max.gt(THRESHOLD).sum() == 769
    return events, full


def budget_curves(events: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    positive_ids = set(full.loc[full.HAA9_full_max.gt(THRESHOLD), "PWSID"])
    rows = []
    for pwsid, group in events.loc[events.PWSID.isin(positive_ids)].groupby("PWSID", sort=True):
        n = len(group)
        m5, m9 = int(group.HAA5.gt(THRESHOLD).sum()), int(group.HAA9.gt(THRESHOLD).sum())
        date_counts = group.groupby("CollectionDate").size().to_numpy()
        site_counts = group.groupby("site_id").size().to_numpy()
        for k in range(1, 17):
            h5, h9 = capture(n, m5, k), capture(n, m9, k)
            rows.append(dict(PWSID=pwsid, requested_samples=k, available_samples=n,
                             realized_samples=min(k, n), strictly_below_requested=n < k,
                             full_record_selected=n <= k, HAA5_capture=h5, HAA9_capture=h9,
                             full_HAA5_capture=float(m5 > 0), same_k_difference=h9-h5,
                             vs_full_HAA5_difference=h9-float(m5 > 0),
                             expected_distinct_dates=expected_distinct(date_counts, k),
                             expected_distinct_sites=expected_distinct(site_counts, k),
                             available_dates=len(date_counts), available_sites=len(site_counts)))
    detail = pd.DataFrame(rows)
    save(detail, "budget_systems")
    fields = ["HAA5_capture", "HAA9_capture", "same_k_difference", "vs_full_HAA5_difference"]
    matrices = [detail.pivot(index="PWSID", columns="requested_samples", values=f).to_numpy() for f in fields]
    values = np.column_stack(matrices)
    means, ci, draws = bootstrap(values)
    contrasts = values[:, 32:64]
    se = contrasts.std(axis=0, ddof=1) / np.sqrt(len(contrasts))
    centered = draws[:, 32:64] - means[32:64]
    standardized = np.divide(centered, se, out=np.zeros_like(centered), where=se > 0)
    critical = float(np.quantile(np.max(np.abs(standardized), axis=1), .95))
    summary = []
    for k in range(1, 17):
        selected = detail.loc[detail.requested_samples.eq(k)]
        row = dict(requested_samples=k, reference_positive_systems=769, bootstrap_reps=REPS,
                   bootstrap_seed=SEED, simultaneous_family_size=32, simultaneous_critical_value=critical,
                   full_HAA5_capture=float(selected.full_HAA5_capture.mean()),
                   mean_available_samples=float(selected.available_samples.mean()),
                   median_available_samples=float(selected.available_samples.median()),
                   mean_realized_samples=float(selected.realized_samples.mean()),
                   systems_strictly_below_requested=int(selected.strictly_below_requested.sum()),
                   systems_full_record_selected=int(selected.full_record_selected.sum()),
                   expected_distinct_dates=float(selected.expected_distinct_dates.mean()),
                   expected_distinct_sites=float(selected.expected_distinct_sites.mean()),
                   cohort="complete_grid_HAA9_full_max_gt60", weighting="equal_PWS")
        for j, field in enumerate(fields):
            col = j * 16 + k - 1
            row[field] = float(means[col])
            row[f"{field}_ci_lower"] = float(ci[0, col])
            row[f"{field}_ci_upper"] = float(ci[1, col])
            if j >= 2:
                row[f"{field}_simultaneous_lower"] = float(means[col] - critical * se[col-32])
                row[f"{field}_simultaneous_upper"] = float(means[col] + critical * se[col-32])
        summary.append(row)
    frame = pd.DataFrame(summary)
    save(frame, "budget_curve")
    extra = matrices[1][:, 3] - matrices[0][:, 11]
    point, interval, _ = bootstrap(extra, SEED + 1)
    save(pd.DataFrame([dict(contrast="HAA9_k4_minus_HAA5_k12", estimate=point[0],
        ci_lower=interval[0, 0], ci_upper=interval[1, 0], systems=len(extra),
        status="exploratory_previously_reported_candidate_outside_32_contrast_family")]), "budget_cross_k_exploratory")
    print("PHYSICAL BUDGET AUDIT", frame.loc[frame.requested_samples.isin([4,5,8,12]),
        ["requested_samples", "HAA5_capture", "HAA9_capture", "same_k_difference", "same_k_difference_simultaneous_lower",
         "vs_full_HAA5_difference", "vs_full_HAA5_difference_ci_lower", "vs_full_HAA5_difference_ci_upper",
         "vs_full_HAA5_difference_simultaneous_lower", "vs_full_HAA5_difference_simultaneous_upper"]].to_string(index=False), flush=True)
    return frame


def unit_curves(events: pd.DataFrame, full: pd.DataFrame) -> None:
    ids = set(full.loc[full.HAA9_full_max.gt(THRESHOLD), "PWSID"])
    rows = []
    for unit, key in [("administrative_event_code_group", "SampleEventCode"), ("distribution_site", "site_id")]:
        units = events.loc[events.PWSID.isin(ids)].groupby(["PWSID", key]).agg(
            HAA5=("HAA5", "max"), HAA9=("HAA9", "max"), physical_samples=("SampleID", "size")).reset_index()
        for pwsid, group in units.groupby("PWSID", sort=True):
            n = len(group)
            for k in range(1, 5):
                h5, h9 = (capture(n, int(group[p].gt(THRESHOLD).sum()), k) for p in ["HAA5", "HAA9"])
                rows.append(dict(PWSID=pwsid, unit=unit, requested_units=k, available_units=n,
                    realized_units=min(k,n), strictly_below_requested=n<k, full_record_selected=n<=k,
                    expected_physical_samples=min(k,n)/n*group.physical_samples.sum(),
                    HAA5_capture=h5, HAA9_capture=h9, same_k_difference=h9-h5,
                    vs_full_HAA5_difference=h9-float(group.HAA5.gt(THRESHOLD).any())))
    detail = pd.DataFrame(rows)
    save(detail, "support_budget_systems")
    results = []
    for i, ((unit, k), g) in enumerate(detail.groupby(["unit", "requested_units"], sort=True)):
        cols = ["HAA5_capture", "HAA9_capture", "same_k_difference", "vs_full_HAA5_difference"]
        means, cis, _ = bootstrap(g[cols].to_numpy(), SEED + 100 + i)
        row = dict(unit=unit, requested_units=k, systems=len(g),
            mean_realized_units=float(g.realized_units.mean()),
            mean_expected_physical_samples=float(g.expected_physical_samples.mean()),
            systems_strictly_below_requested=int(g.strictly_below_requested.sum()),
            systems_full_record_selected=int(g.full_record_selected.sum()),
            inference="secondary_pointwise_95pct_whole_PWS_bootstrap")
        for j, col in enumerate(cols):
            row.update({col:means[j], f"{col}_ci_lower":cis[0,j], f"{col}_ci_upper":cis[1,j]})
        results.append(row)
    save(pd.DataFrame(results), "support_budget_curve")


def season_audit(events: pd.DataFrame, full: pd.DataFrame) -> None:
    lookup = full.set_index("PWSID")
    rows = []
    for unit, key in [("administrative_event_code_group", "SampleEventCode"), ("collection_date", "CollectionDate"), ("physical_sample", "SampleID")]:
        units = events.groupby(["PWSID", key], sort=True).agg(HAA5=("HAA5", "max"), HAA9=("HAA9", "max"),
            representative_date=("CollectionDate", "median"), physical_samples=("SampleID", "size")).reset_index()
        units["warm"] = units.representative_date.dt.month.between(5,9)
        for pwsid, group in units.groupby("PWSID", sort=True):
            warm, cold = group.loc[group.warm], group.loc[~group.warm]
            for k in range(1, 5 if unit == "physical_sample" else 2):
                cells = []
                for panel in ["HAA5", "HAA9"]:
                    cells.append(capture(len(group), int(group[panel].gt(THRESHOLD).sum()), k))
                for panel in ["HAA5", "HAA9"]:
                    cells.append(warm_first_capture(warm[panel].gt(THRESHOLD).to_numpy(), cold[panel].gt(THRESHOLD).to_numpy(), k))
                p5u,p9u,p5w,p9w = cells
                chosen = warm if len(warm) else group
                samples_warm = min(k,len(group)) if unit == "physical_sample" else float(chosen.physical_samples.mean())
                samples_uniform = min(k,len(group)) if unit == "physical_sample" else float(group.physical_samples.mean())
                rows.append(dict(PWSID=pwsid, unit=unit, requested_units=k, available_units=len(group),
                    available_warm_units=len(warm), fallback_no_warm_units=len(warm)==0,
                    supplemental_cold_units=max(0,min(k,len(group))-len(warm)),
                    HAA5_uniform=p5u,HAA9_uniform=p9u,HAA5_warm=p5w,HAA9_warm=p9w,
                    HAA5_season_gain=p5w-p5u,HAA9_season_gain=p9w-p9u,
                    panel_gain_uniform=p9u-p5u,panel_gain_warm=p9w-p5w,
                    panel_by_season_interaction=(p9w-p5w)-(p9u-p5u),
                    expected_physical_samples_uniform=samples_uniform,expected_physical_samples_warm=samples_warm,
                    reference_positive=bool(lookup.loc[pwsid,"HAA9_full_max"]>THRESHOLD),
                    source_profile=lookup.loc[pwsid,"facility_water_type_profile"]))
    detail = pd.DataFrame(rows)
    save(detail, "season_systems")
    cols=["HAA5_uniform","HAA9_uniform","HAA5_warm","HAA9_warm","HAA5_season_gain","HAA9_season_gain",
          "panel_gain_uniform","panel_gain_warm","panel_by_season_interaction"]
    summaries=[]
    for i, ((unit,k), g) in enumerate(detail.groupby(["unit","requested_units"], sort=True)):
        for cohort in ["reference_positive", "all_eligible"]:
            source=g.loc[g.reference_positive] if cohort=="reference_positive" else g
            point, ci, _=bootstrap(source[cols].to_numpy(),SEED+200+i*10+(cohort=="all_eligible"))
            row=dict(unit=unit,requested_units=k,cohort=cohort,systems=len(source),
                systems_fallback_no_warm=int(source.fallback_no_warm_units.sum()),
                systems_supplemented_cold=int(source.supplemental_cold_units.gt(0).sum()),
                expected_physical_samples_uniform=float(source.expected_physical_samples_uniform.mean()),
                expected_physical_samples_warm=float(source.expected_physical_samples_warm.mean()),
                inference="secondary_pointwise_95pct_whole_PWS_bootstrap")
            for j,col in enumerate(cols):row.update({col:point[j],f"{col}_ci_lower":ci[0,j],f"{col}_ci_upper":ci[1,j]})
            summaries.append(row)
    result=pd.DataFrame(summaries)
    save(result,"season_summary")
    strata=[]
    for (unit,profile),g in detail.loc[detail.reference_positive & detail.requested_units.eq(1)].groupby(["unit","source_profile"],sort=True):
        point,ci,_=bootstrap(g[cols].to_numpy(),SEED+400+len(strata))
        row=dict(unit=unit,source_profile=profile,systems=len(g),status="descriptive_stratum_secondary")
        for j,col in enumerate(cols):row.update({col:point[j],f"{col}_ci_lower":ci[0,j],f"{col}_ci_upper":ci[1,j]})
        strata.append(row)
    save(pd.DataFrame(strata),"season_source_strata")
    print("SEASON AUDIT",result.loc[result.cohort.eq("reference_positive"),["unit","requested_units","panel_by_season_interaction","panel_by_season_interaction_ci_lower","panel_by_season_interaction_ci_upper"]].to_string(index=False),flush=True)


def oracle_and_interaction(events: pd.DataFrame, full: pd.DataFrame) -> None:
    units=events.groupby(["PWSID","SampleEventCode"],sort=True).agg(HAA5=("HAA5","max"),HAA9=("HAA9","max")).reset_index()
    lookup=full.set_index("PWSID")
    rows=[]
    for pwsid,g in units.groupby("PWSID",sort=True):
        a,b,ties=shared_oracle(g)
        y00,y10=float(g.HAA5.gt(THRESHOLD).mean()),float(g.HAA9.gt(THRESHOLD).mean())
        y01,y11=float(g.HAA5.gt(THRESHOLD).any()),float(g.HAA9.gt(THRESHOLD).any())
        rows.append(dict(PWSID=pwsid,y00=y00,y10=y10,y01=y01,y11=y11,
            interaction=y11-y10-y01+y00,oracle_HAA5=a,oracle_HAA9=b,oracle_tied_candidates=ties,
            oracle_HAA5_minus_full_HAA5=a-y01,oracle_interaction=y11-b-y01+a,
            source_profile=lookup.loc[pwsid,"facility_water_type_profile"]))
    allrows=pd.DataFrame(rows)
    save(allrows,"oracle_interaction_systems")
    archived=pd.read_csv(ROOT/"wr_results/playbook_extension/round_policy_systems.csv",dtype={"PWSID":str})
    archived=archived.loc[archived.policy.eq("best_observed_HAA9_round_oracle"),["PWSID","y00","y10","y01","y11"]]
    audit=allrows.loc[allrows.y11.eq(1)].merge(archived,on="PWSID",suffixes=("","_archived"),validate="one_to_one")
    audit["oracle_HAA5_error_vs_archive"]=audit.oracle_HAA5-audit.y00_archived
    audit["oracle_HAA9_error_vs_archive"]=audit.oracle_HAA9-audit.y10_archived
    save(audit,"oracle_archive_comparison")
    results=[]
    cols=["y00","y10","y01","y11","interaction","oracle_HAA5","oracle_HAA9","oracle_interaction"]
    for i,cohort in enumerate(["reference_positive","all_eligible"]):
        source=allrows.loc[allrows.y11.eq(1)] if i==0 else allrows
        point,ci,_=bootstrap(source[cols].to_numpy(),SEED+600+i)
        for j,col in enumerate(cols):results.append(dict(cohort=cohort,estimand=col,systems=len(source),estimate=point[j],ci_lower=ci[0,j],ci_upper=ci[1,j]))
    save(pd.DataFrame(results),"oracle_interaction_summary")
    diagnostics=dict(reference_positive_systems=len(audit),
        archive_HAA5_mismatches=int(audit.oracle_HAA5_error_vs_archive.abs().gt(1e-10).sum()),
        archive_HAA9_mismatches=int(audit.oracle_HAA9_error_vs_archive.abs().gt(1e-10).sum()),
        shared_HAA9_oracle_misses_full_HAA5_signal=int(audit.oracle_HAA5_minus_full_HAA5.lt(-1e-10).sum()),
        HAA9_oracle_tie_systems=int(audit.oracle_tied_candidates.gt(1).sum()),
        interpretation="The shared HAA9-maximizing selector is empirically audited; equality with full HAA5 is not an algebraic identity.")
    (OUT/"oracle_diagnostics.json").write_text(json.dumps(diagnostics,indent=2)+"\n",encoding="utf-8",newline="\n")
    print("ORACLE AUDIT",json.dumps(diagnostics),flush=True)


def direction_weighting() -> None:
    path=ROOT/"wr_results/core/disjoint_grid_system_directions.csv"
    detail=pd.read_csv(path,dtype={"PWSID":str})
    cols=["y00","y10","y01","y11","interaction"]
    sums=detail.groupby("PWSID",sort=True)[cols].sum()
    counts=detail.groupby("PWSID",sort=True).size().reindex(sums.index).to_numpy()
    per_pws=sums.to_numpy()/counts[:,None]
    rng=np.random.default_rng(SEED+700)
    draws_equal=np.empty((REPS,len(cols)))
    draws_direction=np.empty_like(draws_equal)
    for start in range(0,REPS,100):
        take=min(100,REPS-start)
        idx=rng.integers(0,len(sums),size=(take,len(sums)))
        draws_equal[start:start+take]=per_pws[idx].mean(axis=1)
        draws_direction[start:start+take]=sums.to_numpy()[idx].sum(axis=1)/counts[idx].sum(axis=1)[:,None]
    rows=[]
    for weighting,point,draws in [("equal_PWS",per_pws.mean(axis=0),draws_equal),("equal_direction_row",detail[cols].mean().to_numpy(),draws_direction)]:
        cis=np.quantile(draws,[.025,.975],axis=0)
        for j,col in enumerate(cols):rows.append(dict(weighting=weighting,estimand=col,estimate=point[j],ci_lower=cis[0,j],ci_upper=cis[1,j],direction_rows=len(detail),unique_PWS=len(sums),PWS_with_two_directions=int((counts==2).sum()),bootstrap_unit="PWS_cluster_preserving_all_direction_rows"))
    out=pd.DataFrame(rows)
    save(out,"support_direction_weighting")
    print("DIRECTION AUDIT",out.loc[out.estimand.eq("interaction")].to_string(index=False),flush=True)


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    events,full=load_events()
    oracle_and_interaction(events,full)
    curve=budget_curves(events,full)
    unit_curves(events,full)
    season_audit(events,full)
    direction_weighting()
    summary={"status":"completed_retrospective_extension", "seed":SEED,"bootstrap_replicates":REPS,
        "event_records":len(events),"eligible_PWS":events.PWSID.nunique(),"reference_positive_PWS":769,
        "simultaneous_family_size":32,
        "first_positive_mean_vs_full_HAA5":int(curve.loc[curve.vs_full_HAA5_difference.gt(0),"requested_samples"].min()),
        "first_positive_pointwise_lower_vs_full_HAA5":int(curve.loc[curve.vs_full_HAA5_difference_ci_lower.gt(0),"requested_samples"].min()),
        "first_positive_simultaneous_lower_vs_full_HAA5":int(curve.loc[curve.vs_full_HAA5_difference_simultaneous_lower.gt(0),"requested_samples"].min()),
        "physical_sample_key":"PWSID,SampleID; one-to-one with released paired event records audited",
        "budget_policy":"uniform_without_replacement_min_requested_available",
        "scope":"finite_observed_grid_not_continuous_time_or_health_risk",
        "protocol_sha256":hashlib.sha256((OUT/"budget_retrospective_protocol.md").read_bytes()).hexdigest()}
    (OUT/"budget_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8",newline="\n")
    inputs=[ROOT/"data/processed/ucmr4/paired_haa_events.parquet",ROOT/"data/processed/ucmr4/system_full_grid_panel_maxima.parquet",ROOT/"wr_results/playbook_extension/round_policy_systems.csv",ROOT/"wr_results/core/disjoint_grid_system_directions.csv"]
    save(pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in inputs]),"budget_input_manifest")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
