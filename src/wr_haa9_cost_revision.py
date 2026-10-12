"""Declared resource scenarios, never market prices or compliance schedules.

Full-record HAA5 defines each system's baseline budget N + f*D, where f is
an assumed cost per distinct collection date in HAA5-assay units. Dates are
not claimed to identify actual field visits. HAA9 assay multiplier r ranges
from1 to5; f is0,1 or4. Both hard and expected constraints are retained.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from wr_haa9_budget_revision import capture, expected_distinct, load_events, bootstrap, save
ROOT=Path(__file__).resolve().parents[1]

def main():
    ev,full=load_events()
    ev=ev[ev.PWSID.isin(full.loc[full.HAA9_full_max>60,'PWSID'])]
    records=[]
    for pws,g in ev.groupby('PWSID',sort=True):
        n=len(g);counts=g.groupby('CollectionDate').size().to_numpy();d=len(counts)
        k=np.arange(n+1); p9=np.array([capture(n,int((g.HAA9>60).sum()),int(v)) for v in k])
        dates=np.array([expected_distinct(counts,int(v)) for v in k])
        h5=float((g.HAA5>60).any())
        for field in [0.,1.,4.]:
            budget=n+field*d
            for r in [1.,1.25,1.5,2.,2.5,3.,3.5,4.,4.5,5.]:
                for mode in ['hard','expected']:
                    cost=r*k+field*(np.minimum(k,d) if mode=='hard' else dates)
                    j=int(np.flatnonzero(cost<=budget+1e-10)[-1]);mix=0.
                    if mode=='expected' and j<n:
                        mix=float((budget-cost[j])/(cost[j+1]-cost[j]))
                    cap=p9[j] if j==n else p9[j]*(1-mix)+p9[j+1]*mix
                    actual_expected_cost=r*(j+mix)+field*(dates[j] if j==n else dates[j]*(1-mix)+dates[j+1]*mix)
                    records.append(dict(PWSID=pws,mode=mode,assay_ratio=r,date_cost_assumption=field,baseline_budget=budget,lower_k=j,upper_k=min(j+1,n) if mix else j,upper_k_probability=mix,expected_samples=j+mix,expected_cost=actual_expected_cost,maximum_cost_of_any_random_draw=r*j+field*min(j,d) if mode=='hard' else np.nan,HAA9_capture=cap,full_HAA5_capture=h5,difference=cap-h5))
    data=pd.DataFrame(records);save(data,'cost_resource_systems')
    rows=[]
    for i,(keys,g) in enumerate(data.groupby(['mode','assay_ratio','date_cost_assumption'],sort=True)):
        if keys[0]=='hard': assert (g.maximum_cost_of_any_random_draw<=g.baseline_budget+1e-9).all()
        assert (g.expected_cost<=g.baseline_budget+1e-9).all()
        point,ci,_=bootstrap(g[['HAA9_capture','difference']].to_numpy(),20260919+600+i,reps=5000)
        rows.append(dict(mode=keys[0],assay_ratio=keys[1],date_cost_assumption=keys[2],n_systems=len(g),HAA9_capture=point[0],difference=point[1],difference_ci_lower=ci[0,1],difference_ci_upper=ci[1,1],mean_expected_samples=g.expected_samples.mean(),mean_expected_cost=g.expected_cost.mean(),mean_baseline_budget=g.baseline_budget.mean()))
    save(pd.DataFrame(rows),'cost_resource_summary')
    (ROOT/'wr_results/haa9_revision/cost_resource_protocol.json').write_text(json.dumps({'status':'exploratory retrospective scenario declared in source before run','mode_hard':'Conservative worst-case number of distinct dates, min(k,D), guarantees per-system cap under stated additive costs','mode_expected':'Interpolate between adjacent k using expected distinct dates; NOT a hard per-draw cap','all_non_assay_components':'Assumed collection-date cost only; shipping, batch and followup costs unmeasured','r_grid':[1,1.25,1.5,2,2.5,3,3.5,4,4.5,5],'date_cost_grid':[0,1,4]},indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(rows).query('date_cost_assumption==0').to_string(index=False))

if __name__=='__main__':main()
