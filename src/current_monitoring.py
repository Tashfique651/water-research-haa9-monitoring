"""Exact monitoring replay and whole-system bootstrap from paired observations.

Run from the repository root:
    python src/current_monitoring.py --bootstrap 10000
The analysis does not change any input or the frozen historical result tables.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'verification/results'
BENCHMARKS = np.arange(20., 101., 5.)
RULES = ('uniform', 'early_HAA5', 'early_HAA9')

def capture(n: int, positive: int, k: int) -> float:
    if not 0 <= positive <= n or k < 0:
        raise ValueError('Require 0 <= positive <= n and k >= 0')
    k = min(k, n)
    if n == 0 or k == 0: return 0.0
    if n-positive < k: return 1.0
    return 1.0 - math.comb(n-positive, k)/math.comb(n,k)

def load_events() -> pd.DataFrame:
    compact = ROOT / 'verification/inputs/paired_events.csv.gz'
    if compact.exists():
        events = pd.read_csv(compact, dtype={x:str for x in ['PWSID','FacilityID','SamplePointID','SampleID','SampleEventCode','site_id']})
    else:
        events = pd.read_parquet(ROOT/'data/processed/ucmr4/paired_haa_events.parquet')
    events = events.loc[events.finite_grid_system_eligible.eq(True)].copy()
    events['CollectionDate'] = pd.to_datetime(events.CollectionDate, format='mixed', errors='raise')
    keys = ['PWSID','FacilityID','SamplePointID','CollectionDate','SampleEventCode','SampleID']
    if events.duplicated(keys).any(): raise ValueError('Duplicate six-field sample key')
    if not (events.HAA9 >= events.HAA5-1e-8).all(): raise ValueError('Non-nested totals')
    return events.sort_values(keys).reset_index(drop=True)

def budget_contributions(events: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for system,g in events.groupby('PWSID',sort=True):
        n=len(g); m5=int(g.HAA5.gt(60).sum()); m9=int(g.HAA9.gt(60).sum()); h=float(m5>0)
        row={'PWSID':system,'n_samples':n,'reference':int(m9>0),'baseline':h}
        for k in range(1,17):
            q=capture(n,m9,k)
            row.update({f'HAA9_k{k}':q,f'HAA5_k{k}':capture(n,m5,k),f'gain_k{k}':(1-h)*q,f'loss_k{k}':h*(1-q),f'added_reports_k{k}':min(k,n)})
        rows.append(row)
    return pd.DataFrame(rows)

def pair_capture(first: np.ndarray, second: np.ndarray, benchmark: float) -> float:
    if not len(first) or not len(second): raise ValueError('Each later round requires samples')
    return 1-float(np.mean(first<=benchmark)*np.mean(second<=benchmark))

def later_contributions(events: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for system,g in events.groupby('PWSID',sort=True):
        dates=g.groupby('SampleEventCode').CollectionDate.agg(['min','max','median']).reset_index()
        if len(dates)!=4: continue
        dates=dates.sort_values(['median','SampleEventCode'])
        early_codes=dates.SampleEventCode.iloc[:2].tolist(); later_codes=dates.SampleEventCode.iloc[2:].tolist()
        early=g.loc[g.SampleEventCode.isin(early_codes)]; late=g.loc[g.SampleEventCode.isin(later_codes)]
        if early.CollectionDate.max()>=late.CollectionDate.min(): continue
        early_parts=[early.loc[early.SampleEventCode.eq(code)] for code in early_codes]
        later_parts=[late.loc[late.SampleEventCode.eq(code)] for code in later_codes]
        common=sorted(set(early_parts[0].site_id)&set(early_parts[1].site_id))
        chosen={}
        for panel in ['HAA5','HAA9']:
            if common:
                means=early.loc[early.site_id.isin(common)].groupby(['site_id','SampleEventCode'])[panel].mean().groupby('site_id').mean()
                best=means.max(); chosen[panel]=sorted(means.index[np.isclose(means,best,rtol=0,atol=1e-12)])
            else: chosen[panel]=[None]
        # One selected site is shared by both later rounds. Only the fallback
        # uses a round-wide set when that selected site's observations are absent.
        arrays={}
        for panel,sites in chosen.items():
            arrays[panel]=[]
            for site in sites:
                pools=[]
                for part in later_parts:
                    pool=part.loc[part.site_id.eq(site)] if site is not None else part.iloc[:0]
                    if pool.empty: pool=part
                    pools.append(pool.HAA9.to_numpy())
                arrays[panel].append(pools)
        for b in BENCHMARKS:
            h=float(late.HAA5.gt(b).any()); ref=int(late.HAA9.gt(b).any())
            q0=pair_capture(*(p.HAA9.to_numpy() for p in later_parts),b)
            q5=float(np.mean([pair_capture(a,c,b) for a,c in arrays['HAA5']]))
            q9=float(np.mean([pair_capture(a,c,b) for a,c in arrays['HAA9']]))
            row={'PWSID':system,'benchmark_ug_L':b,'reference':ref,'baseline':h,'uniform_capture':q0,'early_HAA5_capture':q5,'early_HAA9_capture':q9,'early_end':early.CollectionDate.max().isoformat(),'later_start':late.CollectionDate.min().isoformat()}
            for rule,q in zip(RULES,[q0,q5,q9]):
                row[f'{rule}_gain']=(1-h)*q;row[f'{rule}_augmentation']=h+(1-h)*q
            rows.append(row)
    return pd.DataFrame(rows)

def simultaneous(values:np.ndarray,n_boot:int,seed:int,batch:int=250) -> tuple[np.ndarray,float]:
    """Symmetric 95% maximum-absolute centered bootstrap band."""
    values=np.asarray(values,dtype=float);n=len(values);point=values.mean(axis=0)
    rng=np.random.default_rng(seed); errors=[]
    for start in range(0,n_boot,batch):
        w=rng.multinomial(n,np.repeat(1/n,n),size=min(batch,n_boot-start))
        errors.append(np.abs(w@values/n-point).max(axis=1))
    return point,float(np.quantile(np.concatenate(errors),.95))

def summarize_budgets(contrib:pd.DataFrame,n_boot:int)->pd.DataFrame:
    rows=[]
    for denominator,mask,seed in [('conditional',contrib.reference.eq(1),20261010),('all_systems',np.ones(len(contrib),bool),20261011)]:
        g=contrib.loc[mask]; v=g[[f'gain_k{k}' for k in range(1,17)]].to_numpy();est,hw=simultaneous(v,n_boot,seed)
        for k in range(1,17):
            rows.append({'denominator':denominator,'k':k,'n_systems':len(g),'baseline':g.baseline.mean(),'HAA9_capture':g[f'HAA9_k{k}'].mean(),'HAA5_capture':g[f'HAA5_k{k}'].mean(),'augmentation':g.baseline.mean()+est[k-1],'gain':est[k-1],'gain_simultaneous_low':est[k-1]-hw,'gain_simultaneous_high':est[k-1]+hw,'loss':g[f'loss_k{k}'].mean(),'discordance':(g[f'loss_k{k}']+g[f'gain_k{k}']).mean(),'achieved_reports':g[f'added_reports_k{k}'].mean()})
    return pd.DataFrame(rows)

def summarize_later(contrib:pd.DataFrame,n_boot:int)->pd.DataFrame:
    g=contrib.loc[contrib.benchmark_ug_L.eq(60)].sort_values('PWSID');rows=[]
    for denominator,mask,seed in [('conditional',g.reference.eq(1),20261012),('all_systems',np.ones(len(g),bool),20261013)]:
        d=g.loc[mask]; v=d[[f'{r}_gain' for r in RULES]].to_numpy();est,hw=simultaneous(v,n_boot,seed)
        contrasts=np.column_stack([v[:,1]-v[:,0],v[:,2]-v[:,0],v[:,2]-v[:,1]])
        ce,ch=simultaneous(contrasts,n_boot,seed)
        for j,rule in enumerate(RULES):
            rows.append({'denominator':denominator,'rule':rule,'n_systems':len(d),'baseline':d.baseline.mean(),'HAA9_capture':d[f'{rule}_capture'].mean(),'augmentation':d.baseline.mean()+est[j],'gain':est[j],'gain_simultaneous_low':est[j]-hw,'gain_simultaneous_high':est[j]+hw,'expected_new_systems':d[f'{rule}_gain'].sum(),'paired_contrast':['early_HAA5_minus_uniform','early_HAA9_minus_uniform','early_HAA9_minus_early_HAA5'][j],'contrast':ce[j],'contrast_simultaneous_low':ce[j]-ch,'contrast_simultaneous_high':ce[j]+ch})
    return pd.DataFrame(rows)

def benchmark_family(contrib:pd.DataFrame,n_boot:int)->pd.DataFrame:
    ids=sorted(contrib.PWSID.unique());bs=BENCHMARKS
    def matrix(col):return contrib.pivot(index='PWSID',columns='benchmark_ug_L',values=col).loc[ids,bs].to_numpy(float)
    ref=matrix('reference'); gain=matrix('early_HAA5_gain');diff=gain-matrix('uniform_gain');n=len(ids)
    den=ref.sum(axis=0);point={'conditional_gain':gain.sum(axis=0)/den,'conditional_targeting':diff.sum(axis=0)/den,'all_system_gain':gain.mean(axis=0),'all_system_targeting':diff.mean(axis=0)}
    maxima={k:[] for k in point};rng=np.random.default_rng(2026101201)
    for start in range(0,n_boot,250):
        w=rng.multinomial(n,np.repeat(1/n,n),size=min(250,n_boot-start));denb=w@ref
        if np.any(denb==0):raise ValueError('Bootstrap replicate contains empty reference')
        values={'conditional_gain':w@gain/denb,'conditional_targeting':w@diff/denb,'all_system_gain':w@gain/n,'all_system_targeting':w@diff/n}
        for key in point:maxima[key].append(np.abs(values[key]-point[key]).max(axis=1))
    halfwidth={k:float(np.quantile(np.concatenate(v),.95)) for k,v in maxima.items()}
    baseline=matrix('baseline'); aug=baseline+gain
    rows=[]
    for j,b in enumerate(bs):
        row={'benchmark_ug_L':b,'n_systems':n,'n_reference':int(den[j]),'baseline_conditional':baseline[:,j].sum()/den[j],'augmentation_conditional':aug[:,j].sum()/den[j]}
        for key in point:row.update({key:point[key][j],key+'_low':point[key][j]-halfwidth[key],key+'_high':point[key][j]+halfwidth[key]})
        rows.append(row)
    return pd.DataFrame(rows)

def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--bootstrap',type=int,default=10000);args=p.parse_args()
    if args.bootstrap<100:raise ValueError('Use at least 100 resamples')
    OUT.mkdir(parents=True,exist_ok=True);events=load_events()
    budget=budget_contributions(events);later=later_contributions(events)
    assert len(events)==63105 and len(budget)==4904 and budget.reference.sum()==769
    b60=later.loc[later.benchmark_ug_L.eq(60)]
    assert len(b60)==2636 and b60.reference.sum()==420
    outputs={'budget_contributions':budget,'later_contributions':later,'budget_summary':summarize_budgets(budget,args.bootstrap),'later_summary':summarize_later(later,args.bootstrap),'benchmark_summary':benchmark_family(later,args.bootstrap)}
    for name,table in outputs.items():table.to_csv(OUT/f'{name}.csv',index=False,lineterminator='\n')
    (OUT/'monitoring_verification.json').write_text(json.dumps({'events':len(events),'systems':len(budget),'B60_positive_systems':int(budget.reference.sum()),'strict_date_systems':len(b60),'later_positive_systems':int(b60.reference.sum()),'bootstrap_replicates':args.bootstrap,'source':'archived paired events; existing concentrations unchanged','scope':'independent reconstruction, not an original-publisher download'},indent=2))
    print(outputs['later_summary'].to_string(index=False));print(outputs['benchmark_summary'].loc[lambda d:d.benchmark_ug_L.eq(60)].to_string(index=False))
if __name__=='__main__':main()
