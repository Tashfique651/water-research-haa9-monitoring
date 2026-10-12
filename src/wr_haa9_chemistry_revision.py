"""Retrospective absolute chemistry, source-indicator and coherent interval audit.

Protocol: config/haa9_chemistry_revision_protocol.md. Original totals are never
replaced in the primary analysis. Measurement envelopes are scenarios, not
estimated laboratory uncertainty distributions.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
from wr_playbook_extensions import source_group

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'wr_results/haa9_revision'
P = ROOT / 'data/processed/ucmr4'
H5 = ['MCAA','DCAA','TCAA','MBAA','DBAA']
E4 = ['BCAA','BDCAA','CDBAA','TBAA']

def save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f'{name}.csv', index=False, lineterminator='\n')
    df.to_parquet(OUT / f'{name}.parquet', index=False)

def coherent_discordance(h5lo, h5hi, extra_lo, extra_hi, groups):
    """Sharp feasibility under a Cartesian species interval box, shared HAA5.

    For a discordant system every HAA5 event must stay <=60. Once feasible,
    each event's largest allowed HAA9 is min(HAA5 upper,60)+Extra4 upper.
    This preserves the common species and allows arbitrary interior sums.
    """
    h5lo,h5hi,extra_lo,extra_hi=[np.asarray(v,float) for v in (h5lo,h5hi,extra_lo,extra_hi)]
    if not all(np.isfinite(v).all() and (v>=0).all() for v in (h5lo,h5hi,extra_lo,extra_hi)):
        raise ValueError('Concentration bounds must be finite and nonnegative')
    if np.any(h5lo>h5hi) or np.any(extra_lo>extra_hi):
        raise ValueError('Lower bound exceeds upper bound')
    x = pd.DataFrame({'PWSID':groups, 'h5lo':h5lo, 'h5hi':h5hi,
                      'h9lo':h5lo+extra_lo,
                      'h9feasible':np.minimum(h5hi,60)+extra_hi})
    a = x.groupby('PWSID', sort=True).max()
    a['definite_discordance'] = (a.h9lo>60) & (a.h5hi<=60)
    a['possible_discordance'] = (a.h5lo<=60) & (a.h9feasible>60)
    assert (a.definite_discordance <= a.possible_discordance).all()
    return a.reset_index()

def models(frame, subset, scenario):
    data = frame.dropna(subset=['bromide','toc']).copy()
    if subset in ['conditional_invisibility','conditional_both_required']:
        data = data.loc[data.HAA9_full_max>60].copy()
        y = (data.HAA5_full_max<=60).astype(float) if subset=='conditional_invisibility' else (data.cat_both_required>0).astype(float)
    else:
        wanted = 'both_positive' if subset=='both_vs_neither' else 'HAA9_only'
        data = data.loc[data.category.isin(['neither_positive',wanted])].copy()
        y = (data.category==wanted).astype(float)
    parts = {}
    scale = {}
    for v in ['bromide','toc']:
        z=np.log2(1+data[v])
        iqr=float(z.quantile(.75)-z.quantile(.25))
        if iqr<=0: raise ValueError(f'Zero IQR {v}')
        parts[v]=(z-z.median())/iqr
        scale[v]={'unit':'ug/L','transform':'log2(1+x/(1 ug/L))', 'transformed_median':float(z.median()),'transformed_IQR':iqr}
    for v in ['observed_round_count','observed_site_count']:
        parts[v]=np.log2(data[v].astype(float))
    x=pd.DataFrame(parts,index=data.index)
    for v,ref in [('source_group','surface_water'),('size_profile','L')]:
        dum=pd.get_dummies(data[v].astype(str), prefix=v, dtype=float)
        if f'{v}_{ref}' in dum: dum=dum.drop(columns=f'{v}_{ref}')
        else: dum=dum.iloc[:,1:]
        x=pd.concat([x,dum],axis=1)
    x=sm.add_constant(x,has_constant='add').astype(float)
    try:
        fit=sm.GLM(y,x,family=sm.families.Binomial()).fit(cov_type='HC0')
        rows=[]
        for term in x.columns:
            b=float(fit.params[term]);se=float(fit.bse[term])
            rows.append({'model':subset,'nondetect_scenario':scenario,'term':term,'n_systems':len(data),'outcome_systems':int(y.sum()),'coefficient':b,'robust_se':se,'odds_ratio':float(np.exp(b)),'ci_lower':float(np.exp(b-1.959964*se)),'ci_upper':float(np.exp(b+1.959964*se)),'fit_converged':bool(fit.converged),'scale':json.dumps(scale.get(term,{}))})
        return rows
    except Exception as exc:
        return [{'model':subset,'nondetect_scenario':scenario,'term':'fit_failed','n_systems':len(data),'outcome_systems':int(y.sum()),'error':str(exc)}]

def main():
    ev=pd.read_parquet(P/'paired_haa_events.parquet')
    ev=ev.loc[ev.finite_grid_system_eligible].copy()
    full=pd.read_parquet(P/'system_full_grid_panel_maxima.parquet')
    assert len(full)==4904 and len(ev)==63105
    chemistry=ev.groupby('PWSID').agg(HAA5_median_ug_l=('HAA5','median'),HAA9_median_ug_l=('HAA9','median'),Extra4_median_ug_l=('extra4_ug_l','median'),Extra4_max_ug_l=('extra4_ug_l','max'),Extra4_fraction_median=('f_extra','median')).reset_index()
    base=full.merge(chemistry,on='PWSID',validate='one_to_one')
    contrib=pd.read_parquet(P/'system_contributions_primary_B60.parquet')
    base=base.merge(contrib[['PWSID','cat_both_required']],on='PWSID',validate='one_to_one')
    base['source_group']=base.facility_water_type_profile.map(source_group)
    base['category']=np.select([base.HAA9_full_max<=60,base.HAA5_full_max>60],['neither_positive','both_positive'],default='HAA9_only')
    indicator=pd.read_parquet(P/'indicator_system_round_summary.parquet')
    rows=[];summary=[];corr=[]
    for scenario in ['lower','midpoint','upper']:
        field=f'value_{scenario}_median_ug_l'
        a=indicator.groupby(['PWSID','Contaminant'])[field].median().unstack().rename(columns={'total organic carbon':'toc'})
        data=base.merge(a[['bromide','toc']],on='PWSID',how='left',validate='one_to_one')
        data['nondetect_scenario']=scenario
        save(data,f'chemistry_systems_{scenario}')
        for subset in ['conditional_invisibility','conditional_both_required','both_vs_neither','HAA9_only_vs_neither']:
            rows.extend(models(data,subset,scenario))
        for cohort in ['all_eligible','reference_positive']:
            d=data if cohort=='all_eligible' else data.loc[data.HAA9_full_max>60]
            for source in ['bromide','toc']:
                for endpoint in ['HAA5_full_max','HAA9_full_max','Extra4_max_ug_l','Extra4_fraction_median']:
                    pair=d[[source,endpoint]].dropna()
                    rho,p=spearmanr(pair[source],pair[endpoint])
                    corr.append({'cohort':cohort,'nondetect_scenario':scenario,'indicator':source,'endpoint':endpoint,'n_systems':len(pair),'spearman_rho':float(rho),'p_value_descriptive':float(p)})
        if scenario=='midpoint':
            for category,g in data.groupby('category'):
                for endpoint in ['HAA5_full_max','HAA9_full_max','Extra4_max_ug_l','Extra4_fraction_median','bromide','toc']:
                    z=g[endpoint].dropna()
                    summary.append({'category':category,'n_systems':len(g),'variable':endpoint,'n_observed':len(z),'q25':z.quantile(.25),'median':z.median(),'q75':z.quantile(.75)})
    save(pd.DataFrame(rows),'chemistry_environmental_models')
    save(pd.DataFrame(summary),'chemistry_category_summary')
    save(pd.DataFrame(corr),'chemistry_source_correlations')
    species=pd.read_parquet(ROOT/'results/additional/ucmr4_species_event_panel.parquet')
    species=species.loc[species.PWSID.isin(full.PWSID)].copy()
    assert len(species)==len(ev)
    species=species.sort_values(['PWSID','SampleID','SampleEventCode']).reset_index(drop=True)
    totals=species.groupby('PWSID')[['HAA5_official_ug_L','HAA9_official_ug_L']].max()
    official=((totals.HAA9_official_ug_L>60)&(totals.HAA5_official_ug_L<=60)).rename('official_discordance')
    counts=[]
    for eps in [0.,.05,.1,.2]:
        bounds={}
        for names,label in [(H5,'h5'),(E4,'e4')]:
            lo=np.zeros(len(species));hi=lo.copy()
            for name in names:
                detected=species[f'{name}_detected'].to_numpy(bool)
                low=species[f'{name}_lower_ug_L'].to_numpy(float)
                upper=species[f'{name}_upper_ug_L'].to_numpy(float)
                lo+=np.where(detected,low*(1-eps),low)
                hi+=np.where(detected,upper*(1+eps),upper)
            bounds[label+'lo']=lo;bounds[label+'hi']=hi
        detail=coherent_discordance(bounds['h5lo'],bounds['h5hi'],bounds['e4lo'],bounds['e4hi'],species.PWSID.to_numpy())
        detail=detail.merge(official,on='PWSID',validate='one_to_one')
        detail['quantified_relative_envelope']=eps
        save(detail,f'chemistry_discordance_envelope_{int(eps*100)}')
        for cohort in ['all_eligible','official_discordant']:
            d=detail if cohort=='all_eligible' else detail.loc[detail.official_discordance]
            counts.append({'cohort':cohort,'n_systems':len(d),'quantified_relative_envelope':eps,'definite':int(d.definite_discordance.sum()),'possible_not_definite':int((d.possible_discordance&~d.definite_discordance).sum()),'not_possible':int((~d.possible_discordance).sum()),'possible_including_definite':int(d.possible_discordance.sum())})
    save(pd.DataFrame(counts),'chemistry_discordance_summary')
    meta={'protocol':'config/haa9_chemistry_revision_protocol.md','primary_totals_unchanged':True,'measurement_precision_observed':False,'n_eligible_systems':len(full),'n_events':len(ev),'data_files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'paired_haa_events.parquet',P/'indicator_system_round_summary.parquet',ROOT/'results/additional/ucmr4_species_event_panel.parquet']}}
    (OUT/'chemistry_revision_manifest.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(counts).to_string(index=False))
    print(pd.DataFrame(rows).query("term in ['bromide','toc']").to_string(index=False))

if __name__=='__main__':
    main()
