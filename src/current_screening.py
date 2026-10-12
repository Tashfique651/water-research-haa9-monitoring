"""Refitted, fixed-fold screening sensitivity using released early features.

Run: python src/current_screening.py --bootstrap 2000
This additional reconstruction leaves the archived results unchanged.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'verification/results/screening'
WORKLOADS=np.array([.05,.10,.15,.20,.30])
SPECIES5=['MCAA','DCAA','TCAA','MBAA','DBAA']
SPECIES9=SPECIES5+['BCAA','BDCAA','CDBAA','TBAA']
SPECS=[['early_HAA5'],['early_HAA9'],['early_'+s for s in SPECIES5],['early_'+s for s in SPECIES9]]
NAMES=['HAA5 maximum','HAA9 maximum','HAA5 logistic','HAA9 logistic','Five-species logistic','Nine-species logistic']

def fold_id(pws: str)->int:
    return int(hashlib.sha256(('wr-screening-procedure-20261011|'+pws).encode()).hexdigest(),16)%5

def allocation(train_score:np.ndarray, weights:np.ndarray, score:np.ndarray,target:float)->tuple[np.ndarray,float,float]:
    """Training-only weighted quantile, with fractional allocation of exact ties."""
    order=np.argsort(-train_score,kind='stable'); s=train_score[order]; w=weights[order]
    keep=w>0;s=s[keep];w=w[keep]
    if not len(s):raise ValueError('Empty weighted training set')
    j=int(np.searchsorted(np.cumsum(w),target*w.sum(),side='left'))
    cutoff=s[min(j,len(s)-1)]
    above=w[s>cutoff].sum(); ties=w[s==cutoff].sum()
    fraction=float(np.clip((target*w.sum()-above)/ties,0,1))
    return np.where(score>cutoff,1.,np.where(score==cutoff,fraction,0.)),float(cutoff),fraction

def predictions(features:list[np.ndarray], scalars:np.ndarray,y:np.ndarray,folds:np.ndarray,w:np.ndarray, details:bool=False):
    n=len(y); scores=np.zeros((n,6)); policies=np.zeros((n,6,5)); records=[]
    for f in range(5):
        tr=folds!=f;te=~tr
        trains=[scalars[tr,0],scalars[tr,1]];tests=[scalars[te,0],scalars[te,1]]
        for x in features:
            scale=StandardScaler().fit(x[tr],sample_weight=w[tr])
            model=LogisticRegression(C=1.,solver='lbfgs',max_iter=3000,tol=1e-8)
            model.fit(scale.transform(x[tr]),y[tr],sample_weight=w[tr])
            trains.append(model.predict_proba(scale.transform(x[tr]))[:,1]);tests.append(model.predict_proba(scale.transform(x[te]))[:,1])
        for j in range(6):
            scores[te,j]=tests[j]
            for k,target in enumerate(WORKLOADS):
                p,c,t=allocation(trains[j],w[tr],tests[j],target);policies[te,j,k]=p
                if details:records.append({'fold':f,'strategy':NAMES[j],'target':target,'cutoff':c,'tie_probability':t})
    return scores,policies,records

def metrics(scores:np.ndarray,policies:np.ndarray,y:np.ndarray,w:np.ndarray):
    stats=[]
    for j in range(6):
        s=scores[:,j];m=[roc_auc_score(y,s,sample_weight=w),average_precision_score(y,s,sample_weight=w)]
        if j>=2:
            p=np.clip(s,1e-12,1-1e-12);m += [np.average((p-y)**2,weights=w),np.average(-(y*np.log(p)+(1-y)*np.log1p(-p)),weights=w)]
        else:m += [np.nan,np.nan]
        stats.append(m)
    sens=np.einsum('ijk,i->jk',policies,w*y)/(w@y)
    return np.array(stats), np.stack([sens[1]-sens[0],sens[5]-sens[4]])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bootstrap',type=int,default=2000);args=parser.parse_args()
    if args.bootstrap<2:raise ValueError('At least two bootstrap replicates required')
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'wr_results/haa9_revision/forward_out_of_fold_predictions.csv'
    d=pd.read_csv(source,dtype={'PWSID':str}).sort_values('PWSID').reset_index(drop=True)
    y=d.outcome.to_numpy(int);folds=d.PWSID.map(fold_id).to_numpy();n=len(d)
    assert n==2636 and y.sum()==420 and d.PWSID.is_unique
    assert np.bincount(folds).tolist()==[533,533,537,510,523]
    assert (pd.to_datetime(d.predictor_end,format='mixed')<pd.to_datetime(d.outcome_start,format='mixed')).all()
    features=[np.log1p(d[cols].to_numpy(float)) for cols in SPECS];scalars=d[['early_HAA5','early_HAA9']].to_numpy(float)
    base_s,base_p,cutoffs=predictions(features,scalars,y,folds,np.ones(n),True)
    point_m,point_g=metrics(base_s,base_p,y,np.ones(n))
    pd.DataFrame(cutoffs).to_csv(OUT/'base_cutoffs.csv',index=False)
    d[['PWSID','predictor_end','outcome_start','outcome']].assign(fold=folds).to_csv(OUT/'folds.csv',index=False)
    out=pd.DataFrame({'PWSID':d.PWSID})
    for j,name in enumerate(NAMES):
        out[name]=base_s[:,j]
        for k,t in enumerate(WORKLOADS):out[f'{name}_allocation_{t:.2f}']=base_p[:,j,k]
    out.to_csv(OUT/'base_predictions.csv',index=False)
    rng=np.random.default_rng(2026101107);ix=[np.where(folds==f)[0] for f in range(5)]
    fixed_m=[];full_m=[];fixed_g=[];full_g=[]
    for b in range(args.bootstrap):
        w=np.zeros(n)
        for inds in ix:w[inds]=rng.multinomial(len(inds),np.full(len(inds),1/len(inds)))
        fm,fg=metrics(base_s,base_p,y,w);s,p,_=predictions(features,scalars,y,folds,w);rm,rg=metrics(s,p,y,w)
        fixed_m.append(fm);fixed_g.append(fg);full_m.append(rm);full_g.append(rg)
        if (b+1)%100==0:print(f'Completed {b+1}/{args.bootstrap}',flush=True)
    fixed_m=np.array(fixed_m);full_m=np.array(full_m);fixed_g=np.array(fixed_g);full_g=np.array(full_g)
    np.savez_compressed(OUT/'bootstrap.npz',fixed_metrics=fixed_m,refitted_metrics=full_m,fixed_gains=fixed_g,refitted_gains=full_g)
    rows=[]
    for j,name in enumerate(NAMES):
        for k,metric in enumerate(['AUROC','Average precision','Brier score','Log loss']):
            if j<2 and k>1:continue
            fl,fh=np.quantile(fixed_m[:,j,k],[.025,.975]);rl,rh=np.quantile(full_m[:,j,k],[.025,.975])
            rows.append({'strategy':name,'metric':metric,'estimate':point_m[j,k],'fixed_low':fl,'fixed_high':fh,'refitted_low':rl,'refitted_high':rh})
    pd.DataFrame(rows).to_csv(OUT/'accuracy.csv',index=False)
    rows=[]
    for j,name in enumerate(['Scalar totals','Species models']):
        fh=np.quantile(np.abs(fixed_g[:,j,:]-point_g[j]).max(axis=1),.95);rh=np.quantile(np.abs(full_g[:,j,:]-point_g[j]).max(axis=1),.95)
        for k,t in enumerate(WORKLOADS):
            v=point_g[j,k];rows.append({'comparison':name,'target':t,'gain':v,'fixed_low':v-fh,'fixed_high':v+fh,'refitted_low':v-rh,'refitted_high':v+rh})
    pd.DataFrame(rows).to_csv(OUT/'gains.csv',index=False)
    (OUT/'run.json').write_text(json.dumps({'bootstrap_replicates':args.bootstrap,'seed':2026101107,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'fold_sizes':np.bincount(folds).tolist(),'systems':n,'positive_systems':int(y.sum()),'feature_route':'released early-feature CSV; see input reconstruction audit','complete':True},indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
