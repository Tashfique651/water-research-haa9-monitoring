"""Deterministic checks for the current processed-input reconstruction."""
from pathlib import Path
import sys,hashlib,json,itertools,math
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from current_monitoring import capture,pair_capture
from current_screening import allocation,fold_id
RESULTS=ROOT/'verification/results'

@pytest.mark.parametrize('n,positive,k',[(6,2,3),(4,1,9),(7,0,3),(7,7,2),(0,0,0)])
def test_exact_capture_against_enumeration(n,positive,k):
    picked=list(itertools.combinations(range(n),min(n,k)))
    expected=np.mean([any(x<positive for x in s) for s in picked])
    assert capture(n,positive,k)==pytest.approx(expected)

@pytest.mark.parametrize('args',[(-1,0,1),(4,5,1),(4,2,-1)])
def test_capture_rejects_invalid_counts(args):
    with pytest.raises(ValueError):capture(*args)

def test_two_round_capture():
    a=np.array([20.,61.,80.]);b=np.array([40.,50.])
    assert pair_capture(a,b,60)==pytest.approx(2/3)

def test_baseline_preservation_and_reference():
    x=pd.read_csv(RESULTS/'later_contributions.csv')
    for rule in ['uniform','early_HAA5','early_HAA9']:
        assert np.all(x[f'{rule}_augmentation']>=x.baseline)
        assert np.allclose(x[f'{rule}_augmentation'],x.baseline+x[f'{rule}_gain'])
        assert np.allclose(x.loc[x.reference.eq(0),f'{rule}_gain'],0)

def test_strict_dates():
    x=pd.read_csv(RESULTS/'later_contributions.csv')
    assert (pd.to_datetime(x.early_end)<pd.to_datetime(x.later_start)).all()

def test_cohorts():
    x=pd.read_csv(RESULTS/'budget_contributions.csv');y=pd.read_csv(RESULTS/'later_contributions.csv')
    assert (len(x),int(x.reference.sum()))==(4904,769)
    y=y[y.benchmark_ug_L.eq(60)]
    assert (len(y),int(y.reference.sum()))==(2636,420)

def test_budget_monotonicity():
    x=pd.read_csv(RESULTS/'budget_contributions.csv')
    assert (np.diff(x[[f'HAA9_k{k}' for k in range(1,17)]],axis=1)>=-1e-12).all()

def test_conditional_and_archive_denominators():
    x=pd.read_csv(RESULTS/'later_summary.csv')
    c=x[x.denominator.eq('conditional')].set_index('rule');a=x[x.denominator.eq('all_systems')].set_index('rule')
    assert np.allclose(c.gain*420/2636,a.gain)
    assert c.loc['early_HAA5','augmentation']==pytest.approx(.7917460317460318,abs=1e-12)

def test_exact_tie_allocation():
    score=np.array([3.,3.,1.,0.]);w=np.ones(4)
    p,cut,tie=allocation(score,w,score,.25)
    assert np.allclose(p,[.5,.5,0,0]);assert cut==3.;assert tie==.5

def test_weighted_cutoff_matches_duplicate_cases():
    score=np.array([1.,2.,2.,5.]);w=np.array([2.,1.,3.,2.]);ev=np.array([0.,2.,5.])
    p,_,_=allocation(score,w,ev,.30)
    repeated=np.repeat(score,w.astype(int))
    q,_,_=allocation(repeated,np.ones(len(repeated)),ev,.30)
    assert np.allclose(p,q)

def test_fold_design_and_predictor_audit():
    p=pd.read_csv(RESULTS/'screening/folds.csv',dtype={'PWSID':str})
    assert p.PWSID.map(fold_id).equals(p.fold)
    assert p.groupby('fold').size().tolist()==[533,533,537,510,523]
    audit=json.loads((ROOT/'verification/inputs/screening_feature_audit.json').read_text())
    assert audit['early_matched_samples']==24211
    assert all(v==0 for v in audit['maximum_absolute_column_errors_against_released_features'].values())

def test_benchmark_comparison_family():
    b=pd.read_csv(RESULTS/'benchmark_summary.csv')
    assert b.benchmark_ug_L.tolist()==list(range(20,101,5))
    assert (b.conditional_gain_low>0).all()
    assert b.loc[b.conditional_targeting_low>0,'benchmark_ug_L'].tolist()==[80,90,100]
    x=b[b.benchmark_ug_L.eq(60)].iloc[0]
    assert x.conditional_targeting_low<0<x.conditional_targeting_high

def test_screening_resampling_completed():
    r=json.loads((RESULTS/'screening/run.json').read_text())
    assert r['complete'] and r['bootstrap_replicates']==2000
    x=pd.read_csv(RESULTS/'screening/gains.csv')
    scalar=x[x.comparison.eq('Scalar totals')].set_index('target')
    assert scalar.loc[.1,'refitted_low']<0
    assert scalar.loc[.3,'refitted_low']>0
