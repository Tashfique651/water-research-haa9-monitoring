"""Independent synthetic checks for the frozen budget and policy extension."""
from itertools import combinations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from wr_haa9_budget_revision import capture, expected_distinct, warm_first_capture, shared_oracle, interaction_from_units


@pytest.mark.parametrize("n",range(1,10))
def test_capture_bruteforce(n):
    for m in range(n+1):
        for k in range(n+3):
            subsets=list(combinations(range(n),min(k,n)))
            brute=np.mean([any(i<m for i in s) for s in subsets])
            assert capture(n,m,k)==pytest.approx(brute,abs=1e-12)


@pytest.mark.parametrize("labels",[[0],[0,0,1],[0,1,2,2],[0,0,1,1,2,3]])
def test_expected_distinct_bruteforce(labels):
    for k in range(len(labels)+2):
        brute=np.mean([len({labels[i] for i in s}) for s in combinations(range(len(labels)),min(k,len(labels)))])
        assert expected_distinct(np.unique(labels,return_counts=True)[1],k)==pytest.approx(brute)


@pytest.mark.parametrize("warm,cold",[([], [0,1]),([0,1],[0,0,1]),([1,1],[]),([0,0],[1,1]),([0],[0])])
def test_warm_first_bruteforce(warm,cold):
    for k in range(len(warm)+len(cold)+2):
        use=min(k,len(warm)+len(cold));a=min(use,len(warm));b=use-a
        outcomes=[any(warm[i] for i in w) or any(cold[i] for i in c) for w in combinations(range(len(warm)),a) for c in combinations(range(len(cold)),b)]
        assert warm_first_capture(np.asarray(warm),np.asarray(cold),k)==pytest.approx(np.mean(outcomes))


def test_shared_oracle_counterexample():
    fixture=pd.DataFrame({"HAA5":[40.,70.],"HAA9":[100.,80.]})
    a,b,ties=shared_oracle(fixture)
    assert (a,b,ties)==(0.,1.,1)
    assert fixture.HAA5.gt(60).any()


def test_shared_oracle_tie_uniform():
    fixture=pd.DataFrame({"HAA5":[40.,70.,75.],"HAA9":[100.,100.,80.]})
    assert shared_oracle(fixture)==(.5,1.,2)


@pytest.mark.parametrize("a,b,expected",[([0,0],[0,1],.5),([1,0],[1,0],0.),([1,0],[1,1],-.5)])
def test_nested_interaction_can_be_positive_zero_negative(a,b,expected):
    assert interaction_from_units(a,b)==expected


def test_invalid_capture_inputs_rejected():
    with pytest.raises(ValueError):capture(2,3,1)
    with pytest.raises(ValueError):capture(2,1,-1)
