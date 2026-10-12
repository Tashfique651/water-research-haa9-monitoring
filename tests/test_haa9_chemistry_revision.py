"""Independent finite box and hard-budget checks for the revision."""
import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wr_haa9_chemistry_revision import coherent_discordance

def test_shared_sum_feasibility_interior_required():
    x=coherent_discordance([50],[65],[2],[2],['A']).iloc[0]
    assert x.possible_discordance and not x.definite_discordance
    # A permitted interior HAA5=60, Extra4=2 is discordant. Independently
    # maximizing HAA5 and HAA9 would miss the requirement to share HAA5.

def test_box_classification_matches_exhaustive_grid():
    rng=np.random.default_rng(810)
    for _ in range(100):
        lo=rng.integers(48,66,2);hi=lo+rng.integers(0,5,2)
        elo=rng.integers(0,8,2);ehi=elo+rng.integers(0,4,2)
        flags=[]
        for a,b,c,d in itertools.product(range(lo[0],hi[0]+1),range(lo[1],hi[1]+1),range(elo[0],ehi[0]+1),range(elo[1],ehi[1]+1)):
            flags.append(max(a,b)<=60 and max(a+c,b+d)>60)
        x=coherent_discordance(lo,hi,elo,ehi,['A','A']).iloc[0]
        assert x.possible_discordance==any(flags)
        assert x.definite_discordance==all(flags)

def test_invalid_concentration_box_rejected():
    with pytest.raises(ValueError):coherent_discordance([2],[1],[0],[1],['A'])
    with pytest.raises(ValueError):coherent_discordance([-1],[1],[0],[1],['A'])

def test_all_declared_hard_budget_realizations_bounded():
    path=ROOT/'wr_results/haa9_revision/cost_resource_systems.parquet'
    if not path.exists(): pytest.skip('Run revision resource analysis first')
    d=pd.read_parquet(path);h=d.loc[d['mode'].eq('hard')]
    assert len(h)==769*3*10
    assert (h.upper_k_probability==0).all()
    assert (h.maximum_cost_of_any_random_draw<=h.baseline_budget+1e-9).all()
    assert (d.expected_cost<=d.baseline_budget+1e-9).all()
