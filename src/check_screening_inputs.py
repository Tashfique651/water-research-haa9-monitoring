"""Reconstruct and compare early predictors using the original six-field key."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'src'))
from current_monitoring import load_events
from current_screening import SPECIES9
e=load_events();cols=[f'{s}_lower_ug_L' for s in SPECIES9]
src=r/'results/additional/ucmr4_species_event_panel.parquet';s=pd.read_parquet(src)
keys=['PWSID','FacilityID','SamplePointID','CollectionDate','SampleEventCode','SampleID']
s.CollectionDate=pd.to_datetime(s.CollectionDate,format='mixed')
for k in keys:
 if k!='CollectionDate':e[k]=e[k].astype(str);s[k]=s[k].astype(str)
rows=[];early=[]
for p,g in e.groupby('PWSID',sort=True):
 rd=g.groupby('SampleEventCode').CollectionDate.agg(['median','min','max']).reset_index().sort_values(['median','SampleEventCode'])
 if len(rd)!=4:continue
 ge=g[g.SampleEventCode.isin(rd.SampleEventCode.iloc[:2])];gl=g[g.SampleEventCode.isin(rd.SampleEventCode.iloc[2:])]
 if ge.CollectionDate.max()>=gl.CollectionDate.min():continue
 rows.append({'PWSID':p,'early_HAA5':ge.HAA5.max(),'early_HAA9':ge.HAA9.max(),'outcome':int(gl.HAA9.gt(60).any()),'early_samples':len(ge)})
 early.append(ge[keys])
j=pd.concat(early).merge(s[keys+cols],on=keys,how='left',validate='one_to_one')
assert not j[cols].isna().any().any()
f=j.groupby('PWSID')[cols].max().rename(columns={f'{x}_lower_ug_L':f'early_{x}' for x in SPECIES9})
a=pd.DataFrame(rows).set_index('PWSID').join(f).sort_index()
b=pd.read_csv(r/'wr_results/haa9_revision/forward_out_of_fold_predictions.csv',dtype={'PWSID':str}).set_index('PWSID').sort_index()
errors={c:float(np.max(np.abs(a[c].to_numpy()-b.loc[a.index,c].to_numpy()))) for c in a.columns}
assert all(v<1e-10 for v in errors.values()),errors
out=r/'verification/inputs';a.reset_index().to_csv(out/'screening_features.csv',index=False)
rep={'six_field_linkage':keys,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'eligible_systems':len(a),'early_matched_samples':len(j),'later_positive_systems':int(a.outcome.sum()),'maximum_absolute_column_errors_against_released_features':errors,'result':'matched'}
(out/'screening_feature_audit.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
