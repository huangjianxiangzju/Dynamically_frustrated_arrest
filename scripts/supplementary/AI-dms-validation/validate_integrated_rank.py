"""Read-only independent checks of the deposited integrated ranking tables."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,default=ROOT/'data/AI-validation-[CB,VESM]')
p.add_argument('--tables',type=Path,default=None)
a=p.parse_args();data=a.data;tables=a.tables or data
pairs=['6_vs_8','8_vs_10','10_vs_12','12_vs_14','14_vs_16','16_vs_18']
parts=[]
for pair in pairs:
    q=pd.read_csv(data/f'CB_results_{pair}_proteinmpnn/position_summary.csv')
    assert q.position.is_unique and set(q.position)==set(range(1,1369)),pair
    assert q.CB_bias_zscore.notna().all()
    parts.append(q[['position','wt','CB_bias_zscore']])
cb=pd.concat(parts)
assert cb.groupby('position').wt.nunique().eq(1).all()
ref=pd.DataFrame(index=pd.Index(range(1,1369),name='position'))
ref['wt']=cb.groupby('position').wt.first()
ref['cb_force']=-cb.groupby('position').CB_bias_zscore.mean()
vesm=pd.read_csv(data/'SpCas9_VESM3B_full_position_summary.csv').set_index('position')
assert vesm.index.is_unique and set(vesm.index)==set(ref.index)
ref['vesm_constraint']=-vesm.mean_LLR
md=pd.read_csv(data/'full_superset.csv').set_index('Residue')
cats=['Switch','GCCM','SB_hub','Hydro_hub','BC']
assert md.index.is_unique and len(md)==311
counts=md[cats].eq('Y').sum(axis=1)
assert np.array_equal(counts.to_numpy(),md.n.to_numpy())
assert md[cats].eq('Y').sum().tolist()==[84,54,46,90,89]
assert counts.ge(2).sum()==52
ref['md_n']=counts.reindex(ref.index,fill_value=0)
for source,target in [('cb_force','pct_cb'),('vesm_constraint','pct_vesm'),('md_n','pct_md')]:
    # Independent pandas ranking, not the plotting script's scipy helper.
    ref[target]=ref[source].rank(method='average',pct=True)*100
ref['combined_score']=ref[['pct_cb','pct_vesm','pct_md']].mean(axis=1)
ref['evidence_count']=ref[['pct_cb','pct_vesm','pct_md']].gt(50).sum(axis=1)
full=pd.read_csv(tables/'integrated_rank_table_full_precision.csv').set_index('position')
assert full.index.is_unique and set(full.index)==set(ref.index)
full=full.reindex(ref.index)
assert full.wt.equals(ref.wt)
numeric=['cb_force','vesm_constraint','md_n','pct_cb','pct_vesm','pct_md','combined_score','evidence_count']
for col in numeric:np.testing.assert_allclose(full[col],ref[col],rtol=1e-12,atol=1e-10,err_msg=col)
rounded=pd.read_csv(tables/'integrated_rank_table.csv').set_index('position').reindex(ref.index)
precision={'cb_force':4,'vesm_constraint':3,'pct_cb':1,'pct_vesm':1,'pct_md':1,'combined_score':2}
for col in numeric:np.testing.assert_allclose(rounded[col],ref[col].round(precision[col]) if col in precision else ref[col],rtol=1e-12,atol=1e-10,err_msg='rounded '+col)
for pos,row in md.iterrows():
    assert full.loc[pos,'MD_Roles']==' | '.join(c for c in cats if row[c]=='Y')
consensus=pd.read_csv(tables/'three_method_consensus_residues.csv')
assert consensus.position.is_unique and set(consensus.position)==set(ref.index[ref.evidence_count.eq(3)])
observed=ref.evidence_count.value_counts().sort_index().tolist()
assert observed==[267,606,412,83]
print(json.dumps({'status':'PASS','unique_positions':len(ref),'MD_category_counts':[84,54,46,90,89],
 'MD_union':311,'multi_category_key_residues':52,'evidence_counts_0_1_2_3':observed,
 'consensus_residues':len(consensus),'rank_validation':'independent pandas average-tie ranks',
 'raw_inputs_modified':False},indent=2))
