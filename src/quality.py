"""Retrospective sensitivity audit; not a new independent test."""
import numpy as np,pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score,f1_score
from .core import ROOT,load

def run():
 d=pd.read_csv(ROOT/'data/processed/jobs.csv'); b=load(); tr=d[d.split=='development']; te=d[d.split=='test']; ml=MultiLabelBinarizer(classes=b['vocabulary']); ml.fit([[]]); a=ml.transform(tr.skills.str.split('|')).astype('int32'); z=ml.transform(te.skills.str.split('|')).astype('int32'); inter=z@a.T; union=z.sum(1)[:,None]+a.sum(1)[None,:]-inter; sim=inter/np.maximum(union,1); mx=sim.max(1); pred=pd.read_csv(ROOT/'results/test_predictions.csv').set_index('id').loc[te.id,b['best']].to_numpy(); y=te.career.to_numpy(); rows=[]
 for cut in [1.01,1,.9,.8,.7]:
  mask=mx<cut; rows.append({'exclude_similarity_at_or_above':cut,'retained_test':int(mask.sum()),'excluded_test':int((~mask).sum()),'accuracy':accuracy_score(y[mask],pred[mask]),'macro_f1':f1_score(y[mask],pred[mask],labels=sorted(d.career.unique()),average='macro',zero_division=0)})
 pd.DataFrame(rows).to_csv(ROOT/'results/similarity_audit.csv',index=False)
 amb=d.groupby('group').career.nunique(); d['ambiguous']=d.group.map(amb)>1; d.groupby(['source','career']).agg(profiles=('id','size'),ambiguous_profiles=('ambiguous','sum')).reset_index().to_csv(ROOT/'results/label_audit.csv',index=False)
 # Stratified deterministic review sample, actual human review must be filled separately.
 sample=d.groupby(['source','career'],group_keys=False).sample(frac=1,random_state=42).groupby(['source','career']).head(8)[['id','source','title','career','skills']].copy();sample['reviewed_label']='';sample['review_notes']='';sample.to_csv(ROOT/'results/label_review_template.csv',index=False)
 print(pd.DataFrame(rows).to_string(index=False));print('ambiguous',d.ambiguous.sum(),'review sample',len(sample))
if __name__=='__main__':run()
