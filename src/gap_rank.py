"""Development-only neighborhood skill ranking, evaluated retrospectively."""
import numpy as np,pandas as pd,joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone
from .core import ROOT,load,recommend
from .train import profiles
from .enhance import masked

def build(d,vocab):
 ml=MultiLabelBinarizer(classes=vocab);ml.fit([[]]);return {c:ml.transform(g.skills).astype('float32') for c,g in d.groupby('career')}
def values(a,ix):
 f=a.mean(0); count=a[:,ix].sum(1); w=(count/np.maximum(a.sum(1)+len(ix)-count,1))**2; n=w@a/max(w.sum(),1e-9) if w.sum() else f; c=(a[:,ix].T@a/np.maximum(a[:,ix].sum(0)[:,None],1)).mean(0) if ix else f
 return [f,c,n,.5*n+.5*c,.7*n+.3*f]
POLICIES=['frequency','conditional','neighbor','blend','neighbor_frequency']
def ranking(b,career,known):
 v=b['vocabulary'];a=b['gap_matrix'][career];ix=[i for i,s in enumerate(v) if s in known];f=a.mean(0);s=values(a,ix)[POLICIES.index(b['gap_v3_policy'])];r=pd.DataFrame({'Skill':v,'Demand share':f,'Learning priority score':s});return r[(r['Demand share']>0)&~r.Skill.isin(known)].sort_values(['Learning priority score','Skill'],ascending=[False,True],ignore_index=True)
def run():
 b=load();d=pd.read_csv(ROOT/'data/processed/jobs.csv');d.skills=d.skills.str.split('|');dev=d[d.split=='development'].reset_index(drop=True);test=d[d.split=='test'];v=b['vocabulary'];hits=[[] for _ in POLICIES]
 for a,z in StratifiedGroupKFold(4,shuffle=True,random_state=43).split(dev,dev.career,dev.group):
  fit,val=dev.iloc[a],dev.iloc[z];tmp=dict(b);tmp['profiles']=profiles(fit);mat=build(fit,v);tmp['models']={b['base']:clone(b['models'][b['base']]).fit(fit.x,fit.career)}
  for r,h,known in masked(val):
   c=recommend(tmp,known).iloc[0].Career;ix=[i for i,s in enumerate(v) if s in known];eligible=(mat[c].sum(0)>0);eligible[ix]=False
   for j,p in enumerate(values(mat[c],ix)):
    order=np.argsort(-np.where(eligible,p,-np.inf),kind='stable');hits[j].append(h in [v[i] for i in order[:5] if eligible[i]])
  print('fold done',flush=True)
 b['gap_v3_policy']=POLICIES[int(np.argmax([np.mean(h) for h in hits]))];b['gap_matrix']=build(dev,v);pd.DataFrame([{'policy':k,'cv_end_to_end_hit5':np.mean(h),'n':len(h)} for k,h in zip(POLICIES,hits)]).to_csv(ROOT/'results/gap_v3_tuning.csv',index=False)
 rows=[]
 for r,h,known in masked(test):
  c=recommend(b,known).iloc[0].Career;ls=ranking(b,c,known).Skill.tolist();rows.append({'id':r.id,'hit5':int(h in ls[:5]),'hit10':int(h in ls[:10]),'reciprocal_rank':1/(ls.index(h)+1) if h in ls else 0})
 pd.DataFrame(rows).to_csv(ROOT/'results/gap_v3_evaluation.csv',index=False);joblib.dump(b,ROOT/'models/bundle.joblib',compress=3);print(b['gap_v3_policy'],pd.DataFrame(rows).mean(),flush=True)
if __name__=='__main__':run()
