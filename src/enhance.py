"""Fit descriptive specialization profiles and choose skill recovery policy on grouped development folds."""
from collections import Counter
from itertools import combinations
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import StratifiedGroupKFold
from .core import ROOT,load,specialization,learning_rank,recommend,features,score,overlap
from .train import profiles,ovmat,pipe

def stats(d):
    co={};counts={}
    for c,g in d.groupby('career'):
        counts[c]=dict(Counter(s for ss in g.skills for s in ss));co[c]=dict(Counter(pair for ss in g.skills for pair in combinations(sorted(set(ss)),2)))
    return co,counts

def masked(d):
    for r in d.itertuples():
        if len(r.skills)>=2:
            h=str(np.random.default_rng(42+r.id).choice(r.skills));yield r,h,sorted(set(r.skills)-{h})
def run():
    b=load();d=pd.read_csv(ROOT/'data/processed/jobs.csv');d.skills=d.skills.str.split('|');dev=d[d.split=='development'].reset_index(drop=True);test=d[d.split=='test'];co,counts=stats(dev);b['cooccurrence']=co;b['skill_counts']=counts
    sd=dev.copy();sd['specialization']=[specialization(c,t) for c,t in zip(sd.career,sd.title)];specs={};sc=[]
    for (c,k),g in sd.dropna(subset=['specialization']).groupby(['career','specialization']):
        if len(g)>=10:
            specs.setdefault(c,{})[k]={s:n/len(g) for s,n in Counter(s for ss in g.skills for s in ss).most_common(25)};sc.append({'field':c,'specialization':k,'development_profiles':len(g)})
    b['specializations']=specs;pd.DataFrame(sc).to_csv(ROOT/'results/specialization_support.csv',index=False)
    # Evaluate policies using known field during selection, then evaluate end-to-end on holdout.
    rows=[];opts=['frequency','0.25','0.5','0.75','1.0'];hits={k:[] for k in opts}
    for a,v in StratifiedGroupKFold(4,shuffle=True,random_state=43).split(dev,dev.career,dev.group):
        fit,val=dev.iloc[a],dev.iloc[v];c,n=stats(fit);tmp={'profiles':profiles(fit),'cooccurrence':c,'skill_counts':n};
        for r,h,known in masked(val):
            for policy in opts:
                tmp['gap_policy']=policy;hits[policy].append(int(h in learning_rank(tmp,r.career,known).Skill.head(5).tolist()))
    b['gap_policy']=max(opts,key=lambda k:np.mean(hits[k]));pd.DataFrame([{'policy':k,'cv_oracle_hit_at5':np.mean(v),'n':len(v)} for k,v in hits.items()]).to_csv(ROOT/'results/gap_tuning.csv',index=False)
    # Same masked profiles as original test; use only known input and selected field.
    globalcounts=Counter(s for ss in dev.skills for s in ss);gaps=[]
    for r,h,known in masked(test):
        rec=recommend(b,known)
        if rec.empty:continue
        c=rec.iloc[0].Career;gaps.append({'id':r.id,'hidden':h,'actual_field':r.career,'predicted_field':c,'global_hit':int(h in [s for s,n in globalcounts.most_common() if s not in known][:5]),'oracle_hit':int(h in learning_rank(b,r.career,known).Skill.head(5).tolist()),'end_to_end_hit':int(h in learning_rank(b,c,known).Skill.head(5).tolist()),'frequency_end_to_end_hit':int(h in [s for s in b['profiles'][c] if s not in known][:5])})
    pd.DataFrame(gaps).to_csv(ROOT/'results/gap_evaluation.csv',index=False);joblib.dump(b,ROOT/'models/bundle.joblib',compress=3);print('gap policy',b['gap_policy'],pd.DataFrame(gaps).filter(like='hit').mean().to_dict(),flush=True)
if __name__=='__main__':run()
