"""Retrospective source robustness audit; no role in model selection."""
import pandas as pd,numpy as np
from sklearn.naive_bayes import MultinomialNB
from .core import ROOT,load,score
from .train import pipe,profiles,ovmat,measure

def run():
    b=load();d=pd.read_csv(ROOT/'data/processed/jobs.csv');d.skills=d.skills.str.split('|');dev=d[d.split=='development'];test=d[d.split=='test'];cls=np.array(sorted(d.career.unique()));rows=[]
    for src,t in test.groupby('source'):
        # Hold source out of development; remove any matching profile group as precaution.
        fit=dev[(dev.source!=src)&~dev.group.isin(t.group)];
        if set(fit.career)!=set(cls):continue
        m=pipe(b['models'][b['base']].steps[-1][1]).fit(fit.x,fit.career);p=score(m,t.x);p=b['alpha']*p+(1-b['alpha'])*ovmat(profiles(fit),t.skills,cls);rows.append({'held_out_source':src,'train_profiles':len(fit),'test_profiles':len(t),'test_classes':t.career.nunique(),**measure(t.career,p,cls)})
    pd.DataFrame(rows).to_csv(ROOT/'results/source_holdout.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':run()
