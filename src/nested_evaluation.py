"""Nested grouped CV on local profiles; historical data, not a fresh external test."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
from .core import ROOT, score
from .train import pipe, profiles, ovmat, measure


def run():
    d = pd.read_csv(ROOT / 'data/processed/jobs.csv')
    d = d[d.split.eq('development')].reset_index(drop=True)
    d['skills'] = d.skills.str.split('|')
    classes = np.array(sorted(d.career.unique()))
    configs = [(f'NB alpha={a}', MultinomialNB(alpha=a)) for a in [.1,.5,1]] + [(f'SVM C={c}', LinearSVC(C=c,class_weight='balanced')) for c in [.3,1,3]]
    outer = StratifiedGroupKFold(4, shuffle=True, random_state=73)
    pred = np.zeros((len(d), len(classes)))
    rows = []
    for fold, (a,b) in enumerate(outer.split(d,d.career,d.group)):
        tr, te = d.iloc[a].reset_index(drop=True), d.iloc[b]
        assert not set(tr.group)&set(te.group)
        inner = list(StratifiedGroupKFold(3,shuffle=True,random_state=81+fold).split(tr,tr.career,tr.group))
        overlap = np.zeros((len(tr),len(classes)))
        for i,j in inner:
            overlap[j] = ovmat(profiles(tr.iloc[i]),tr.iloc[j].skills,classes)
        best = (-1,None,None,None)
        for name,est in configs:
            p = np.zeros_like(overlap)
            for i,j in inner:
                p[j] = score(pipe(est).fit(tr.iloc[i].x,tr.iloc[i].career),tr.iloc[j].x)
            for alpha in [0,.25,.5,.75,1]:
                f = measure(tr.career,alpha*p+(1-alpha)*overlap,classes)['macro_f1']
                if f>best[0]: best = (f,name,est,alpha)
        f,name,est,alpha=best
        p=alpha*score(pipe(est).fit(tr.x,tr.career),te.x)+(1-alpha)*ovmat(profiles(tr),te.skills,classes)
        pred[b]=p
        rows.append({'outer_fold':fold+1,'selected':name,'alpha':alpha,'inner_macro_f1':f,'test_profiles':len(te),**measure(te.career,p,classes),'balanced_accuracy':balanced_accuracy_score(te.career,classes[p.argmax(1)])})
    rows.append({'outer_fold':'pooled','selected':'fold-specific','alpha':np.nan,'inner_macro_f1':np.nan,'test_profiles':len(d),**measure(d.career,pred,classes),'balanced_accuracy':balanced_accuracy_score(d.career,classes[pred.argmax(1)])})
    pd.DataFrame(rows).to_csv(ROOT/'results/nested_cv.csv',index=False)
    d[['id','career','source']].assign(prediction=classes[pred.argmax(1)]).to_csv(ROOT/'results/nested_cv_predictions.csv',index=False)


if __name__ == '__main__':
    run()
