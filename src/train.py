"""Rolling-origin tuning, retrospective backtests and deployment refit. Run from project root."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json
import numpy as np,pandas as pd,joblib
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,roc_auc_score,precision_score,recall_score,f1_score,confusion_matrix
from src.forecast import *

def metric(y,s,t,phase,model,horizon='All'):
    a=s>=t;tn,fp,fn,tp=confusion_matrix(y,a,labels=[0,1]).ravel()
    return dict(phase=phase,model=model,horizon=horizon,n=len(y),threshold=t,AP=average_precision_score(y,s),AUC=roc_auc_score(y,s) if len(np.unique(y))==2 else np.nan,precision=precision_score(y,a,zero_division=0),recall=recall_score(y,a,zero_division=0),F1=f1_score(y,a,zero_division=0),FPR=fp/max(1,fp+tn),TP=tp,FP=fp,FN=fn,TN=tn,prevalence=np.mean(y))

def choose_threshold(y,s):
    # Precision-first is not imposed when the data cannot support it. Constrain the validation FPR.
    y=np.asarray(y,dtype=bool);s=np.asarray(s);order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order];ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1];tp=np.cumsum(yy)[ends];fp=ends+1-tp;ok=(fp/max(1,(~y).sum())<=.1)&(ends+1>=30)
    if not ok.any():return float(np.nextafter(ss.max(),np.inf))
    ids=np.flatnonzero(ok);j=ids[np.argmax(tp[ids])];return float(ss[ends[j]])

AUTO='Automatic (by horizon)'
HGB='Histogram Gradient Boosting'
BASE=['Persistence','Seasonal Naive']
BLENDS={'HGB blend 25%':.25,'HGB blend 50%':.50,'HGB blend 75%':.75}
def blend(p):
    for name,w in BLENDS.items():p[name]=p.base*np.exp(w*np.log(p[HGB]/p.base))
VAL=['2020-12-01','2021-06-01','2021-12-01','2022-06-01']
TEST=['2022-12-01','2023-06-01','2023-12-01','2024-06-01']
PC={'Ridge 1':('ridge',1),'Ridge 10':('ridge',10),'HGB small':('hgb',7,80,.06,2,50),'HGB standard':('hgb',15,60,.1,2,20),'HGB flexible':('hgb',31,100,.06,5,50)}
RC={'Logistic .1':('logistic',.1),'Logistic 1':('logistic',1),**{k:v for k,v in PC.items() if k.startswith('HGB')}}
SETS={'Momentum + horizon':['mom1','mom3','mom6','mom12','h','h2','mom6_missing','mom12_missing'],'+ volatility & trend':['mom1','mom3','mom6','mom12','vol3','trend','history','h','h2','trend_h','mom6_missing','mom12_missing'],'Full':FEATURES}
def fit_one(p,c,risk=False,cols=FEATURES,cap=90000):
    p=p.dropna(subset=['shock']) if risk else p
    if len(p)>cap:p=p.sample(cap,random_state=42)
    k,*v=c
    if k=='ridge':m=make_pipeline(StandardScaler(),Ridge(alpha=v[0]))
    elif k=='logistic':m=make_pipeline(StandardScaler(),LogisticRegression(C=v[0],max_iter=300))
    else:
        a,b,lr,l2,leaf=v;m=(HistGradientBoostingClassifier if risk else HistGradientBoostingRegressor)(max_leaf_nodes=a,max_iter=b,learning_rate=lr,l2_regularization=l2,min_samples_leaf=leaf,random_state=42,early_stopping=False)
    return m.fit(p[cols],p.shock.astype(int) if risk else p.y.clip(-3,3))
def groups(p):return horizon_group(p.h)
def parts(p):return [('All',p)]+[(str(h),p[p.h==h]) for h in [1,3,6,12,24,36]]+[(g,p[groups(p)==g]) for g in GROUPS]
def report(p,names,phase):
    return [dict(phase=phase,model=n,horizon=h,n=len(z),MAPE=np.abs(z[n]/z.actual-1).mean()*100,RMSLE=np.sqrt(np.mean(np.log(z[n]/z.actual)**2))) for n in names for h,z in parts(p) if len(z)]
def thresholds(p,n):
    q=p.dropna(subset=['shock']);return {g:choose_threshold(z.shock,z[n]) if len(z)>=100 and z.shock.nunique()==2 else choose_threshold(q.shock,q[n]) for g in GROUPS for z in [q[groups(q)==g]]}
def tv(p,t):return pd.Series(groups(p)).map(t).to_numpy()
def risk_report(p,names,ts,phase):
    q=p.dropna(subset=['shock']);out=[]
    for n in names:
        for h,z in parts(q):
            if len(z):
                r=metric(z.shock.to_numpy(),z[n].to_numpy(),tv(z,ts[n]),phase,n,h);r['threshold']=ts[n].get(h,np.nan);out.append(r)
    return out

def bands(p,names):
    return {n:{g:[float(r.quantile(.1)),float(r.quantile(.9))] for g in GROUPS for z in [p[groups(p)==g]] for r in [np.log(z.actual/z[n])]} for n in names}
def evaldata(d,o,end):
    p=pairs(d,o,False);p=p[p.target_m<=end].copy();p['Persistence']=p.base;p['Seasonal Naive']=p.seasonal;return p
def auto(p,policy):
    p[AUTO]=np.nan
    for g,n in policy.items():
        mask=groups(p)==g;p.loc[mask,AUTO]=p.loc[mask,n]

def main():
    from sklearn.inspection import permutation_importance
    d=prepare(pd.read_csv(ROOT/'data/raw/indonesia.csv.gz',low_memory=False));d.to_csv(ROOT/'data/processed/monthly.csv.gz',index=False,compression='gzip');vals=[];aud=[];abl=[];sampling=[]
    for o in VAL:
        tr=pairs(d,o);v=evaldata(d,o,2022*12+12);print('validation',o,len(tr),len(v),flush=True);aud.append(dict(phase='Validation',origin=o,training_pairs=len(tr),evaluation_pairs=len(v)))
        for n,c in PC.items():v['p:'+n]=predict_price(fit_one(tr,c),v[FEATURES],v.base)
        for n,c in RC.items():v['r:'+n]=fit_one(tr,c,True).predict_proba(v[FEATURES])[:,1]
        q=v.dropna(subset=['shock'])
        for label,cols in SETS.items():
            pred=predict_price(fit_one(tr,PC['HGB standard'],cols=cols),v[cols],v.base) if label!='Full' else v['p:HGB standard'];score=fit_one(tr,RC['HGB standard'],True,cols).predict_proba(q[cols])[:,1] if label!='Full' else q['r:HGB standard'];abl.append(dict(origin=o,features=label,n_features=len(cols),n=len(v),MAPE=np.abs(pred/v.actual-1).mean()*100,AP=average_precision_score(q.shock,score)))
        if o==VAL[-1]:
            for cap in [30000,90000]:
                m=fit_one(tr,PC['HGB standard'],cap=cap);r=fit_one(tr,RC['HGB standard'],True,cap=cap);sampling.append(dict(origin=o,pair_cap=cap,available_pairs=len(tr),MAPE=np.abs(predict_price(m,v[FEATURES],v.base)/v.actual-1).mean()*100,AP=average_precision_score(q.shock,r.predict_proba(q[FEATURES])[:,1])))
        vals.append(v)
    v=pd.concat(vals,ignore_index=True);pn=['p:'+n for n in PC];rn=['r:'+n for n in RC];tm=pd.DataFrame(report(v,BASE+pn,'Validation pooled'));rt={n:thresholds(v,n) for n in rn};rm=pd.DataFrame(risk_report(v,rn,rt,'Validation pooled'));tm.to_csv(ROOT/'results/tuning_price.csv',index=False);rm.to_csv(ROOT/'results/tuning_risk.csv',index=False);pd.DataFrame(abl).to_csv(ROOT/'results/ablation.csv',index=False);pd.DataFrame(sampling).to_csv(ROOT/'results/sampling_sensitivity.csv',index=False)
    bp=lambda prefix:tm[(tm.horizon=='All')&tm.model.str.startswith('p:'+prefix)].sort_values('MAPE').iloc[0].model[2:]
    br=lambda prefix:rm[(rm.horizon=='All')&rm.model.str.startswith('r:'+prefix)].sort_values(['recall','AP'],ascending=False).iloc[0].model[2:]
    psel={'Ridge':bp('Ridge'),HGB:bp('HGB')};rsel={'Logistic Regression':br('Logistic'),HGB:br('HGB')};pc={n:PC[k] for n,k in psel.items()};rc={n:RC[k] for n,k in rsel.items()}
    for n,k in psel.items():v[n]=v['p:'+k]
    for n,k in rsel.items():v['score:'+n]=v['r:'+k]
    blend(v);names=BASE+list(pc)+list(BLENDS);rn=['score:'+n for n in rc];rep=pd.DataFrame(report(v,names,'Validation pooled'));policy={g:rep[rep.horizon==g].sort_values('MAPE').iloc[0].model for g in GROUPS};auto(v,policy);ts={n:thresholds(v,n) for n in rn};rr=risk_report(v,rn,ts,'Validation pooled');selected=max([x for x in rr if x['horizon']=='All'],key=lambda x:(x['recall'],x['AP']))['model'];b=bands(v,names+[AUTO]);print('selection',psel,rsel,policy,selected,ts,flush=True)
    tests=[];folds=[];riskfolds=[]
    for o in TEST:
        tr=pairs(d,o);te=evaldata(d,o,2025*12+12);print('test',o,len(tr),len(te),flush=True);aud.append(dict(phase='Retrospective test',origin=o,training_pairs=len(tr),evaluation_pairs=len(te)))
        for n,c in pc.items():te[n]=predict_price(fit_one(tr,c),te[FEATURES],te.base)
        for n,c in rc.items():te['score:'+n]=fit_one(tr,c,True).predict_proba(te[FEATURES])[:,1]
        blend(te);auto(te,policy);folds+=report(te,names+[AUTO],o);riskfolds+=risk_report(te,rn,ts,o);tests.append(te)
    pd.DataFrame(riskfolds).to_csv(ROOT/'results/rolling_risk.csv',index=False);te=pd.concat(tests,ignore_index=True);pd.DataFrame(report(v,names+[AUTO],'Validation pooled')+report(te,names+[AUTO],'Test 2023–2025')).to_csv(ROOT/'results/forecast_metrics.csv',index=False);pd.DataFrame(folds).to_csv(ROOT/'results/rolling_origin.csv',index=False);rr+=risk_report(te,rn,ts,'Test 2023–2025');rf=pd.DataFrame(rr);rf.model=rf.model.str.removeprefix('score:');rf.to_csv(ROOT/'results/shock_metrics.csv',index=False)
    te['prediction']=te[AUTO];te['risk_score']=te[selected];te['alert']=te.risk_score>=tv(te,ts[selected]);te['threshold']=tv(te,ts[selected]);te[['series','origin_m','target_m','h','actual','prediction','Ridge',HGB,'base','seasonal','shock','risk_score','alert','threshold','province','commodity']].to_csv(ROOT/'results/test_predictions.csv.gz',index=False,compression='gzip')
    coverage=[]
    for n in names+[AUTO]:
        for g in GROUPS:
            z=te[groups(te)==g];lo,hi=b[n][g];e=np.log(z.actual/z[n]);coverage.append(dict(model=n,horizon=g,n=len(z),coverage=((e>=lo)&(e<=hi)).mean(),log_error_q10=lo,log_error_q90=hi))
    pd.DataFrame(coverage).to_csv(ROOT/'results/interval_coverage.csv',index=False)
    seg=[]
    for col in ['commodity','province']:
        for k,z in te.groupby(col):
            q=z.dropna(subset=['shock']);row=metric(q.shock.to_numpy(),q.risk_score.to_numpy(),tv(q,ts[selected]),'test',selected) if len(q)>1 and q.shock.nunique()==2 else {};seg.append(dict(segment=col,value=k,n=len(z),MAPE=np.abs(z.prediction/z.actual-1).mean()*100,**{k:row.get(k,np.nan) for k in ['precision','recall','FPR','AP']}))
    pd.DataFrame(seg).to_csv(ROOT/'results/segment_metrics.csv',index=False)
    # Update deployment using OOF historical predictions; no independent post-2025 test exists.
    combined=pd.concat([v,te],ignore_index=True);dt=thresholds(combined,selected);db=bands(combined,names+[AUTO]);pd.DataFrame([dict(horizon=g,validation_threshold=ts[selected][g],deployment_oof_threshold=dt[g],price_model=policy[g],validation_n=int((groups(v)==g).sum()),deployment_oof_n=int((groups(combined)==g).sum())) for g in GROUPS]).to_csv(ROOT/'results/deployment_policy.csv',index=False)
    final=pairs(d,'2025-12-01');aud.append(dict(phase='Deployment refit',origin='2025-12-01',training_pairs=len(final),evaluation_pairs=0));pd.DataFrame(aud).to_csv(ROOT/'results/split_audit.csv',index=False);models={n:fit_one(final,c) for n,c in pc.items()};risk=fit_one(final,rc[selected[6:]],True)
    tr=pairs(d,VAL[-1]);z=evaldata(d,VAL[-1],2022*12+12).dropna(subset=['shock']);z=z.sample(min(5000,len(z)),random_state=42);r=fit_one(tr,rc[selected[6:]],True);imp=permutation_importance(r,z[FEATURES],z.shock.astype(int),scoring='average_precision',n_repeats=3,random_state=42,n_jobs=1);pd.DataFrame({'feature':FEATURES,'AP_drop':imp.importances_mean,'std':imp.importances_std}).sort_values('AP_drop',ascending=False).to_csv(ROOT/'results/importance.csv',index=False)
    joblib.dump(dict(price_models=models,risk_model=risk,risk_selected=selected[6:],selected=AUTO,thresholds=dt,validation_thresholds=ts[selected],price_policy=policy,price_configs=pc,risk_configs=rc,blends=BLENDS,policy='Validation horizon FPR <=10%; deployment thresholds adapted from OOF through 2025',train_end='2025-12',features=FEATURES,bands=db),ROOT/'models/forecast.joblib',compress=3)
    bundle=joblib.load(ROOT/'models/forecast.joblib');origin,path=forecast_path(d,868,'2028-01-31',bundle);path.to_csv(ROOT/'results/example_jan2028.csv',index=False);origin.to_frame().T.to_csv(ROOT/'results/example_origin.csv',index=False)
    print('done',flush=True)
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
