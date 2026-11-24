"""Fixed-origin temporal backtests and deployment refit. Run from project root."""
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

def fit(p):
    if len(p)>90000:p=p.sample(90000,random_state=42)
    x=p[FEATURES];y=p.y.clip(-3,3)
    models={'Ridge':make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(x,y),'Histogram Gradient Boosting':HistGradientBoostingRegressor(max_iter=60,max_leaf_nodes=15,l2_regularization=2,random_state=42).fit(x,y)}
    q=p.dropna(subset=['shock']);y=q.shock.astype(int);x=q[FEATURES]
    risks={'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=300,C=.3)).fit(x,y),'Histogram Gradient Boosting':HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=15,l2_regularization=2,random_state=42).fit(x,y)}
    return models,risks

def evaluate(p,models,phase):
    out=[]
    for name,yp in [('Persistence',p.base),('Seasonal Naive',p.seasonal)]+[(n,predict_price(m,p[FEATURES],p.base)) for n,m in models.items()]:
        e=np.abs(np.asarray(yp)/p.actual-1);q=p.assign(error=e,logerr=np.log(np.asarray(yp)/p.actual)**2)
        for label,part in [('All',q)]+[(str(h),q[q.h==h]) for h in [1,3,6,12,24,36]]:
            if len(part):out.append(dict(phase=phase,model=name,horizon=label,n=len(part),MAPE=part.error.mean()*100,RMSLE=np.sqrt(part.logerr.mean())))
    return out

def metric(y,s,t,phase,model,horizon='All'):
    a=s>=t;tn,fp,fn,tp=confusion_matrix(y,a,labels=[0,1]).ravel()
    return dict(phase=phase,model=model,horizon=horizon,n=len(y),threshold=t,AP=average_precision_score(y,s),AUC=roc_auc_score(y,s) if len(np.unique(y))==2 else np.nan,precision=precision_score(y,a,zero_division=0),recall=recall_score(y,a,zero_division=0),F1=f1_score(y,a,zero_division=0),FPR=fp/max(1,fp+tn),TP=tp,FP=fp,FN=fn,TN=tn,prevalence=np.mean(y))

def choose_threshold(y,s):
    # Precision-first is not imposed when the data cannot support it. Constrain the validation FPR.
    y=np.asarray(y,dtype=bool);s=np.asarray(s);order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order];ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1];tp=np.cumsum(yy)[ends];fp=ends+1-tp;ok=(fp/max(1,(~y).sum())<=.1)&(ends+1>=30)
    if not ok.any():return float(np.nextafter(ss.max(),np.inf))
    ids=np.flatnonzero(ok);j=ids[np.argmax(tp[ids])];return float(ss[ends[j]])

def main():
    d=prepare(pd.read_csv(ROOT/'data/raw/indonesia.csv.gz',low_memory=False));d.to_csv(ROOT/'data/processed/monthly.csv.gz',index=False,compression='gzip');print('monthly',len(d),'series',d.series.nunique(),flush=True)
    rows=[];validation=[];residuals=[];counts=[]
    for origin in ['2020-12-01','2021-12-01']:
        tr=pairs(d,origin);va=pairs(d,origin,False);va=va[va.target_m<=2022*12+12];print('validation',origin,len(tr),len(va),flush=True);counts.append({'origin':origin,'training_pairs':len(tr),'evaluation_pairs':len(va)});models,risks=fit(tr);rows+=evaluate(va,models,origin)
        q=va.dropna(subset=['shock']);v=q[['shock','h']].copy();v['origin']=origin
        for name,r in risks.items():v[name]=r.predict_proba(q[FEATURES])[:,1]
        validation.append(v)
        for name,m in models.items():residuals.append(pd.DataFrame({'model':name,'h':va.h,'residual':np.log(va.actual/predict_price(m,va[FEATURES],va.base))}))
    v=pd.DataFrame(rows);sel=v[(v.horizon=='All')&v.model.isin(['Ridge','Histogram Gradient Boosting'])].groupby('model').apply(lambda z:np.average(z.MAPE,weights=z.n),include_groups=False).idxmin();r=pd.concat(validation);risk_names=['Logistic Regression','Histogram Gradient Boosting'];risk_rows=[];thresholds={}
    for name in risk_names:
        thresholds[name]=choose_threshold(r.shock,r[name]);risk_rows.append(metric(r.shock.to_numpy(),r[name].to_numpy(),thresholds[name],'Validation pooled',name))
    # Pick the highest recall subject to a <=10% false-positive rate on validation only.
    risk_selected=max(risk_rows,key=lambda z:(z['recall'],z['AP']))['model'];threshold=thresholds[risk_selected]
    # Empirical 80% error bands from validation, grouped by horizon; last bucket is reused for h>24.
    rr=pd.concat(residuals);rr['bucket']=pd.cut(rr.h,[0,3,6,12,24,36],labels=['1–3','4–6','7–12','13–24','25–36']);bands={}
    for name in ['Ridge','Histogram Gradient Boosting']:
        g=rr[rr.model==name];band={}
        for label,z in g.groupby('bucket',observed=True):band[str(label)]=[float(z.residual.quantile(.1)),float(z.residual.quantile(.9))]
        band['25–36']=band.get('25–36',band['13–24']);bands[name]=band
    tr=pairs(d,'2022-12-01');te=pairs(d,'2022-12-01',False);counts.append({'origin':'2022-12-01','training_pairs':len(tr),'evaluation_pairs':len(te)});print('test',len(tr),len(te),'selected',sel,'risk',risk_selected,'threshold',threshold,flush=True);models,risks=fit(tr);rows+=evaluate(te,models,'Test 2023–2025');pd.DataFrame(rows).to_csv(ROOT/'results/forecast_metrics.csv',index=False)
    q=te.dropna(subset=['shock']);y=q.shock.to_numpy()
    for name,m in risks.items():
        score=m.predict_proba(q[FEATURES])[:,1];risk_rows.append(metric(y,score,thresholds[name],'Test 2023–2025',name));
        for h in [1,3,6,12,24,36]:
            mask=q.h.to_numpy()==h
            if mask.any():risk_rows.append(metric(y[mask],score[mask],thresholds[name],'Test 2023–2025',name,str(h)))
    pd.DataFrame(risk_rows).to_csv(ROOT/'results/shock_metrics.csv',index=False)
    for name,m in models.items():te[name]=predict_price(m,te[FEATURES],te.base)
    te['prediction']=te[sel];te['risk_score']=risks[risk_selected].predict_proba(te[FEATURES])[:,1];te['alert']=te.risk_score>=threshold;te[['series','origin_m','target_m','h','actual','prediction','Ridge','Histogram Gradient Boosting','base','seasonal','shock','risk_score','alert','province','commodity']].to_csv(ROOT/'results/test_predictions.csv.gz',index=False,compression='gzip')
    band_rows=[]
    for name,m in models.items():
        residual=np.log(te.actual/te[name]);bucket=pd.cut(te.h,[0,3,6,12,24,36],labels=['1–3','4–6','7–12','13–24','25–36'])
        for label,z in te.groupby(bucket,observed=True):
            lo,hi=bands[name][str(label)];v=residual.loc[z.index];band_rows.append({'model':name,'horizon':str(label),'n':len(v),'coverage':((v>=lo)&(v<=hi)).mean(),'log_error_q10':lo,'log_error_q90':hi,'calibration_note':'13–24 reused' if str(label)=='25–36' else 'validation quantiles'})
    pd.DataFrame(band_rows).to_csv(ROOT/'results/interval_coverage.csv',index=False)
    final=pairs(d,'2025-12-01');counts.append({'origin':'2025-12-01','training_pairs':len(final),'evaluation_pairs':0});pd.DataFrame(counts).to_csv(ROOT/'results/split_audit.csv',index=False);print('deployment',len(final),flush=True);models,risks=fit(final);joblib.dump({'price_models':models,'risk_model':risks[risk_selected],'risk_selected':risk_selected,'selected':sel,'threshold':threshold,'policy':'Maximum recall at validation FPR <= 10%, minimum 30 alerts','train_end':'2025-12','features':FEATURES,'bands':bands},ROOT/'models/forecast.joblib',compress=3);print('done',flush=True)
if __name__=='__main__':main()
