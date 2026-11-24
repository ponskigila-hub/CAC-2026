"""FoodShock V9: canonical seven-fold benchmark, ablations, frozen models, diagnostics.

Run `python src/run_v9.py --trials 25` from the project root (or any cwd).
The canonical input is the supplied model-ready CSV. No network is required.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, platform
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.models import (train_and_evaluate,make_bundle,split_panel,fit_point,validate_panel,
                    _fit_encoders,_transform,ALGORITHMS,DEFAULT_PARAMS)
from core.utils import (FEATURE_COLS, CAT_COLS, alert_metrics, smape, price_change, shock_event, select_threshold)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,precision_score,recall_score,average_precision_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

DATA=ROOT/'data'; FIG=ROOT/'figures'
CUTOFFS=pd.to_datetime(['2019-06-01','2020-06-01','2021-06-01','2022-06-01','2023-06-01','2024-06-01','2025-06-01'])
SEED=42
FEATURE_GROUPS={
    'Lag + current price':FEATURE_COLS[:6],
    '+ Rolling / momentum / seasonality':FEATURE_COLS[:18],
    '+ FX':FEATURE_COLS[:21],
    'Full FoodShock (+ ASEAN spillover)':FEATURE_COLS,
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def dm_panel(y,p,b,months,max_lag=3):
    """HAC test on month-mean squared loss difference, not IID panel rows."""
    # Explicit column indexing avoids conflict with DataFrame.diff method.
    d=pd.DataFrame({'month':np.asarray(months),'loss_diff':(np.asarray(y)-np.asarray(p))**2-(np.asarray(y)-np.asarray(b))**2}).groupby('month')['loss_diff'].mean().sort_index().to_numpy()
    n=len(d)
    if n<8: return np.nan,np.nan
    mu=d.mean(); z=d-mu; L=min(max_lag,n-1); var=float(z@z/n)
    for k in range(1,L+1): var+=2*(1-k/(L+1))*float(z[k:]@z[:-k]/n)
    if var<=0: return np.nan,np.nan
    stat=float(mu/np.sqrt(var/n)); return stat,float(2*norm.sf(abs(stat)))


def regression(g):
    y,p,b=g.actual.to_numpy(),g.predicted.to_numpy(),g.current.to_numpy()
    rmse=float(np.sqrt(mean_squared_error(y,p))); base=float(np.sqrt(mean_squared_error(y,b)))
    errors=pd.DataFrame({'country':g.countryiso3.to_numpy(),'ae':np.abs(y-p)})
    dm,dp=dm_panel(y,p,b,g.date.dt.to_period('M').astype(str))
    return dict(n=len(g),MAE_micro=mean_absolute_error(y,p),RMSE_micro=rmse,sMAPE_micro=smape(y,p),R2_micro=r2_score(y,p) if len(g)>1 else np.nan,R2_delta=r2_score(y-b,p-b) if len(g)>1 else np.nan,MAE_macro=float(errors.groupby('country').ae.mean().mean()),Persistence_MAE=mean_absolute_error(y,b),Persistence_RMSE=base,SS_RMSE=1-rmse/base if base else np.nan,DM_stat=dm,DM_pvalue=dp)


def month_bootstrap(y,pred,score,dates,n_boot=200):
    """Resample entire forecast-origin months, preserving cross-sectional dependence."""
    rng=np.random.default_rng(SEED); y=np.asarray(y,int); pred=np.asarray(pred,bool); score=None if score is None else np.asarray(score,float)
    ym=pd.to_datetime(dates).to_period('M').astype(str) if not isinstance(dates,pd.Series) else dates.dt.to_period('M').astype(str).to_numpy()
    groups=[np.flatnonzero(np.asarray(ym)==m) for m in sorted(set(ym))]; values=[]
    for _ in range(n_boot):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]); yy=y[ix]; pp=pred[ix]
        values.append([precision_score(yy,pp,zero_division=0),recall_score(yy,pp,zero_division=0),average_precision_score(yy,score[ix]) if score is not None and len(np.unique(yy))==2 else np.nan,roc_auc_score(yy,score[ix]) if score is not None and len(np.unique(yy))==2 else np.nan])
    a=np.asarray(values)
    return {f'{name}_CI95':f'[{np.nanquantile(a[:,i],.025):.4f}, {np.nanquantile(a[:,i],.975):.4f}]' if np.isfinite(a[:,i]).any() else 'n/a' for i,name in enumerate(['Precision','Recall','PR_AUC','ROC_AUC'])}


def regression_bootstrap(g,n_boot=200):
    rng=np.random.default_rng(SEED); ym=g.date.dt.to_period('M').astype(str).to_numpy(); groups=[np.flatnonzero(ym==m) for m in sorted(set(ym))]
    a=(g.actual-g.predicted).to_numpy()**2; b=(g.actual-g.current).to_numpy()**2
    vals=[]
    for _ in range(n_boot):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]); vals.append(1-np.sqrt(a[ix].mean()/b[ix].mean()) if b[ix].mean()>0 else np.nan)
    return [float(x) for x in np.nanquantile(vals,[.025,.975])]


def _frame(o,alg,fold):
    t=o['test_rows'][['series_id','date','countryiso3','market','commodity_group','pricetype']].copy()
    return t.assign(model=alg,fold=fold,actual=o['y_true'],predicted=o['y_pred'],current=o['y_pers'],lower=o['y_lo'],upper=o['y_hi'],score=o['shock_proba'],alert=o['alerts'] if o['threshold'] is not None else np.nan,shock=o['shock_true'],threshold=o['threshold'] if o['threshold'] is not None else np.nan)


def classification_baselines(fit,cal,test):
    """Same fit/calibration/test dates; no threshold chosen on OOS."""
    yfit=shock_event(price_change(fit.usdprice,fit.target_price_usd_t1)).astype(int)
    ycal=shock_event(price_change(cal.usdprice,cal.target_price_usd_t1)).astype(int)
    enc=_fit_encoders(fit); scaler=StandardScaler()
    Xfit=scaler.fit_transform(_transform(fit,enc)); Xcal=scaler.transform(_transform(cal,enc)); Xtest=scaler.transform(_transform(test,enc))
    lr=LogisticRegression(class_weight='balanced',max_iter=500,random_state=SEED); lr.fit(Xfit,yfit)
    pc,pt=lr.predict_proba(Xcal)[:,1],lr.predict_proba(Xtest)[:,1]; t=select_threshold(ycal,pc)
    rates=fit.assign(_shock=yfit).groupby(['countryiso3','commodity_group'])._shock.mean().to_dict(); global_rate=float(yfit.mean())
    def history(d): return np.array([rates.get((a,b),global_rate) for a,b in zip(d.countryiso3,d.commodity_group)])
    hc,ht=history(cal),history(test); h=select_threshold(ycal,hc)
    m=test.pct_change_1m.to_numpy()
    return dict(LogisticRegression=(pt,pt>=t,t),HistoricalShockRate=(ht,ht>=h,h),LaggedMomentum=(m,shock_event(m),.05))


def run(trials):
    if trials<0: raise ValueError('Trials must be >= 0.')
    path=DATA/'foodshock_asean_model_ready.csv'; df=validate_panel(pd.read_csv(path))
    DATA.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); fold_dir=DATA/'fold_models'; fold_dir.mkdir(exist_ok=True)
    rows=[]; hp=[]; fold_rows=[]; ablation=[]; base_rows=[]; model_files={}
    for cutoff in CUTOFFS:
        fold=str(cutoff.date()); end=cutoff+pd.DateOffset(months=12)
        fit,cal,test,_=split_panel(df,cutoff,end)
        for alg in ALGORITHMS:
            n=trials if alg in ALGORITHMS[:3] else 0
            o=train_and_evaluate(df,algorithm=alg,cutoff=cutoff,end=end,tune_trials=n,risk_layer=(alg=='CatBoost'),**DEFAULT_PARAMS)
            t=_frame(o,alg,fold); rows.append(t)
            hp.append(dict(fold=fold,model=alg,trials=n,**o['params']))
            fm=dict(o['metrics'],fold=fold,model=alg,fit_end=o['fit_end'],cal_start=o['cal_start'],cal_end=o['cal_end'],tune_end=o['tune_end'],test_end=o['test_end'])
            if o['threshold_selection'] is not None: fm.update({f'validation_{k}':v for k,v in o['threshold_selection'].items()},qcal=o['qcal'])
            fold_rows.append(fm)
            print(f'{fold} {alg}: RMSE={o["metrics"]["RMSE"]:.6f}',flush=True)
            if alg=='CatBoost':
                file=fold_dir/f'{fold}.joblib'; joblib.dump(make_bundle(o),file,compress=3); model_files[str(file.relative_to(DATA))]=sha(file)
                for label,features in FEATURE_GROUPS.items():
                    if features==FEATURE_COLS: p=o['y_pred']
                    else: _,_,p=fit_point(fit,test,alg,o['params'],features)
                    ablation.append(t[['date','countryiso3','actual','current']].assign(predicted=p,feature_set=label,fold=fold))
                for label,(score,pred,threshold) in classification_baselines(fit,cal,test).items():
                    base_rows.append(t[['date','shock']].assign(method=label,score=score,alert=pred,threshold=threshold,fold=fold))
    allrows=pd.concat(rows,ignore_index=True); cat=allrows[allrows.model=='CatBoost'].copy(); base=cat.assign(predicted=cat.current)
    # The official app model is a separate frozen holdout fit using the same methodology.
    o=train_and_evaluate(df,algorithm='CatBoost',tune_trials=trials,**DEFAULT_PARAMS)
    official=DATA/'v9_official_bundle.joblib'; joblib.dump(make_bundle(o),official,compress=3); model_files[official.name]=sha(official)
    holdout=_frame(o,'CatBoost',o['cutoff'])
    # Commit evaluation files from the completed run, not hand-edited benchmark values.
    allrows.to_csv(DATA/'v9_oos_predictions.csv',index=False)
    holdout.to_csv(DATA/'v9_official_holdout_predictions.csv',index=False)
    pd.DataFrame(hp).to_csv(DATA/'hyperparameter_tuning_summary.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(DATA/'v9_fold_metrics.csv',index=False)
    reg=[dict(model='Persistence (y_t)',**regression(base),SS_RMSE_CI95='[0, 0]')]
    for alg,g in allrows.groupby('model'):
        ci=regression_bootstrap(g)
        reg.append(dict(model=alg,**regression(g),SS_RMSE_CI95=f'[{ci[0]:.4f}, {ci[1]:.4f}]'))
    reg_df=pd.DataFrame(reg).sort_values('RMSE_micro'); reg_df.to_csv(DATA/'model_evaluation_results.csv',index=False)
    for col,name in [('countryiso3','country'),('commodity_group','commodity'),('pricetype','pricetype')]:
        pd.DataFrame([dict(**{col:k},**regression(g)) for k,g in cat.groupby(col)]).to_csv(DATA/f'model_eval_by_{name}.csv',index=False)
    cov=((cat.actual>=cat.lower)&(cat.actual<=cat.upper)).mean(); width=(cat.upper-cat.lower).mean()
    shock=[]
    def add(method,y,pred,score,dates,threshold,ranking):
        m=alert_metrics(y,pred,score)
        shock.append(dict(method=method,**m,Ranking_Score=ranking,Base_Rate=float(np.mean(y)),Support_Shock=int(np.sum(y)),Support_Total=len(y),Lift=m['Precision']/float(np.mean(y)) if np.mean(y)>0 else np.nan,Alerts_Per_Month=float(np.sum(pred)/dates.dt.to_period('M').nunique()),Overall_PICP_P10_P90=cov if method=='Hybrid_Operational_Alert' else np.nan,Mean_Interval_Width=width if method=='Hybrid_Operational_Alert' else np.nan,Mean_Train_Threshold=threshold,**month_bootstrap(y,pred,score,dates)))
    ch=price_change(cat.current,cat.predicted); mean_thr=cat.groupby('fold').threshold.first().mean()
    add('Hybrid_Operational_Alert',cat.shock,cat.alert.astype(bool),None,cat.date,mean_thr,'n/a (binary hybrid rule)')
    add('Stage2_Only',cat.shock,cat.score>=cat.threshold,cat.score,cat.date,mean_thr,'Stage-2 risk score')
    add('Forecast_Only',cat.shock,shock_event(ch),ch,cat.date,.05,'Forecast price change')
    for method,g in pd.concat(base_rows).groupby('method'):
        add(method,g.shock,g.alert,g.score,g.date,g.groupby('fold').threshold.first().mean(),'Lagged return' if method=='LaggedMomentum' else 'Uncalibrated score')
    shock_df=pd.DataFrame(shock); shock_df.to_csv(DATA/'model_eval_shock_classification.csv',index=False)
    ab=pd.concat(ablation,ignore_index=True); ab_rows=[dict(feature_set='Persistence (y_t)',n_numeric_features=0,**regression(base))]
    for label,features in FEATURE_GROUPS.items(): ab_rows.append(dict(feature_set=label,n_numeric_features=len(features),**regression(ab[ab.feature_set==label])))
    pd.DataFrame(ab_rows).to_csv(DATA/'v9_feature_ablation.csv',index=False)
    # Evaluate deployment feature importances on the exact frozen official CatBoost.
    from catboost import Pool
    sample=o['test_rows'].sample(n=min(1200,len(o['test_rows'])),random_state=SEED)
    X=_transform(sample,o['encoders']); shap=o['model'].get_feature_importance(Pool(X),type='ShapValues',thread_count=2)[:,:-1]
    importance=pd.DataFrame({'feature':X.columns,'mean_abs_shap':np.abs(shap).mean(axis=0)}).sort_values('mean_abs_shap',ascending=False)
    importance.to_csv(DATA/'v9_shap_importance.csv',index=False)
    generate_figures(df,cat,reg_df,shock_df,importance,pd.DataFrame(ab_rows))
    meta=dict(version='V9',seed=SEED,trials_per_tuned_model=trials,cutoffs=[str(c.date()) for c in CUTOFFS],dataset_sha256=sha(path),dataset_rows=len(df),dataset_columns=len(df.columns),oos_rows=len(cat),official_model='CatBoost residual',official_holdout_cutoff=o['cutoff'],official_fit_end=o['fit_end'],official_cal_start=o['cal_start'],official_cal_end=o['cal_end'],official_threshold=o['threshold'],official_qcal=o['qcal'],official_params=o['params'],official_holdout_metrics=o['metrics'],model_sha256=model_files,versions=dict(python=platform.python_version(),**{x:version(x) for x in ['numpy','pandas','scikit-learn','lightgbm','xgboost','catboost','scipy','matplotlib','optuna','joblib']}))
    meta['artifact_sha256']={p.name:sha(p) for p in sorted(DATA.glob('*.csv')) if p.name!='foodshock_asean_model_ready.csv'}
    (DATA/'v9_manifest.json').write_text(json.dumps(meta,indent=2,default=float,allow_nan=False)+'\n')
    # The former average-fold app threshold is no longer used; remove it from the active dataset folder.
    for name in ('app_shock_threshold.txt','optimization_notes_v7.txt','shock_threshold_justification.txt'):
        (DATA/name).unlink(missing_ok=True)
    (DATA/'shock_threshold_justification.txt').write_text('Upward shock: (next price - current price) / current price > 0.05 (numerical tolerance 1e-12). The threshold is a project-specific upper-tail policy definition, not official ALPS. Hybrid risk-score thresholds are selected separately on chronological calibration blocks with the exact frozen point and classifier; see v9_fold_metrics.csv and v9_manifest.json. Stage-2 scores are not calibrated probabilities.\n')
    print('V9 COMPLETE',flush=True)
    print(reg_df.to_string(index=False),flush=True)
    print(shock_df[['method','Precision','Recall','F1','FPR','PR_AUC','ROC_AUC']].to_string(index=False),flush=True)


def generate_figures(df,cat,reg,shock,imp,ab):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name): fig.tight_layout(); fig.savefig(FIG/name,dpi=150,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); ax.barh(reg.model[::-1],reg.RMSE_micro[::-1]); ax.set_xlabel('OOS RMSE (USD)'); ax.set_title('V9 walk-forward price forecasting'); save(fig,'model_comparison.png')
    fig,ax=plt.subplots(figsize=(5.5,5)); s=cat.sample(min(5000,len(cat)),random_state=SEED); ax.scatter(s.actual,s.predicted,s=6,alpha=.25); lim=[0,max(s.actual.max(),s.predicted.max())]; ax.plot(lim,lim,'--',color='gray'); ax.set(xlabel='Actual next-month USD',ylabel='Predicted next-month USD',title='V9 CatBoost OOS'); save(fig,'actual_vs_predicted.png')
    fig,ax=plt.subplots(figsize=(8,4)); d=cat.assign(ae=abs(cat.actual-cat.predicted)).groupby('commodity_group').ae.mean().sort_values(); ax.barh(d.index,d.values); ax.set_xlabel('MAE (USD)'); ax.set_title('V9 CatBoost by commodity'); save(fig,'error_by_commodity.png')
    fig,ax=plt.subplots(figsize=(8,4)); err=cat.predicted-cat.actual; ax.hist(err[cat.shock==0],bins=60,range=(-1,1),density=True,alpha=.55,label='Non-shock'); ax.hist(err[cat.shock==1],bins=60,range=(-1,1),density=True,alpha=.55,label='Shock'); ax.set(xlabel='Forecast error (USD, shown ±1)',title='V9 error distribution'); ax.legend(); save(fig,'shock_residual_dist.png')
    fig,ax=plt.subplots(figsize=(8,4)); trend=df.groupby(['date','commodity_group']).usdprice.mean().unstack(); trend.plot(ax=ax); ax.set_ylabel('Mean observed USD'); ax.set_title('Supplied model-ready panel'); save(fig,'eda_trends.png')
    fig,ax=plt.subplots(figsize=(8,5)); s=imp.head(15).iloc[::-1]; ax.barh(s.feature,s.mean_abs_shap); ax.set_xlabel('Mean absolute SHAP (residual USD)'); ax.set_title('Exact frozen official CatBoost, holdout sample'); save(fig,'shap_summary.png')
    fig,ax=plt.subplots(figsize=(9,4)); ax.barh(ab.feature_set[::-1],ab.RMSE_micro[::-1]); ax.set_xlabel('OOS RMSE (USD)'); ax.set_title('V9 controlled feature ablation'); save(fig,'feature_ablation.png')
    fig,ax=plt.subplots(figsize=(9,4)); x=np.arange(len(shock)); ax.bar(x-.18,shock.Precision,.36,label='Precision'); ax.bar(x+.18,shock.Recall,.36,label='Recall'); ax.set_xticks(x,shock.method,rotation=25,ha='right'); ax.legend(); ax.set_title('V9 alert decision comparison'); save(fig,'shock_classification.png')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--trials',type=int,default=25,help='Optuna trials per tuned algorithm per fold; use 1 for quick smoke run.'); run(p.parse_args().trials)
