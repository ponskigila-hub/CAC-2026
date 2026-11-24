"""V9 shared temporal fit/calibration/evaluation engine for research and Streamlit."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from core.utils import (FEATURE_COLS,CAT_COLS,smape,reconstruct_price,ensure_feature_row,
                   alert_mask,select_threshold,calibration_offset,alert_metrics,price_change,shock_event)
try:
    import lightgbm as lgb
    HAS_LGBM=True
except ImportError:
    HAS_LGBM=False
try:
    import xgboost as xgb
    HAS_XGB=True
except ImportError:
    HAS_XGB=False
try:
    from catboost import CatBoostRegressor
    HAS_CAT=True
except ImportError:
    HAS_CAT=False

ALGORITHMS=['CatBoost','LightGBM','XGBoost','HistGradientBoosting','Random Forest']
DEFAULT_PARAMS=dict(n_estimators=120,learning_rate=.05,max_depth=5)
BUNDLE_KEYS=('model','q10_model','q90_model','qcal','shock_clf','encoders','q_encoders','feature_names','target_mode','threshold','cutoff','params','metrics','fit_end','cal_start','cal_end','tune_end','threshold_selection','known_combos','algorithm','test_end')


def available_algorithms():
    return [a for a,ok in zip(ALGORITHMS,[HAS_CAT,HAS_LGBM,HAS_XGB,True,True]) if ok]


def _build_regressor(name,n_estimators,learning_rate,max_depth,random_state=42,subsample=.8,colsample_bytree=.8):
    if name not in available_algorithms(): raise ValueError(f'Algorithm unavailable: {name}')
    if name=='CatBoost':
        return CatBoostRegressor(iterations=n_estimators,learning_rate=learning_rate,depth=max_depth,random_seed=random_state,thread_count=2,verbose=0,allow_writing_files=False)
    if name=='LightGBM':
        return lgb.LGBMRegressor(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,subsample=subsample,subsample_freq=1,colsample_bytree=colsample_bytree,min_child_samples=30,reg_lambda=1.,random_state=random_state,n_jobs=2,verbose=-1)
    if name=='XGBoost':
        return xgb.XGBRegressor(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,subsample=subsample,colsample_bytree=colsample_bytree,random_state=random_state,n_jobs=2,verbosity=0)
    if name=='HistGradientBoosting':
        return HistGradientBoostingRegressor(max_iter=n_estimators,learning_rate=learning_rate,max_depth=max_depth,random_state=random_state)
    return RandomForestRegressor(n_estimators=min(n_estimators,200),max_depth=max_depth,min_samples_leaf=5,n_jobs=2,random_state=random_state)


def _fit_encoders(train_df):
    return {c:LabelEncoder().fit(train_df[c].astype(str)) for c in CAT_COLS}


def _transform(df,encoders,features=None):
    f=FEATURE_COLS if features is None else features
    out=df[list(f)].astype(float).copy()
    for c in CAT_COLS:
        mapping={v:i for i,v in enumerate(encoders[c].classes_)}
        out[c+'_enc']=df[c].astype(str).map(mapping).fillna(-1).astype(int)
    return out


def validate_panel(df):
    required=['date','series_id','target_price_usd_t1']+FEATURE_COLS+CAT_COLS
    missing=set(required)-set(df.columns)
    if missing: raise ValueError(f'Missing panel columns: {sorted(missing)}')
    d=df.copy(); d['date']=pd.to_datetime(d.date)
    if d[required].isna().any().any(): raise ValueError('Missing data in required fields.')
    if not np.isfinite(d[FEATURE_COLS+['target_price_usd_t1']].to_numpy(float)).all(): raise ValueError('Non-finite prices/features.')
    if (d[['usdprice','target_price_usd_t1','lag_1','lag_2','lag_3','lag_6','lag_12','rolling_mean_3','rolling_mean_6']]<=0).any().any(): raise ValueError('Non-positive prices.')
    if (d.date.dt.day!=1).any() or d.duplicated(['series_id','date']).any(): raise ValueError('Non-month-start or duplicate series/date.')
    if not np.allclose(d.country_vs_asean_mom,d.pct_change_1m-d.asean_comm_mom_1m,atol=1e-12,rtol=1e-10): raise ValueError('Regional gap feature inconsistent.')
    if 'shock_label' in d and not np.array_equal(d.shock_label.to_numpy(int),shock_event(price_change(d.usdprice,d.target_price_usd_t1)).astype(int)):
        raise ValueError('Shock labels inconsistent with shared policy.')
    return d.sort_values(['date','series_id']).reset_index(drop=True)


def time_based_split(df,test_months=18):
    start=df.date.max()-pd.DateOffset(months=test_months-1)
    return df.date+pd.DateOffset(months=1)<start, df.date>=start


def _month_block_cal_split(train_df,val_ratio=.2):
    months=sorted(train_df.date.unique())
    if len(months)<5: raise ValueError('At least five distinct months needed.')
    cutoff=pd.Timestamp(months[-max(1,int(len(months)*val_ratio))])
    fit=train_df[train_df.date+pd.DateOffset(months=1)<cutoff]
    val=train_df[train_df.date>=cutoff]
    if len(fit)<50 or len(val)<15: raise ValueError('Insufficient chronological fit/validation data.')
    return fit,val


def split_panel(df,cutoff=None,end=None,test_months=18):
    data=validate_panel(df)
    if cutoff is None:
        cutoff=data.date.max()-pd.DateOffset(months=test_months-1)
    cutoff=pd.Timestamp(cutoff)
    outer=data[data.date+pd.DateOffset(months=1)<cutoff]
    test=data[data.date>=cutoff]
    if end is not None: test=test[test.date<pd.Timestamp(end)]
    if len(outer)<100 or len(test)<10: raise ValueError('Insufficient data after temporal split.')
    fit,cal=_month_block_cal_split(outer,.2)
    return fit,cal,test,cutoff


def _target(df,mode):
    y=df.target_price_usd_t1.to_numpy(float)
    return y-df.usdprice.to_numpy(float) if mode=='residual' else y


def _price(raw,current,mode):
    # Food prices cannot be negative. Apply the same transform in all paths.
    return np.maximum(0.,reconstruct_price(current,raw) if mode=='residual' else np.asarray(raw,float))


def tune_point(algorithm,fit,features,mode,trials,base_params):
    if not trials: return dict(base_params),None
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tr,va=_month_block_cal_split(fit,.2)
    enc=_fit_encoders(tr); Xtr=_transform(tr,enc,features); Xva=_transform(va,enc,features)
    ytr=_target(tr,mode); yva=va.target_price_usd_t1.to_numpy(float); cur=va.usdprice.to_numpy(float)
    def objective(trial):
        p=dict(n_estimators=trial.suggest_int('n_estimators',80,180),max_depth=trial.suggest_int('max_depth',3,7),learning_rate=trial.suggest_float('learning_rate',.02,.1,log=True))
        if algorithm in ('LightGBM','XGBoost'):
            p.update(subsample=trial.suggest_float('subsample',.6,1.),colsample_bytree=trial.suggest_float('colsample_bytree',.6,1.))
        model=_build_regressor(algorithm,**p); model.fit(Xtr,ytr)
        return float(np.sqrt(np.mean((yva-_price(model.predict(Xva),cur,mode))**2)))
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective,n_trials=trials,show_progress_bar=False)
    return dict(study.best_params),str(va.date.max().date())


def fit_point(fit,test,algorithm,params,features=None,target_mode='residual'):
    """Used by benchmark, app, and controlled ablation."""
    enc=_fit_encoders(fit); Xtr=_transform(fit,enc,features); Xte=_transform(test,enc,features)
    model=_build_regressor(algorithm,**params); model.fit(Xtr,_target(fit,target_mode))
    pred=_price(model.predict(Xte),test.usdprice.to_numpy(float),target_mode)
    return model,enc,pred


def train_and_evaluate(df:pd.DataFrame,algorithm='CatBoost',target_mode='residual',n_estimators=120,learning_rate=.05,max_depth=5,test_months=18,cutoff=None,end=None,features=None,tune_trials=0,risk_layer=True)->dict[str,Any]:
    if target_mode not in ('residual','direct'): raise ValueError('Invalid target mode.')
    features=list(FEATURE_COLS if features is None else features)
    if not features or not set(features)<=set(FEATURE_COLS): raise ValueError('Invalid feature subset.')
    fit,cal,test,cutoff=split_panel(df,cutoff,end,test_months)
    params,tune_end=tune_point(algorithm,fit,features,target_mode,tune_trials,dict(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth))
    model,enc,y_pred=fit_point(fit,test,algorithm,params,features,target_mode)
    Xtr=_transform(fit,enc,features); Xca=_transform(cal,enc,features); Xte=_transform(test,enc,features)
    y_true=test.target_price_usd_t1.to_numpy(float); current=test.usdprice.to_numpy(float)
    ycal=_target(cal,target_mode); yfit=_target(fit,target_mode)
    true_shock=shock_event(price_change(current,y_true)).astype(int)
    q10=q90=clf=None; qcal=0.; threshold=None; selection=None
    lo=hi=score=np.full(len(test),np.nan); alerts=np.zeros(len(test),bool)
    if risk_layer:
        if HAS_LGBM and target_mode=='residual':
            q10,q90=[lgb.LGBMRegressor(objective='quantile',alpha=a,n_estimators=100,max_depth=4,learning_rate=.05,random_state=42,n_jobs=2,verbose=-1) for a in (.1,.9)]
            for q in (q10,q90): q.fit(Xtr,yfit)
            a,b=q10.predict(Xca),q90.predict(Xca)
            qcal=calibration_offset(np.maximum(np.minimum(a,b)-ycal,ycal-np.maximum(a,b)))
            a,b=q10.predict(Xte),q90.predict(Xte)
            lo=np.maximum(0,current+np.minimum(a,b)-qcal); hi=np.maximum(0,current+np.maximum(a,b)+qcal)
        ys=shock_event(price_change(fit.usdprice,fit.target_price_usd_t1)).astype(int)
        ysc=shock_event(price_change(cal.usdprice,cal.target_price_usd_t1)).astype(int)
        if len(np.unique(ys))<2 or len(np.unique(ysc))<2: raise ValueError('Both shock classes required in fit and calibration.')
        if HAS_LGBM:
            clf=lgb.LGBMClassifier(n_estimators=120,max_depth=4,min_child_samples=50,reg_lambda=5.,subsample=.8,subsample_freq=1,colsample_bytree=.8,learning_rate=.03,class_weight='balanced',random_state=42,n_jobs=2,verbose=-1)
        else:
            clf=HistGradientBoostingClassifier(max_iter=120,max_depth=4,learning_rate=.03,class_weight='balanced',random_state=42)
        clf.fit(Xtr,ys); sc=clf.predict_proba(Xca)[:,1]
        pc=_price(model.predict(Xca),cal.usdprice.to_numpy(float),target_mode)
        selection=select_threshold(ysc,sc,price_change(cal.usdprice,pc),return_details=True)
        threshold=selection['threshold']; score=clf.predict_proba(Xte)[:,1]
        alerts=alert_mask(price_change(current,y_pred),score,threshold)
    metrics=dict(model=f'{algorithm} ({target_mode.title()})',MAE=mean_absolute_error(y_true,y_pred),RMSE=float(np.sqrt(mean_squared_error(y_true,y_pred))),sMAPE=smape(y_true,y_pred),R2=r2_score(y_true,y_pred),Persistence_MAE=mean_absolute_error(y_true,current),Persistence_RMSE=float(np.sqrt(mean_squared_error(y_true,current))),Persistence_sMAPE=smape(y_true,current),Persistence_R2=r2_score(y_true,current),PICP_P10_P90=float(np.mean((y_true>=lo)&(y_true<=hi))) if q10 is not None else float('nan'),n_train=len(fit),n_calibration=len(cal),n_test=len(test))
    if risk_layer: metrics.update(alert_metrics(true_shock,alerts,score))
    combos=[tuple(x) for x in fit[CAT_COLS].drop_duplicates().astype(str).itertuples(index=False,name=None)]
    return dict(model=model,q10_model=q10,q90_model=q90,qcal=qcal,shock_clf=clf,encoders=enc,q_encoders=enc,feature_names=list(Xtr.columns),target_mode=target_mode,metrics=metrics,threshold=threshold,cutoff=str(cutoff.date()),params=params,algorithm=algorithm,fit_end=str(fit.date.max().date()),cal_start=str(cal.date.min().date()),cal_end=str(cal.date.max().date()),tune_end=tune_end,test_end=str(test.date.max().date()),threshold_selection=selection,known_combos=combos,test_rows=test,y_true=y_true,y_pred=y_pred,y_pers=current,y_lo=lo,y_hi=hi,shock_proba=score,alerts=alerts,shock_true=true_shock,dates=test.date.to_numpy())


def make_bundle(out):
    return {k:out[k] for k in BUNDLE_KEYS}


def _row_to_frame(row,feature_names,encoders):
    r=ensure_feature_row(row)
    features=[c for c in feature_names if c in FEATURE_COLS]
    return _transform(pd.DataFrame([r]),encoders,features)[feature_names]


def predict_single(model,encoders,feature_names,target_mode,row):
    X=_row_to_frame(row,feature_names,encoders)
    return float(_price(model.predict(X)[0],row['usdprice'],target_mode))


def predict_two_stage(bundle,row):
    r=ensure_feature_row(row); X=_row_to_frame(r,bundle['feature_names'],bundle['encoders']); current=r['usdprice']
    yhat=float(_price(bundle['model'].predict(X)[0],current,bundle['target_mode']))
    q10,q90=bundle['q10_model'],bundle['q90_model']; lo=hi=None
    if q10 is not None and q90 is not None:
        a,b=float(q10.predict(X)[0]),float(q90.predict(X)[0]); qcal=float(bundle['qcal'])
        lo=max(0.,current+min(a,b)-qcal); hi=max(0.,current+max(a,b)+qcal)
    score=float(bundle['shock_clf'].predict_proba(X)[0,1]); change=float(price_change(current,yhat))
    return dict(y_hat=yhat,y_p10=lo,y_p90=hi,shock_proba=score,y_t=current,change=change,alert=bool(alert_mask(change,score,bundle['threshold'])))
