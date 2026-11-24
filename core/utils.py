"""FoodShock V9: common feature schema, target definition and alert policy."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLS = [
    'usdprice','lag_1','lag_2','lag_3','lag_6','lag_12',
    'rolling_mean_3','rolling_mean_6','rolling_std_6','rolling_min_6','rolling_max_6',
    'price_to_mean6','cv_6','pct_change_1m','pct_change_3m','pct_change_12m',
    'month_sin','month_cos','fx_pct_1m','fx_pct_3m','fx_vol_6m',
    'asean_comm_mom_1m','country_vs_asean_mom',
]
CAT_COLS=['countryiso3','commodity_group','pricetype']
SHOCK_PCT_THRESHOLD=.05
# Numerical guard only: a mathematically exact 5% move is not a >5% shock.
SHOCK_EPS=1e-12


def smape(y_true,y_pred):
    y,p=np.asarray(y_true,float),np.asarray(y_pred,float)
    d=(np.abs(y)+np.abs(p))/2; m=d!=0
    return float(np.mean(np.abs(y[m]-p[m])/d[m])*100) if m.any() else 0.


def reconstruct_price(y_t,residual_hat):
    return np.asarray(y_t,float)+np.asarray(residual_hat,float)


def price_change(current,predicted):
    c,p=np.asarray(current,float),np.asarray(predicted,float)
    if np.any(~np.isfinite(c)) or np.any(c<=0) or np.any(~np.isfinite(p)):
        raise ValueError('Prices must be finite; current prices must be positive.')
    return (p-c)/c


def pct_change_from_prices(current,predicted):
    return float(price_change(current,predicted))


def shock_event(change):
    return np.asarray(change)>SHOCK_PCT_THRESHOLD+SHOCK_EPS


def alert_mask(change,score,threshold):
    return shock_event(change)|(np.asarray(score)>=np.asarray(threshold,float))


def classify_shock(pct_change:float,shock_proba:float|None=None,threshold:float=.5):
    """The only alert policy; pct_change is a fraction, score is not a calibrated probability."""
    pct=float(pct_change); score=0. if shock_proba is None else float(shock_proba)
    if not np.isfinite([pct,score,threshold]).all(): raise ValueError('Non-finite alert input.')
    if alert_mask(pct,score,threshold): return 'FOOD SHOCK ALERT','#F43F5E'
    # A warning is an illustrative monitoring tier; binary metrics evaluate ALERT only.
    if pct>=.03 or score>=min(.25,threshold): return 'WASPADA / ELEVATED RISK','#F59E0B'
    return 'NORMAL','#10B981'


def select_threshold(y,score,change=None,return_details=False):
    """Validation-only threshold: maximize recall with precision >=20%; else F0.5."""
    from sklearn.metrics import precision_score,recall_score,fbeta_score
    records=[]
    for t in np.linspace(.05,.95,37):
        pred=(np.asarray(score)>=t) if change is None else alert_mask(change,score,t)
        records.append((float(t),float(precision_score(y,pred,zero_division=0)),float(recall_score(y,pred,zero_division=0)),float(fbeta_score(y,pred,beta=.5,zero_division=0))))
    eligible=[r for r in records if r[1]>=.2]
    chosen=max(eligible,key=lambda r:(r[2],r[1],r[0])) if eligible else max(records,key=lambda r:(r[3],r[1],r[0]))
    return {'threshold':chosen[0],'precision':chosen[1],'recall':chosen[2],'f0_5':chosen[3],'rule':'precision_floor' if eligible else 'f0.5_fallback'} if return_details else chosen[0]


def calibration_offset(scores):
    """Finite-sample 80% split-calibration order statistic; expansion-only."""
    scores=np.sort(np.asarray(scores,float)); n=len(scores)
    if n<5 or not np.isfinite(scores).all(): raise ValueError('Insufficient/invalid calibration scores.')
    k=min(n-1,int(np.ceil((n+1)*.8))-1)
    return float(max(0,scores[k]))


def alert_metrics(y,pred,score=None):
    from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix,average_precision_score,roc_auc_score
    y,pred=np.asarray(y,int),np.asarray(pred,bool)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    valid=score is not None and len(np.unique(y))==2
    return dict(Precision=precision_score(y,pred,zero_division=0),Recall=recall_score(y,pred,zero_division=0),F1=f1_score(y,pred,zero_division=0),FPR=fp/max(1,fp+tn),FDR=fp/max(1,fp+tp),PR_AUC=average_precision_score(y,score) if valid else float('nan'),ROC_AUC=roc_auc_score(y,score) if valid else float('nan'),TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn))


def ensure_feature_row(row):
    """Reject incomplete/invalid inputs rather than silently synthesising features."""
    missing=[c for c in FEATURE_COLS+CAT_COLS if c not in row]
    if missing: raise ValueError(f'Missing feature(s): {missing}')
    out={c:float(row[c]) for c in FEATURE_COLS}
    if not np.isfinite(list(out.values())).all(): raise ValueError('Features must be finite.')
    if any(out[c]<=0 for c in ['usdprice','lag_1','lag_2','lag_3','lag_6','lag_12','rolling_mean_3','rolling_mean_6']):
        raise ValueError('Current, lagged and mean prices must be positive.')
    if any(pd.isna(row[c]) for c in CAT_COLS): raise ValueError('Missing categorical feature.')
    out.update({c:str(row[c]) for c in CAT_COLS})
    return out
