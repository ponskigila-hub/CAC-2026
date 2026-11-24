"""Calendar-safe feature engineering shared by raw preprocessing and scenarios."""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.utils import FEATURE_COLS,ensure_feature_row

PRICE_FEATURES=FEATURE_COLS[:18]
FX_FEATURES=['fx_pct_1m','fx_pct_3m','fx_vol_6m']
SCENARIO_LAGS=(1,2,3,4,5,6,12,13)


def monthly_price_features(price:pd.Series)->pd.DataFrame:
    """Input: monthly calendar-indexed USD prices; retain missing months as NaN.

    All lags, rolling statistics and momentum are based on t-1 or earlier;
    current price is separately included as usdprice at the forecast origin.
    """
    price=price.astype(float)
    d=pd.DataFrame({'usdprice':price},index=price.index)
    for lag in (1,2,3,6,12): d[f'lag_{lag}']=price.shift(lag)
    shifted=price.shift(1); r=shifted.rolling(6,min_periods=3)
    d['rolling_mean_3']=shifted.rolling(3,min_periods=2).mean()
    d['rolling_mean_6']=r.mean(); d['rolling_std_6']=r.std(); d['rolling_min_6']=r.min(); d['rolling_max_6']=r.max()
    d['price_to_mean6']=shifted/d['rolling_mean_6']; d['cv_6']=d['rolling_std_6']/d['rolling_mean_6']
    for lag in (1,3,12): d[f'pct_change_{lag}m']=shifted.pct_change(lag,fill_method=None)
    month=price.index.month; d['month']=month; d['month_sin']=np.sin(2*np.pi*month/12); d['month_cos']=np.cos(2*np.pi*month/12)
    return d


def add_regional_gap(frame:pd.DataFrame)->pd.DataFrame:
    frame=frame.copy(); frame['asean_comm_mom_1m']=frame['asean_comm_mom_1m'].replace([np.inf,-np.inf],np.nan).fillna(0.)
    frame['country_vs_asean_mom']=frame['pct_change_1m']-frame['asean_comm_mom_1m']
    return frame


def scenario_features(ref:dict,current:float,history:dict[int,float],asean_mom:float,date)->dict:
    """Recompute the complete lag-derived feature block from a synthetic history.

    Months 7–11 need no input because none of the requested features use them.
    FX features are deliberately held fixed to the reference market observation.
    """
    if set(SCENARIO_LAGS)-set(history): raise ValueError('Provide every required historical lag.')
    values=[float(current),*[float(history[k]) for k in SCENARIO_LAGS]]
    if not np.isfinite(values).all() or min(values)<=0: raise ValueError('Scenario prices must be finite and positive.')
    date=pd.Timestamp(date).to_period('M').to_timestamp(); ix=pd.date_range(end=date,periods=14,freq='MS')
    price=pd.Series(np.nan,index=ix); price.loc[date]=current
    for lag in SCENARIO_LAGS: price.loc[date-pd.DateOffset(months=lag)]=history[lag]
    price_features=monthly_price_features(price).iloc[-1].to_dict()
    row=dict(ref); row.update({c:price_features[c] for c in PRICE_FEATURES}); row['asean_comm_mom_1m']=float(asean_mom)
    row['country_vs_asean_mom']=row['pct_change_1m']-row['asean_comm_mom_1m']
    return ensure_feature_row(row)
