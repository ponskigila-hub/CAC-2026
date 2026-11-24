"""Prepare model-ready input from the supplied V8 panel, or rebuild from a WFP raw extract.

The official V9 run uses the bundled corrected panel. Raw rebuild is an optional
path, not a claim of bit-for-bit reproduction of a source file that was not supplied.
Outputs go to a separate directory unless explicitly specified.
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from core.features import monthly_price_features,add_regional_gap,PRICE_FEATURES,FX_FEATURES
from core.utils import FEATURE_COLS,shock_event,price_change
from core.models import validate_panel
COUNTRIES=['IDN','KHM','LAO','PHL','THA','TLS','MMR']
STATIC=['countryiso3','market','commodity_group','pricetype','priceflag','unit']


def commodity_group(commodity,unit):
    c,u=str(commodity).lower(),str(unit).upper().strip()
    for word,name in [('rice','Rice'),('chicken','Chicken Meat'),('beef','Beef Meat'),('pork','Pork Meat')]:
        if word in c and u=='KG': return name
    if any(x in c for x in ['oil (vegetable','oil (palm','oil (soybean','oil (cooking','oil (groundnut']) and u in ['L','KG','5 L']: return 'Cooking Oil'
    return None


def normalize_panel(input_path,out_dir):
    d=pd.read_csv(input_path); d=add_regional_gap(d); d=validate_panel(d)
    # Leave the observed prices and target labels unchanged.
    d.to_csv(out_dir/'foodshock_asean_model_ready.csv',index=False)
    print('Corrected model-ready panel:',d.shape)


def rebuild_raw(input_path,out_dir):
    raw=pd.read_csv(input_path,low_memory=False)
    required=['countryiso3','date','market','commodity','pricetype','priceflag','unit','price','usdprice']
    missing=set(required)-set(raw.columns)
    if missing: raise ValueError(f'Missing raw WFP columns: {sorted(missing)}')
    raw['date']=pd.to_datetime(raw.date,errors='coerce').dt.to_period('M').dt.to_timestamp()
    for c in ['price','usdprice']: raw[c]=pd.to_numeric(raw[c],errors='coerce')
    raw=raw[raw.countryiso3.isin(COUNTRIES)&raw.date.between('2015-01-01','2025-12-01')&(raw.usdprice>0)].copy()
    # FX proxies are computed before standardizing USD commodity units.
    fxraw=raw[raw.price>0].copy(); fxraw['ratio']=fxraw.price/fxraw.usdprice
    fx_rows=[]
    for country,g in fxraw.groupby('countryiso3'):
        r=g.groupby('date').ratio.median().sort_index(); r=r.reindex(pd.date_range(r.index.min(),r.index.max(),freq='MS'))
        f=pd.DataFrame({'date':r.index,'countryiso3':country,'fx_pct_1m':r.pct_change(1,fill_method=None).values,'fx_pct_3m':r.pct_change(3,fill_method=None).values,'fx_vol_6m':r.pct_change(1,fill_method=None).shift(1).rolling(6,min_periods=3).std().values}); fx_rows.append(f)
    fx=pd.concat(fx_rows,ignore_index=True) if fx_rows else pd.DataFrame(columns=['date','countryiso3']+FX_FEATURES)
    d=raw.dropna(subset=['market','pricetype','priceflag','unit','commodity']).copy(); d['unit']=d.unit.astype(str).str.strip().str.upper()
    d['commodity_group']=[commodity_group(c,u) for c,u in zip(d.commodity,d.unit)]
    d=d.dropna(subset=['commodity_group']).copy(); oil=d.commodity_group=='Cooking Oil'
    d.loc[oil&(d.unit=='5 L'),'usdprice']/=5.; d.loc[oil&(d.unit=='KG'),'usdprice']*=.92; d.loc[oil,'unit']='L'
    d['series_id']=d[STATIC].astype(str).agg('|'.join,axis=1)
    coverage=d.groupby(['series_id']+STATIC).agg(n_months=('date','nunique'),n_obs=('usdprice','count'),min_date=('date','min'),max_date=('date','max'),mean_usdprice=('usdprice','mean')).reset_index()
    cov=coverage[coverage.n_months>=60]; d=d[d.series_id.isin(cov.series_id)].copy()
    if d.empty: raise ValueError('No series meets the retrospective 60-month coverage rule.')
    cov.groupby('countryiso3').agg(n_series=('series_id','count'),n_commodity_groups=('commodity_group','nunique'),avg_months=('n_months','mean'),max_months=('n_months','max'),min_date=('min_date','min'),max_date=('max_date','max')).reset_index().to_csv(out_dir/'fpma_wfp_country_coverage.csv',index=False)
    cov.groupby('commodity_group').agg(n_series=('series_id','count'),n_countries=('countryiso3','nunique'),countries=('countryiso3',lambda x:','.join(sorted(x.unique()))),avg_months=('n_months','mean'),avg_price_usd=('mean_usdprice','mean')).reset_index().to_csv(out_dir/'fpma_wfp_commodity_coverage.csv',index=False)
    monthly=d.groupby(['series_id','date']+STATIC,as_index=False).usdprice.mean(); frames=[]
    for sid,g in monthly.groupby('series_id'):
        g=g.sort_values('date'); p=g.set_index('date').usdprice; p=p.reindex(pd.date_range(p.index.min(),p.index.max(),freq='MS'))
        f=monthly_price_features(p); f['target_price_usd_t1']=p.shift(-1); f['date']=f.index; f['series_id']=sid
        for c in STATIC: f[c]=g[c].iloc[0]
        frames.append(f.reset_index(drop=True))
    panel=pd.concat(frames,ignore_index=True)
    # Aggregate lagged momentum from available history before dropping future labels;
    # join by exactly one calendar month, never by next available row.
    reg=panel.groupby(['date','commodity_group'],as_index=False).pct_change_1m.mean().rename(columns={'pct_change_1m':'asean_comm_mom_1m'})
    reg['date']=reg.date+pd.DateOffset(months=1)
    panel=panel.merge(reg,on=['date','commodity_group'],how='left').merge(fx,on=['date','countryiso3'],how='left')
    for c in FX_FEATURES: panel[c]=panel[c].replace([np.inf,-np.inf],np.nan).fillna(0.)
    panel=add_regional_gap(panel)
    panel=panel.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURE_COLS+['target_price_usd_t1']).copy()
    panel=panel[(panel.usdprice>0)&(panel.target_price_usd_t1>0)].copy()
    panel['target_pct_change_t1']=price_change(panel.usdprice,panel.target_price_usd_t1)
    panel['residual_from_yt']=panel.target_price_usd_t1-panel.usdprice
    panel['log_return_t1']=np.log(panel.target_price_usd_t1/panel.usdprice)
    panel['series_key']=panel[['countryiso3','commodity_group','pricetype']].agg('|'.join,axis=1)
    panel['shock_label']=shock_event(panel.target_pct_change_t1).astype(int)
    panel['year_month']=panel.date.dt.to_period('M').astype(str)
    panel=validate_panel(panel)
    # Column order matches canonical V9 dataset.
    columns=['series_id','date']+STATIC+PRICE_FEATURES+['month','target_price_usd_t1','target_pct_change_t1','residual_from_yt','log_return_t1','series_key','shock_label','year_month']+FX_FEATURES+['asean_comm_mom_1m','country_vs_asean_mom']
    panel[columns].to_csv(out_dir/'foodshock_asean_model_ready.csv',index=False)
    print('Raw reconstruction:',panel.shape,'(not the original snapshot unless exact raw file is supplied)')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); g=p.add_mutually_exclusive_group(required=True); g.add_argument('--panel',type=Path); g.add_argument('--raw',type=Path); p.add_argument('--out-dir',type=Path,default=ROOT/'data_rebuilt'); args=p.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.panel: normalize_panel(args.panel,args.out_dir)
    else: rebuild_raw(args.raw,args.out_dir)
