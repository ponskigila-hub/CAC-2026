from pathlib import Path
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1]
KEY=['market_id','commodity_id','unit','currency']
HORIZONS=36
FEATURES=['mom1','mom3','mom6','mom12','vol3','trend','history','season_sin','season_cos','origin_sin','origin_cos','h','h2','trend_h','season_delta','mom6_missing','mom12_missing']

def prepare(raw):
    d=raw.copy(); d['price']=pd.to_numeric(d.price,errors='coerce'); d['date']=pd.to_datetime(d.date,errors='coerce').dt.to_period('M').dt.to_timestamp()
    d=d[d.countryiso3.eq('IDN')&d.currency.eq('IDR')&d.pricetype.eq('Retail')&d.category.ne('non-food')&d.price.gt(0)&np.isfinite(d.price)].dropna(subset=KEY+['date','market','commodity'])
    d=d.groupby(KEY+['date'],as_index=False).agg(price=('price','median'),market=('market','first'),commodity=('commodity','first'),province=('admin1','first'),category=('category','first'))
    d['province']=d.province.fillna('National / unspecified');d['series']=d.groupby(KEY,sort=True).ngroup();d['m']=d.date.dt.year*12+d.date.dt.month
    d=d.sort_values(['series','m']).reset_index(drop=True);g=d.groupby('series',sort=False)
    for k in [1,2,3,6,12]:
        lag=g.price.shift(k).where(d.m-g.m.shift(k)==k);d[f'mom{k}']=np.log(d.price/lag)
    d['vol3']=pd.concat([d.mom1,g.mom1.shift(1),g.mom1.shift(2)],axis=1).std(axis=1,ddof=0).where(d.m-g.m.shift(3)==3)
    d['history']=g.cumcount()+1
    d['trend']=d.mom12.div(12).fillna(d.mom6.div(6)).fillna(d.mom3.div(3)).clip(-.04,.04)
    return d

def features(origin,h):
    x=origin.copy();x['h']=h;x['h2']=(np.asarray(h)/12)**2;x['trend_h']=x.trend*np.asarray(h)
    target=(x.m+np.asarray(h)-1)%12+1;om=(x.m-1)%12+1
    x['season_sin']=np.sin(2*np.pi*target/12);x['season_cos']=np.cos(2*np.pi*target/12)
    x['origin_sin']=np.sin(2*np.pi*om/12);x['origin_cos']=np.cos(2*np.pi*om/12);x['season_delta']=x.season_sin-x.origin_sin
    for k in [6,12]:x[f'mom{k}_missing']=x[f'mom{k}'].isna().astype(int)
    return x[FEATURES].fillna(0).replace([np.inf,-np.inf],0)

def eligible(d): return d[d.mom3.notna()&d.mom1.notna()&d.vol3.notna()].copy()

def pairs(d,cutoff,training=True):
    cutoff=pd.Timestamp(cutoff);cm=cutoff.year*12+cutoff.month
    a=eligible(d);a=a[(a.m<=cm)&(a.m%3==0)] if training else a[a.m==cm]
    if training and len(a)>6000:a=a.sample(6000,random_state=42).sort_values(['series','m'])
    lookup=d[['series','m','price','mom1']].rename(columns={'m':'target_m','price':'actual','mom1':'actual_mom'})
    frames=[]
    for h in range(1,HORIZONS+1):
        o=a.copy();o['target_m']=o.m+h
        if training:o=o[o.target_m<=cm]
        o=o.merge(lookup,on=['series','target_m'],how='inner');
        if o.empty:continue
        f=features(o,h);f['y']=np.log(o.actual/o.price);f['actual']=o.actual;f['base']=o.price;f['shock']=np.where(o.actual_mom.notna(),o.actual_mom>=np.log(1.05)-1e-10,np.nan);f['series']=o.series;f['origin_m']=o.m;f['target_m']=o.target_m;f['province']=o.province;f['commodity']=o.commodity
        ref=o[['series','m','target_m']].copy();ref['ref_m']=ref.target_m-12*((h+11)//12)
        ref=ref.merge(d[['series','m','price']].rename(columns={'m':'ref_m','price':'seasonal'}),on=['series','ref_m'],how='left');f['seasonal']=ref.seasonal.fillna(o.price).to_numpy();frames.append(f)
    return pd.concat(frames,ignore_index=True)

def predict_price(model,x,base): return np.asarray(base)*np.exp(np.clip(model.predict(x),-5,5))

def forecast_path(d,series,target,bundle,algorithm=None):
    target=pd.Timestamp(target).to_period('M').to_timestamp();s=eligible(d);s=s[s.series==series].sort_values('m')
    if s.empty:raise ValueError('Riwayat empat bulan berurutan tidak tersedia.')
    origin=s.iloc[[-1]].copy();start=int(origin.m.iloc[0]);end=target.year*12+target.month
    if end<=start:raise ValueError('Pilih bulan setelah observasi terakhir pasar ini.')
    alg=algorithm or bundle['selected'];model=bundle['price_models'][alg];rows=[];prev=float(origin.price.iloc[0]);base=prev
    hs=np.arange(1,min(end-start,HORIZONS)+1);batch=origin.loc[origin.index.repeat(len(hs))].reset_index(drop=True);x=features(batch,hs);prices=predict_price(model,x,np.repeat(base,len(hs)));scores=bundle['risk_model'].predict_proba(x)[:,1]
    for h in range(1,end-start+1):
        if h<=HORIZONS: p=float(prices[h-1])
        else:
            # Outside evaluated horizon: repeat final forecast seasonal pattern with damped historical trend.
            p=rows[h-13]['price']*np.exp(float(origin.trend.iloc[0])*12*.5**((h-37)//12+1))
        score=float(scores[h-1]) if h<=HORIZONS else np.nan
        bucket=next((label for limit,label in [(3,'1–3'),(6,'4–6'),(12,'7–12'),(24,'13–24'),(36,'25–36')] if h<=limit),None)
        band=bundle.get('bands',{}).get(alg,{}).get(bucket);lo,hi=(p*np.exp(band[0]),p*np.exp(band[1])) if band is not None else (np.nan,np.nan)
        m=start+h;rows.append({'lower':lo,'upper':hi,'date':pd.Timestamp(year=(m-1)//12,month=(m-1)%12+1,day=1),'horizon':h,'price':p,'change':p/prev-1,'risk_score':score,'shock':bool(score>=bundle['threshold']) if np.isfinite(score) else None,'price_implied_shock':bool(p/prev-1>=.05),'extrapolation':h>HORIZONS});prev=p
    return origin.iloc[0],pd.DataFrame(rows)

def load():return pd.read_csv(ROOT/'data/processed/monthly.csv.gz',parse_dates=['date']),joblib.load(ROOT/'models/forecast.joblib')
