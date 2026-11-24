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

GROUPS=['1–3','4–6','7–12','13–36']
def horizon_group(h):
    return np.select([np.asarray(h)<=3,np.asarray(h)<=6,np.asarray(h)<=12],GROUPS[:3],default=GROUPS[3])

def pairs(d,cutoff,training=True):
    cutoff=pd.Timestamp(cutoff);cm=cutoff.year*12+cutoff.month
    a=eligible(d);a=a[(a.m<=cm)&(a.m%3==0)] if training else a[a.m==cm]
    if training and len(a)>6000:a=a.sample(6000,random_state=42).sort_values(['series','m'])
    lookup=d.set_index(['series','m'])[['price','mom1']];frames=[]
    for h in range(1,HORIZONS+1):
        o=a if not training else a[a.m+h<=cm]
        if o.empty:continue
        tm=o.m.to_numpy()+h;actual=lookup.reindex(pd.MultiIndex.from_arrays([o.series.to_numpy(),tm])).to_numpy();mask=np.isfinite(actual[:,0]);o=o.iloc[np.flatnonzero(mask)].reset_index(drop=True);actual=actual[mask];tm=tm[mask]
        if o.empty:continue
        f=features(o,h);f['y']=np.log(actual[:,0]/o.price);f['actual']=actual[:,0];f['base']=o.price;f['shock']=np.where(np.isfinite(actual[:,1]),actual[:,1]>=np.log(1.05)-1e-10,np.nan);f['series']=o.series;f['origin_m']=o.m;f['target_m']=tm;f['province']=o.province;f['commodity']=o.commodity
        ref=lookup.price.reindex(pd.MultiIndex.from_arrays([o.series.to_numpy(),tm-12*((h+11)//12)])).to_numpy();f['seasonal']=np.where(np.isfinite(ref),ref,o.price);frames.append(f)
    return pd.concat(frames,ignore_index=True)

def predict_price(model,x,base): return np.asarray(base)*np.exp(np.clip(model.predict(x),-5,5))

def forecast_path(d,series,target,bundle,algorithm=None):
    target=pd.Timestamp(target).to_period('M').to_timestamp();s=eligible(d);s=s[s.series==series].sort_values('m')
    if s.empty:raise ValueError('Riwayat empat bulan berurutan tidak tersedia.')
    origin=s.iloc[[-1]].copy();start=int(origin.m.iloc[0]);end=target.year*12+target.month
    if end<=start:raise ValueError('Pilih bulan setelah observasi terakhir pasar ini.')
    alg=algorithm or bundle['selected'];rows=[];prev=float(origin.price.iloc[0]);base=prev
    hs=np.arange(1,min(end-start,HORIZONS)+1);batch=origin.loc[origin.index.repeat(len(hs))].reset_index(drop=True);x=features(batch,hs);scores=bundle['risk_model'].predict_proba(x)[:,1]
    prices={n:predict_price(m,x,np.repeat(base,len(hs))) for n,m in bundle['price_models'].items()};prices['Persistence']=np.repeat(base,len(hs));hist=d[d.series.eq(series)].set_index('m').price;ref=start+hs-12*((hs+11)//12);prices['Seasonal Naive']=hist.reindex(ref).fillna(base).to_numpy()
    for n,w in bundle.get('blends',{}).items():prices[n]=base*np.exp(w*np.log(prices['Histogram Gradient Boosting']/base))
    for h in range(1,end-start+1):
        bucket=str(horizon_group(h));chosen=bundle['price_policy'][bucket] if alg==bundle['selected'] else alg
        if h<=HORIZONS:p=float(prices[chosen][h-1])
        else:p=rows[h-13]['price']*np.exp(float(origin.trend.iloc[0])*12*.5**((h-37)//12+1))
        score=float(scores[h-1]) if h<=HORIZONS else np.nan;threshold=bundle['thresholds'][bucket] if h<=HORIZONS else np.nan
        band=bundle['bands'].get(alg,{}).get(bucket) if h<=HORIZONS else None;lo,hi=(p*np.exp(band[0]),p*np.exp(band[1])) if band is not None else (np.nan,np.nan)
        m=start+h;rows.append({'lower':lo,'upper':hi,'date':pd.Timestamp(year=(m-1)//12,month=(m-1)%12+1,day=1),'horizon':h,'price':p,'change':p/prev-1,'risk_score':score,'threshold':threshold,'shock':bool(score>=threshold) if np.isfinite(score) else None,'price_implied_shock':bool(p/prev-1>=.05),'price_model':chosen if h<=HORIZONS else 'Extrapolation','extrapolation':h>HORIZONS});prev=p
    return origin.iloc[0],pd.DataFrame(rows)

def load():return pd.read_csv(ROOT/'data/processed/monthly.csv.gz',parse_dates=['date']),joblib.load(ROOT/'models/forecast.joblib')
