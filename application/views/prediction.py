"""V9: replay the exact OOS model or simulate a coherent price-history scenario."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from application.data_loader import load_model_ready
from core.models import predict_two_stage
from core.model_store import load_bundle
from core.features import scenario_features,SCENARIO_LAGS
from application.styles import inject_css,hero,section,status_badge
from core.utils import classify_shock,price_change,shock_event,FEATURE_COLS,CAT_COLS
DATA=Path(__file__).resolve().parents[2]/'data'

@st.cache_resource(show_spinner=False)
def frozen_bundle(fold=None):
    return load_bundle(fold)

@st.cache_data(show_spinner=False)
def historical_oos():
    d=pd.read_csv(DATA/'v9_oos_predictions.csv',parse_dates=['date'])
    return d[d.model=='CatBoost'].copy()


def choose_row(df,key):
    import hashlib
    d=df.copy(); cols=st.columns(3); scope=key
    for i,(col,label) in enumerate([('countryiso3','Country'),('commodity_group','Commodity'),('market','Market'),('pricetype','Price type'),('series_id','Market series')]):
        options=sorted(d[col].unique())
        with cols[i%3]:
            # Include upstream filter selections in key to avoid stale incompatible choices.
            val=st.selectbox(label,options,key=f'{scope}_{col}')
        d=d[d[col]==val]
        scope=hashlib.sha1((scope+str(val)).encode()).hexdigest()[:16]
    dates=sorted(d.date.unique(),reverse=True)
    with cols[2]:
        date=st.selectbox('Forecast origin (month)',dates,format_func=lambda x:pd.Timestamp(x).strftime('%Y-%m'),key=f'{scope}_date')
    return d[d.date==date].iloc[0]


def display_prediction(current,predicted,score,threshold,lower=None,upper=None,actual=None):
    change=float(price_change(current,predicted)); label,_=classify_shock(change,score,threshold)
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Current USD',f'{current:.4f}')
    c2.metric('Next-month USD',f'{predicted:.4f}',f'{change:+.2%}')
    c3.metric('Shock risk score',f'{score:.3f}')
    c4.markdown(status_badge(label,'shock' if 'ALERT' in label else 'watch' if 'WASPADA' in label else 'normal'),unsafe_allow_html=True)
    if lower is not None and upper is not None and np.isfinite([lower,upper]).all():
        st.write(f'Calibrated interval (80% target): USD {lower:.4f} to {upper:.4f}')
    else: st.caption('No validated interval for this configuration (direct mode or quantiles unavailable).')
    if actual is not None:
        a,b=st.columns(2); a.metric('Observed next-month USD',f'{actual:.4f}'); b.metric('Forecast error (USD)',f'{predicted-actual:+.4f}')
        st.write('Observed shock:', 'Yes' if shock_event(price_change(current,actual)) else 'No')
    st.caption(f'Hybrid alert: score ≥ {threshold:.3f} OR forecast increase > 5%. Risk score is not a calibrated probability. The warning tier is illustrative; binary evaluation measures ALERT only.')


def render():
    inject_css()
    st.markdown(hero('Prediction & Early Shock Detection','Replay historical OOS forecasts · frozen official model · coherent what-if inputs',pills=[('V9 shared engine','pill-sky')]),unsafe_allow_html=True)
    mode=st.radio('Mode',['Historical out-of-sample','Official holdout model what-if','Custom trained model what-if'],horizontal=True)
    df=load_model_ready()
    if mode=='Historical out-of-sample':
        if not (DATA/'v9_oos_predictions.csv').exists(): st.warning('Run python src/run_v9.py to generate predictions.'); return
        d=historical_oos(); row=choose_row(d,'hist')
        try: bundle=frozen_bundle(str(row.fold))
        except Exception as exc: st.error(f'Cannot load historical model: {exc}'); return
        ref=df[(df.series_id==row.series_id)&(df.date==row.date)].iloc[0]
        out=predict_two_stage(bundle,ref.to_dict())
        replay=np.allclose([out['y_hat'],out['y_p10'],out['y_p90'],out['shock_proba']],[row.predicted,row.lower,row.upper,row.score],atol=1e-8,rtol=1e-8) and out['alert']==bool(row.alert)
        if replay: st.success('Live inference matches the stored out-of-sample forecast.')
        else: st.error('Replay differs from the stored benchmark. Check package versions and rerun the benchmark.'); return
        st.caption(f"Fold cutoff {row.fold} · model fit through {bundle['fit_end']} · calibration {bundle['cal_start']} to {bundle['cal_end']} · forecast origin {row.date:%Y-%m}. Actual next-month price is displayed after inference only.")
        if tuple(str(ref[c]) for c in CAT_COLS) not in set(bundle['known_combos']): st.warning('This category combination was not seen in fit data. Historical OOS evaluation deliberately includes such cases.')
        display_prediction(out['y_t'],out['y_hat'],out['shock_proba'],bundle['threshold'],out['y_p10'],out['y_p90'],row.actual)
        series=d[(d.series_id==row.series_id)&(d.date<=row.date)].sort_values('date').copy(); series['target_month']=series.date+pd.DateOffset(months=1)
        st.caption('Chart aligns predicted and actual prices by target month t+1 (not forecast origin).')
        st.line_chart(series.set_index('target_month')[['predicted','actual','lower','upper']])
        return
    if mode.startswith('Custom'):
        bundle=st.session_state.get('two_stage_bundle')
        if bundle is None: st.info('Train a custom model on Model Training first.'); return
    else:
        try: bundle=frozen_bundle()
        except Exception as exc: st.error(f'Official model unavailable: {exc}. Run src/run_v9.py.'); return
    st.caption(f"{bundle['algorithm']} · {bundle['target_mode']} · model fit through {bundle['fit_end']} · calibration ends {bundle['cal_end']} · holdout starts {bundle['cutoff']}")
    combos=set(bundle['known_combos']); known=[tuple(str(v) for v in x) in combos for x in df[CAT_COLS].itertuples(index=False,name=None)]
    df=df.loc[np.asarray(known)&(df.date>=pd.Timestamp(bundle['cutoff']))]
    if df.empty: st.warning('No reference observations from category combinations seen during fit.'); return
    ref=choose_row(df,'scenario_'+mode)
    row={k:ref[k] for k in FEATURE_COLS+CAT_COLS}
    st.caption(f'Units: USD per {ref.unit}. Reference month {ref.date:%Y-%m}. These are hypothetical scenarios, not real-time forecasts.')
    advanced=st.checkbox('Edit full historical price features (synthetic scenario)',value=False)
    current=st.number_input('Scenario current price (USD)',min_value=0.000001,value=float(ref.usdprice),format='%.6f',key=f'current_{ref.series_id}_{ref.date}_{mode}')
    if advanced:
        st.info('Specify a coherent synthetic history. Rolling features, momentum, seasonality and regional gap are recomputed together; FX stays fixed. Unavailable historical values are only scenario defaults, not observed data.')
        lag_defaults={k:float(ref[f'lag_{k}']) for k in (1,2,3,6,12)}
        lag_defaults[4]=float(ref.lag_1/(1+ref.pct_change_3m))
        lag_defaults[13]=float(ref.lag_1/(1+ref.pct_change_12m))
        # Use lag-5 if recoverable from model-ready history; otherwise explicitly use a synthetic default.
        prior=load_model_ready(); m=prior[(prior.series_id==ref.series_id)&(prior.date==ref.date-pd.DateOffset(months=5))]
        lag_defaults[5]=float(m.usdprice.iloc[0]) if len(m) else float(ref.lag_1)
        cols=st.columns(3); history={}
        for i,k in enumerate(SCENARIO_LAGS):
            with cols[i%3]: history[k]=st.number_input(f'Price t−{k} (USD)',min_value=0.000001,value=max(.000001,lag_defaults[k]),format='%.6f',key=f'lag{k}_{ref.series_id}_{ref.date}_{mode}')
        regional=st.number_input('Lagged ASEAN commodity momentum (fraction)',value=float(ref.asean_comm_mom_1m),format='%.6f',key=f'reg_{ref.series_id}_{ref.date}_{mode}')
        row=scenario_features(row,current,history,regional,ref.date)
    else:
        row['usdprice']=current
        st.caption('Only current price changes; the original historical features (all based on t−1 or earlier) remain fixed. FX and regional data remain fixed.')
    with st.expander('Model input features',expanded=False): st.dataframe(pd.DataFrame({'feature':FEATURE_COLS,'value':[row[k] for k in FEATURE_COLS]}),hide_index=True)
    if st.button('Predict next month',type='primary'):
        try:
            out=predict_two_stage(bundle,row)
            display_prediction(out['y_t'],out['y_hat'],out['shock_proba'],bundle['threshold'],out['y_p10'],out['y_p90'])
        except Exception as exc: st.error(f'Invalid scenario: {exc}')
