"""Model overview and optional custom training; official artifacts are read-only."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from application.data_loader import (load_model_ready, load_model_results, load_shock_metrics,
    load_eval_by_commodity, load_eval_by_pricetype, load_eval_by_country, get_filter_options)
from core.models import available_algorithms, train_and_evaluate, make_bundle
from core.model_store import load_manifest
from application.styles import inject_css, apply_plot_style, hero, COLORS

DATA = Path(__file__).resolve().parents[2] / 'data'
FIG = DATA.parent / 'figures'


def table(df):
    fmt = {c: '{:.4f}' for c in df.select_dtypes(include='floating').columns}
    st.dataframe(df.style.format(fmt, na_rep='—'), use_container_width=True, hide_index=True)


def overview(results, shock, manifest):
    cat = results.loc[results.model.eq('CatBoost')].iloc[0]
    base = results.loc[results.model.str.startswith('Persistence')].iloc[0]
    alert = shock.loc[shock.method.eq('Hybrid_Operational_Alert')].iloc[0]
    st.subheader('Official benchmark')
    st.caption(f"7 walk-forward folds · {int(cat['n']):,} out-of-sample observations · saved results, no training required")
    cols = st.columns(4)
    cols[0].metric('CatBoost RMSE', f"{cat.RMSE_micro:.4f}", f"{cat.RMSE_micro-base.RMSE_micro:+.4f} vs baseline", delta_color='inverse')
    cols[1].metric('Average error (MAE)', f"{cat.MAE_micro:.4f} USD")
    cols[2].metric('Shock precision', f"{alert.Precision:.1%}")
    cols[3].metric('Interval coverage', f"{alert.Overall_PICP_P10_P90:.1%}")
    left, right = st.columns([1.25, 1])
    with left:
        with st.container(border=True):
            st.markdown('**Forecast accuracy**')
            st.caption('RMSE in USD · lower is better')
            ranked = results.sort_values('RMSE_micro')
            fig, ax = plt.subplots(figsize=(6, 3.1))
            ax.barh(ranked.model, ranked.RMSE_micro, color=[COLORS['primary'] if n=='CatBoost' else COLORS['sky'] if n.startswith('Persistence') else COLORS['border'] for n in ranked.model])
            ax.invert_yaxis(); ax.set_xlim(0, ranked.RMSE_micro.max()*1.24)
            for i, v in enumerate(ranked.RMSE_micro):
                ax.text(v+.003, i, f'{v:.4f}', va='center', fontsize=9)
            ax.set_xlabel('RMSE (USD)'); fig.tight_layout()
            st.pyplot(fig); plt.close(fig)
    with right:
        with st.container(border=True):
            st.markdown('**Early warning performance**')
            st.caption('Hybrid alert · same rule as Prediction')
            for label, key in [('Precision','Precision'), ('Recall','Recall'), ('F1 score','F1'), ('False positive rate','FPR')]:
                value=float(alert[key]); st.write(f'{label}: {value:.1%}'); st.progress(value)
            st.caption(f"Lift: {alert.Lift:.2f}× · interval target: 80%")
    if cat.RMSE_micro >= base.RMSE_micro:
        st.info('Persistence slightly outperforms CatBoost on RMSE. FoodShock adds risk screening and uncertainty estimates, not a proven improvement in point forecasting.')
    else:
        st.info('CatBoost has lower observed RMSE than Persistence. See Advanced analysis for statistical significance.')
    with st.expander('Compare all forecasting models'):
        table(results[['model','MAE_micro','RMSE_micro','sMAPE_micro','R2_micro']].rename(columns={'model':'Model','MAE_micro':'MAE (USD)','RMSE_micro':'RMSE (USD)','sMAPE_micro':'sMAPE (%)','R2_micro':'R²'}))
    st.subheader('Ready-to-use model')
    with st.container(border=True):
        st.markdown('**CatBoost residual forecasting + LightGBM risk classifier**')
        st.write('The frozen official model is already trained and available on Prediction & Alert. Custom training is optional.')
        st.caption(f"Separate 18-month holdout starts {manifest['official_holdout_cutoff']}. See Advanced analysis for its metrics.")


def custom_training():
    st.subheader('Train your own model')
    st.caption('Choose an algorithm and optionally customize it. Results use a separate 18-month holdout and do not replace the official benchmark.')
    df=load_model_ready(); opts=get_filter_options(df)
    algorithms=available_algorithms()
    if not algorithms:
        st.error('No supported algorithms are installed.'); return
    with st.form('custom_training_form'):
        algorithm=st.selectbox('Forecasting algorithm', algorithms, index=algorithms.index('CatBoost') if 'CatBoost' in algorithms else 0)
        st.caption('Default: residual forecasting, all countries and commodities, 150 iterations, learning rate 0.05, depth 6.')
        with st.expander('Optional settings'):
            target_mode=st.radio('Forecast approach',['residual','direct'],format_func=lambda x:'Residual (predict price change)' if x=='residual' else 'Direct (predict next price)',horizontal=True)
            a,b=st.columns(2)
            countries=a.multiselect('Countries (empty = all)',opts['countries'])
            commodities=b.multiselect('Commodities (empty = all)',opts['commodities'])
            a,b,c=st.columns(3)
            n=a.slider('Iterations',50,600,150,25)
            rate=b.select_slider('Learning rate',options=[.01,.02,.03,.05,.08,.1,.15],value=.05)
            depth=c.slider('Max depth',3,12,6)
            tune=st.checkbox('Tune hyperparameters (25 Optuna trials; slower)',value=False)
        run=st.form_submit_button('Train model',type='primary',use_container_width=True)
    if run:
        mask=pd.Series(True,index=df.index)
        if countries: mask &= df.countryiso3.isin(countries)
        if commodities: mask &= df.commodity_group.isin(commodities)
        if mask.sum()<200:
            st.error('Too few observations. Select more countries or commodities.')
        else:
            with st.spinner('Training and evaluating model…'):
                try:
                    out=train_and_evaluate(df.loc[mask].copy(),algorithm=algorithm,target_mode=target_mode,n_estimators=n,learning_rate=rate,max_depth=depth,tune_trials=25 if tune else 0)
                    bundle=make_bundle(out)
                    st.session_state['two_stage_bundle']=bundle
                    st.session_state['trained_metrics']=out['metrics']
                    st.session_state['training_display']={'fit_end':out['fit_end'],'cal_end':out['cal_end'],'cutoff':out['cutoff'],'threshold':out['threshold'],'actual':out['y_true'],'predicted':out['y_pred']}
                    st.success('Training complete. Your model is ready for Prediction & Alert → Custom trained model what-if.')
                except Exception as exc:
                    st.error(f'Training failed: {exc}. Any previously trained model is unchanged.')
    m=st.session_state.get('trained_metrics')
    if m is None:
        st.info('No custom model trained yet. Click Train model to begin, or use the official model without training.'); return
    st.subheader('Latest successful training')
    st.caption(f"{m['model']} · {m['n_train']:,} fit · {m['n_calibration']:,} calibration · {m['n_test']:,} test observations")
    a,b,c,d=st.columns(4)
    a.metric('RMSE',f"{m['RMSE']:.4f}",f"{m['RMSE']-m['Persistence_RMSE']:+.4f} vs baseline",delta_color='inverse')
    b.metric('MAE',f"{m['MAE']:.4f}")
    c.metric('Shock precision',f"{m['Precision']:.1%}")
    d.metric('Shock recall',f"{m['Recall']:.1%}")
    st.caption('Use Prediction & Alert → Custom trained model what-if to explore this model.')
    with st.expander('Detailed results and forecast plot'):
        table(pd.DataFrame([{'Model':m['model'],'MAE':m['MAE'],'RMSE':m['RMSE'],'sMAPE':m['sMAPE'],'R²':m['R2']},{'Model':'Persistence','MAE':m['Persistence_MAE'],'RMSE':m['Persistence_RMSE'],'sMAPE':m['Persistence_sMAPE'],'R²':m['Persistence_R2']}]))
        st.write({k:m[k] for k in ('Precision','Recall','F1','FPR','TP','FP','FN','TN','PR_AUC','ROC_AUC','PICP_P10_P90')})
        detail=st.session_state.get('training_display')
        if detail:
            st.caption(f"Fit ends {detail['fit_end']} · calibration ends {detail['cal_end']} · holdout starts {detail['cutoff']} · threshold {detail['threshold']:.3f}")
            y,p=detail['actual'],detail['predicted']; fig,ax=plt.subplots(figsize=(5,3.5))
            ax.scatter(y,p,s=10,alpha=.25,color=COLORS['primary']); lim=[min(y.min(),p.min()),max(y.max(),p.max())]; ax.plot(lim,lim,'--',color=COLORS['coral']); ax.set(xlabel='Actual (USD)',ylabel='Predicted (USD)'); fig.tight_layout(); st.pyplot(fig); plt.close(fig)


def advanced(results,shock,manifest):
    st.subheader('Advanced analysis')
    st.caption('Full scientific evidence from the official benchmark and separate frozen holdout.')
    with st.expander('Full regression benchmark & statistical tests'):
        table(results)
    with st.expander('Shock classification & baseline comparisons'):
        table(shock); st.caption('PR-AUC and ROC-AUC apply to continuous ranking scores, not the binary hybrid decision.')
    with st.expander('Performance by market segment'):
        choice=st.radio('Group by',['Commodity','Country','Price type'],horizontal=True)
        table({'Commodity':load_eval_by_commodity,'Country':load_eval_by_country,'Price type':load_eval_by_pricetype}[choice]())
    with st.expander('Feature ablation and SHAP'):
        table(pd.read_csv(DATA/'v9_feature_ablation.csv')); st.caption('Controlled ablation with the same fit periods and tuned parameters.')
        st.image(str(FIG/'shap_summary.png'),caption='SHAP on the exact frozen official CatBoost holdout model.',use_container_width=True)
    with st.expander('Frozen holdout model & configuration'):
        st.caption(f"Fit through {manifest['official_fit_end']} · calibration {manifest['official_cal_start']} to {manifest['official_cal_end']} · holdout starts {manifest['official_holdout_cutoff']}")
        table(pd.DataFrame([manifest['official_holdout_metrics']]))
        st.json({'parameters':manifest['official_params'],'threshold':manifest['official_threshold'],'qcal':manifest['official_qcal']},expanded=False)
    with st.expander('Temporal splits & validation thresholds'):
        table(pd.read_csv(DATA/'v9_fold_metrics.csv'))


def render():
    inject_css(); apply_plot_style()
    st.markdown(hero('Model Training & Evaluation','Explore model performance or train your own forecasting model.',pills=[('7-Fold Walk-Forward','pill-sky'),('Official model ready','pill-green')]),unsafe_allow_html=True)
    page=st.radio('Training workspace',['Overview','Train a model','Advanced analysis'],horizontal=True,label_visibility='collapsed')
    if page=='Train a model': custom_training()
    else:
        results=load_model_results(); shock=load_shock_metrics(); manifest=load_manifest()
        if page=='Overview': overview(results,shock,manifest)
        else: advanced(results,shock,manifest)
    st.markdown('<p class="footer-note">FoodShock · MLSC 2026 · SDG 2 — Zero Hunger</p>',unsafe_allow_html=True)
