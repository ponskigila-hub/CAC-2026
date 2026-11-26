import json
import pandas as pd
import streamlit as st
from src.core import ROOT,load,recommend,gap,skills as extract,explain,overlap,learning_rank
st.set_page_config(page_title='SkillBridge AI',layout='wide')
@st.cache_resource
def bundle():return load()
@st.cache_data
def data():return pd.read_csv(ROOT/'data/processed/jobs.csv'),pd.read_csv(ROOT/'results/metrics.csv'),json.loads((ROOT/'results/summary.json').read_text())
b=bundle();d,m,s=data();opts=[k for k in b['models'] if k!='Majority baseline']+['Hybrid'];best=b['best'];r=m[(m.model==best)&(m.split=='test')].iloc[0]
with st.sidebar:
    st.title('SkillBridge AI');st.caption('Career exploration · Indonesia');page=st.radio('Navigation',['Overview','EDA','Preprocessing','Training & Testing','Prediction'],label_visibility='collapsed');st.divider();st.caption('SDG 4.4 · SDG 8.5 | Research prototype')
if page=='Overview':
    st.title('Bridge your skills. Discover your career.');st.subheader('Latar belakang');st.write('Ketidaksesuaian keterampilan dengan kebutuhan industri menyulitkan pencari kerja menentukan jalur pengembangan kompetensi. SkillBridge menggunakan lowongan historis Indonesia untuk membantu eksplorasi keluarga karier digital dan menyusun prioritas belajar.');st.markdown('[Konteks: Kemnaker](https://www.kemnaker.go.id/)');st.subheader('Rumusan masalah dan tujuan');st.write('Apakah profil keterampilan dapat mengidentifikasi keluarga pekerjaan? Apakah model ML dan hybrid mengungguli baseline? Apakah rekomendasi skill dapat memulihkan skill yang disembunyikan?');a,c,e=st.columns(3);a.metric('Profil lokal unik',len(d));c.metric('Career fields',len(s['classes']));e.metric('Top-3 hit (test)',f'{r.top3_hit:.1%}');st.info('SDG 4.4: keterampilan relevan untuk pekerjaan. SDG 8.5: kesempatan kerja produktif. Tidak mengukur keberhasilan rekrutmen.');st.caption('Data historis; bukan lowongan aktif atau representasi seluruh pasar kerja.')
elif page=='EDA':
    st.title('Explore the data');a,c=st.columns(2);a.subheader('Career fields');a.bar_chart(d.career.value_counts());c.subheader('Sources');c.bar_chart(d.source.value_counts());st.subheader('Common skills');st.bar_chart(d.skills.str.split('|').explode().value_counts().head(20));st.dataframe(d[['title','career','skills','source']].head(30),hide_index=True)
elif page=='Preprocessing':
    st.title('From job text to skills');st.dataframe(pd.read_csv(ROOT/'results/data_audit.csv'),hide_index=True);st.write('Digital-sector filtering → title-derived labels → bilingual skill extraction → exact deduplication → grouped split → TF-IDF fitted within each training fold. Titles and source are not features.');st.dataframe(pd.crosstab(d.career,d.split));st.warning('Labels are weak labels inferred from titles. Identical skill profiles are grouped; near-duplicates and temporal overlap can remain. International augmentation is an experiment, not a source of Indonesian demand weights.')
elif page=='Training & Testing':
    st.title('Model Training & Testing');st.caption('Pretrained models. Default selected by four-fold grouped development CV, not holdout. To reproduce training: python -m src.train.');model=st.selectbox('Algorithm',opts,index=opts.index(best));row=m[(m.model==model)&(m.split=='test')].iloc[0];cols=st.columns(4)
    for c,k,n in zip(cols,['accuracy','macro_f1','top3_hit','mrr'],['Accuracy','Macro F1','Top-3 hit','MRR']):c.metric(n,f'{row[k]:.1%}')
    st.dataframe(m[m.split=='test'].drop(columns='split').set_index('model').style.format('{:.3f}'));st.subheader('Confusion matrix');st.dataframe(pd.read_csv(ROOT/'results'/f'cm_{model}.csv',index_col=0));st.caption('Rows: actual title-derived family. Columns: predicted family.')
    ge=pd.read_csv(ROOT/'results/gap_evaluation.csv');st.subheader('Masked skill recovery (Hit@5)');a,c,e=st.columns(3);a.metric('End-to-end',f"{pd.read_csv(ROOT/'results/gap_v3_evaluation.csv').hit5.mean():.1%}");c.metric('Previous oracle (V2)',f'{ge.oracle_hit.mean():.1%}');e.metric('Global baseline',f'{ge.global_hit.mean():.1%}');st.caption('One observed skill hidden. Retrospective evaluation; not a measure of hiring or learning success.'); st.metric('Skill recovery Hit@10',f"{pd.read_csv(ROOT/'results/gap_v3_evaluation.csv').hit10.mean():.1%}")
    with st.expander('Advanced evaluation'):
        for f in ['tuning','class_report','robustness','bootstrap_ci','gap_tuning','gap_v3_tuning','similarity_audit','label_audit','specialization_support','source_holdout']:
            st.write(f);st.dataframe(pd.read_csv(ROOT/'results'/f'{f}.csv'))
else:
    st.title('Find your career direction');st.caption('Choose skills you already have. Explore up to three career fields and plan learning.')
    presets={'Custom':[],'Web':['HTML','CSS','JavaScript','React','Git'],'Data':['Python','SQL','Pandas','Statistics','Power BI'],'Infrastructure':['Linux','Docker','Networking'],'Design':['Figma','Adobe Photoshop'],'Marketing':['SEO','Google Analytics','Social media']};preset=st.selectbox('Example profile',list(presets));known=st.multiselect('Your skills',b['vocabulary'],default=[x for x in presets[preset] if x in b['vocabulary']],key='skills_'+preset);known=sorted(set(known+extract(st.text_input('Additional skills (optional)')))&set(b['vocabulary']));model=st.selectbox('Algorithm',opts,index=opts.index(best));ranked=recommend(b,known,model)
    if ranked.empty:st.info('Select at least one recognized skill.');st.stop()
    if len(known)<3:st.warning('Few skills: results are tentative.')
    st.subheader('Top 3 career fields')
    for c,row in zip(st.columns(3),ranked.head(3).itertuples(index=False,name=None)):
        c.metric(row[0],f'{row[2]:.0%} coverage');c.caption(f'Ranking score: {row[1]:.3f}')
    st.caption('Ranking score and coverage are not hiring probabilities or skill proficiency. Demand shares refer to retained local development profiles, not all job postings.')
    career=st.selectbox('Explore a field',ranked.Career.tolist());g=gap(b,career,known);missing=learning_rank(b,career,known);a,c=st.columns(2);a.subheader('Matched skills');a.dataframe(g[g.Have][['Skill','Demand share']],hide_index=True);c.subheader('Skills to learn');c.dataframe(missing.head(8)[['Skill','Demand share','Learning priority score']],hide_index=True)
    p=b['profiles'][career];new=st.multiselect('Learning simulator',[s for s in missing.Skill if s in p],key='sim_'+career);before=overlap({career:p},known,[career])[0];after=overlap({career:p},known+new,[career])[0];a,c=st.columns(2);a.metric('Current coverage',f'{before:.1%}');c.metric('Simulated coverage',f'{after:.1%}',f'{(after-before)*100:.1f} pp');st.progress(float(min(1,after)))
    with st.expander('Why this recommendation?'):
        st.dataframe(explain(b,known,career,model),hide_index=True);st.caption('Change in ranking score when a skill is removed; not causal evidence.')
    specs=b['specializations'].get(career,{}); 
    if specs:
        with st.expander('Explore specializations (descriptive, not validated classifier)'):
            st.dataframe(pd.DataFrame([{'Specialization':k,'Coverage':overlap({k:v},known,[k])[0]} for k,v in specs.items()]).sort_values('Coverage',ascending=False),hide_index=True)
    st.download_button('Export career ranking',ranked.to_csv(index=False),'skillbridge_ranking.csv','text/csv')
