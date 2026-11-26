import json
import pandas as pd
import streamlit as st
from src.core import ROOT, load, recommend, gap, skills as extract, explain, overlap, learning_rank

st.set_page_config(page_title='SkillBridge AI', page_icon='🎯', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container {max-width: 1180px; padding-top: 2.1rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(100,116,139,.17);}
[data-testid="stSidebar"] [data-testid="stRadio"] label {padding: .35rem .45rem; border-radius: .55rem;}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background: rgba(37,99,235,.08);}
[data-testid="stMetric"] {border: 1px solid rgba(100,116,139,.18); border-radius: .85rem; padding: .8rem 1rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
h1 {letter-spacing: -.035em;} h2,h3 {letter-spacing: -.018em;}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def bundle():
    return load()

@st.cache_data
def data():
    return (pd.read_csv(ROOT/'data/processed/jobs.csv'), pd.read_csv(ROOT/'results/metrics.csv'),
            json.loads((ROOT/'results/summary.json').read_text()))

@st.cache_data
def result(name):
    return pd.read_csv(ROOT/'results'/f'{name}.csv')

b=bundle(); d,m,s=data()
opts=[k for k in b['models'] if k!='Majority baseline']+['Hybrid']
best=b['best']; r=m[(m.model==best)&(m.split=='test')].iloc[0]
pages=['Overview','EDA','Preprocessing','Training & Testing','Prediction']
labels=dict(zip(pages,['01  Overview','02  Explore data','03  Preprocessing','04  Training & Testing','05  Career prediction']))
if 'page' not in st.session_state: st.session_state.page='Overview'
def go(page):
    st.session_state.page=page

def heading(k,title,subtitle):
    st.caption(f'SKILLBRIDGE  /  {pages.index(k)+1:02d} OF 05')
    st.title(title)
    st.caption(subtitle)

def footer(page):
    i=pages.index(page)
    st.divider(); a,c=st.columns(2)
    if i: a.button('← '+pages[i-1],on_click=go,args=(pages[i-1],),key='prev_'+page,use_container_width=True)
    if i<4: c.button(pages[i+1]+' →',on_click=go,args=(pages[i+1],),key='next_'+page,use_container_width=True)

with st.sidebar:
    st.markdown('## SkillBridge AI')
    st.caption('Career & skill intelligence')
    st.divider()
    st.caption('RESEARCH WORKFLOW')
    page=st.radio('Navigation',pages,format_func=lambda p:labels[p],key='page',label_visibility='collapsed')
    st.divider()
    st.button('Start career demo',type='primary',use_container_width=True,on_click=go,args=('Prediction',))
    st.caption('SDG 4.4 · SDG 8.5')
    st.caption('Indonesia · Research prototype')

if page=='Overview':
    heading(page,'Bridge your skills. Discover your career.','Career exploration grounded in historical Indonesian job advertisements.')
    st.button('Explore my career',type='primary',on_click=go,args=('Prediction',))
    a,c,e=st.columns(3); a.metric('Unique local profiles',len(d));c.metric('Career fields',len(s['classes']));e.metric('Macro F1 · test',f'{r.macro_f1:.1%}')
    st.subheader('Latar belakang')
    st.write('Ketidaksesuaian keterampilan dengan kebutuhan industri menyulitkan pencari kerja menentukan jalur pengembangan kompetensi. SkillBridge menggunakan lowongan historis Indonesia untuk membantu eksplorasi keluarga karier digital dan menyusun prioritas belajar.')
    st.markdown('[Konteks: Kemnaker](https://www.kemnaker.go.id/)')
    st.subheader('Rumusan masalah dan tujuan')
    st.write('Apakah profil keterampilan dapat mengidentifikasi keluarga pekerjaan? Apakah model ML dan hybrid mengungguli baseline? Apakah rekomendasi skill dapat memulihkan skill yang disembunyikan?')
    a,c=st.columns(2)
    with a:
        st.markdown('**SDG 4.4 · Quality Education**');st.write('Keterampilan relevan untuk pekerjaan.')
    with c:
        st.markdown('**SDG 8.5 · Decent Work**');st.write('Kesempatan kerja produktif.')
    st.info('Historical research prototype. Not a live job board or a measure of hiring success.')
elif page=='EDA':
    heading(page,'Explore the data','Understand the dataset before interpreting recommendations.')
    counts=d.career.value_counts(); a,c,e=st.columns(3);a.metric('Profiles',len(d));c.metric('Largest class',f'{counts.iloc[0]/len(d):.1%}');e.metric('Smallest class',f'{counts.iloc[-1]/len(d):.1%}')
    t1,t2,t3=st.tabs(['Career distribution','Skill demand','Data sources'])
    with t1:
        st.bar_chart(counts);table=counts.rename('Profiles').to_frame();table['Share']=table.Profiles/len(d);st.dataframe(table.style.format({'Share':'{:.1%}'}),use_container_width=True)
        st.warning('Software Development dominates the retained sample. It is not an estimate of national job demand. Evaluate minority classes using macro F1, balanced accuracy and per-class recall.')
    with t2:
        field=st.selectbox('Career field',['All fields']+sorted(d.career.unique()));subset=d if field=='All fields' else d[d.career==field];st.bar_chart(subset.skills.str.split('|').explode().value_counts().head(20))
    with t3:
        st.bar_chart(d.source.value_counts());st.dataframe(d[['title','career','skills','source']].head(30),hide_index=True,use_container_width=True)
elif page=='Preprocessing':
    heading(page,'From job text to skills','A reproducible pipeline with grouped evaluation.')
    for i,(title,desc) in enumerate([('Filter digital-sector jobs','Retain relevant records and audit source quality.'),('Assign career labels','Infer weak labels from titles, not from model inputs.'),('Extract skills','Normalize bilingual skill names and aliases.'),('Deduplicate and split','Group identical skill profiles across splits.'),('Build model features','Fit TF-IDF inside each training fold.')],1):
        with st.container(border=True): st.markdown(f'**{i:02d} · {title}**');st.caption(desc)
    st.subheader('Development and test distribution');st.dataframe(pd.crosstab(d.career,d.split),use_container_width=True)
    with st.expander('Source audit and limitations'):
        st.dataframe(result('data_audit'),hide_index=True);st.write('Near-duplicates and temporal overlap can remain. International augmentation is an experiment, not a source of Indonesian demand weights. Labels require independent human review.')
elif page=='Training & Testing':
    heading(page,'Model evaluation','Pretrained models; default selected using four-fold grouped development CV.')
    model=st.selectbox('Select algorithm',opts,index=opts.index(best),key='evaluation_model');row=m[(m.model==model)&(m.split=='test')].iloc[0]
    report=result('class_report');report=report[report.model==model];bal=report.recall.mean()
    cols=st.columns(4)
    for col,n,v in zip(cols,['Accuracy','Macro F1','Balanced accuracy','Top-3 hit'],[row.accuracy,row.macro_f1,bal,row.top3_hit]):col.metric(n,f'{v:.1%}')
    t1,t2,t3,t4=st.tabs(['Model comparison','Per-class results','Skill recommendation','Class imbalance'])
    with t1:
        test=m[m.split=='test'].set_index('model');st.bar_chart(test[['accuracy','macro_f1']]);st.dataframe(test.drop(columns='split').style.format('{:.3f}'),use_container_width=True)
    with t2:
        st.caption('Balanced accuracy is mean per-class recall. Support is the number of test profiles.')
        st.dataframe(report.drop(columns='model').style.format({'precision':'{:.1%}','recall':'{:.1%}','f1-score':'{:.1%}','support':'{:.0f}'}),hide_index=True,use_container_width=True)
        st.subheader('Confusion matrix');st.dataframe(pd.read_csv(ROOT/'results'/f'cm_{model}.csv',index_col=0),use_container_width=True);st.caption('Rows: actual title-derived family. Columns: predicted family.')
    with t3:
        ge=result('gap_evaluation');v3=result('gap_v3_evaluation');a,c,e=st.columns(3);a.metric('Hit@5',f'{v3.hit5.mean():.1%}');c.metric('Hit@10',f'{v3.hit10.mean():.1%}');e.metric('Global baseline Hit@5',f'{ge.global_hit.mean():.1%}');st.caption('One observed skill hidden. Retrospective evaluation, not hiring or learning success.')
    with t4:
        st.subheader('Balancing experiments')
        st.caption('Ten scenarios evaluated on the same local split. Original test has been used previously; results are retrospective. Only training folds are resampled. Local demand profiles remain unchanged.')
        bm=result('balance_metrics');bs=json.loads((ROOT/'results/balance_summary.json').read_text());st.info('CV winner: '+bs['selected']+' | Balanced alternative: '+bs['alternative'])
        chosen_split=st.radio('Evaluation split',['grouped CV','test'],horizontal=True,key='balance_split');view=bm[bm.split.eq(chosen_split)].set_index('method');st.bar_chart(view[['macro_f1','balanced_accuracy']]);st.dataframe(view.drop(columns='split').style.format('{:.2%}'),use_container_width=True)
        st.subheader('Minority augmentation');st.dataframe(result('balance_data_audit'),hide_index=True);st.write(bs['augmentation_counts'])
        selected_method=st.selectbox('Training distribution',bm.method.unique(),key='balance_method');dist=result('balance_training_distribution');st.bar_chart(dist[dist.method.eq(selected_method)].set_index('career').training_rows);st.caption('Counts reflect resampling, not unique profiles or national demand.')
        st.code('python -m src.balance',language='bash')
    with st.expander('Advanced evaluation and reproducibility'):
        st.code('python -m src.train',language='bash')
        for f in ['tuning','robustness','bootstrap_ci','gap_tuning','gap_v3_tuning','similarity_audit','label_audit','specialization_support','source_holdout']:
            st.markdown('**'+f.replace('_',' ').title()+'**');st.dataframe(result(f),hide_index=True)
    st.button('Try the model in Prediction',on_click=go,args=('Prediction',),type='primary')
else:
    heading(page,'Find your career direction','Select your skills, explore a career, and build a learning plan.')
    presets={'Custom':[],'Web':['HTML','CSS','JavaScript','React','Git'],'Data':['Python','SQL','Pandas','Statistics','Power BI'],'Infrastructure':['Linux','Docker','Networking'],'Design':['Figma','Adobe Photoshop'],'Marketing':['SEO','Google Analytics','Social media']}
    with st.container(border=True):
        st.subheader('1. Your skills')
        preset=st.selectbox('Quick-start profile',list(presets));known=st.multiselect('Select skills you already have',b['vocabulary'],default=[x for x in presets[preset] if x in b['vocabulary']],key='skills_'+preset)
        with st.expander('Additional input and model settings'):
            text=st.text_input('Additional skills (optional)');model=st.selectbox('Algorithm',opts,index=opts.index(best),key='prediction_model')
            if model=='Balancing experiment': st.caption('Moderate class weighting alternative; higher minority recall, lower macro F1 in development CV.')
        known=sorted(set(known+extract(text))&set(b['vocabulary']))
    ranked=recommend(b,known,model)
    if ranked.empty: st.info('Choose a quick-start profile or select at least one recognized skill.');st.stop()
    if len(known)<3: st.warning('Few skills: results are tentative.')
    st.subheader('2. Your recommended career fields')
    for col,row in zip(st.columns(3),ranked.head(3).itertuples(index=False,name=None)):
        with col:
            with st.container(border=True):
                st.markdown(f'**{row[0]}**');st.metric('Ranking score',f'{row[1]:.1%}');st.caption(f'Skill coverage: {row[2]:.1%}')
    st.caption('Scores are not hiring probabilities or skill proficiency. Coverage uses retained local development profiles, not all job postings.')
    career=st.selectbox('Career to explore',ranked.Career.tolist(),key='career_'+str(tuple(known))+'_'+model)
    g=gap(b,career,known);missing=learning_rank(b,career,known)
    st.subheader('3. Build your learning plan')
    t1,t2,t3=st.tabs(['Skill gap','Specializations','Learning simulator'])
    with t1:
        a,c=st.columns(2)
        with a:
            st.markdown('**Matched skills**');st.dataframe(g[g.Have][['Skill','Demand share']].style.format({'Demand share':'{:.1%}'}),hide_index=True,use_container_width=True)
        with c:
            st.markdown('**Top skills to learn**');st.dataframe(missing.head(8)[['Skill','Demand share','Learning priority score']].style.format({'Demand share':'{:.1%}','Learning priority score':'{:.3f}'}),hide_index=True,use_container_width=True)
    with t2:
        specs=b['specializations'].get(career,{})
        if specs: st.dataframe(pd.DataFrame([{'Specialization':k,'Coverage':overlap({k:v},known,[k])[0]} for k,v in specs.items()]).sort_values('Coverage',ascending=False).style.format({'Coverage':'{:.1%}'}),hide_index=True,use_container_width=True)
        else: st.info('No supported specialization for this field.')
        st.caption('Descriptive skill coverage, not a separately validated classifier.')
    with t3:
        p=b['profiles'][career];new=st.multiselect('Skills you plan to learn',[skill for skill in missing.Skill if skill in p],key='sim_'+career+'_'+str(tuple(known)));before=overlap({career:p},known,[career])[0];after=overlap({career:p},known+new,[career])[0];a,c=st.columns(2);a.metric('Current coverage',f'{before:.1%}');c.metric('Simulated coverage',f'{after:.1%}',f'{(after-before)*100:.1f} pp');st.progress(float(min(1,after)));st.caption('Coverage simulation is not a prediction of employment success.')
    with st.expander('Why this recommendation?'):
        st.dataframe(explain(b,known,career,model),hide_index=True);st.caption('Change in ranking score when a skill is removed; not causal evidence.')
    st.download_button('Export career ranking',ranked.to_csv(index=False),'skillbridge_ranking.csv','text/csv')
footer(page)
