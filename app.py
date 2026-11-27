import json
import pandas as pd
import streamlit as st
from src.course_matching import match_topics
from src.core import ROOT, load, recommend, gap, skills as extract, explain, overlap, learning_rank
st.set_page_config(page_title='SkillBridge', page_icon='🎯', layout='wide', initial_sidebar_state='expanded')
st.markdown(
    """<style>
.block-container {max-width: 1240px; padding: 4.5rem 2.5rem 3rem;}
[data-testid="stSidebar"] {border-right: 1px solid #dbe4ee;}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding-top: 1rem;}
[data-testid="stSidebar"] button {justify-content: flex-start; text-align: left; border-radius: 8px; min-height: 2.65rem; font-weight: 550;}
[data-testid="stSidebar"] button p {font-size: .91rem; width:100%; text-align:left;}
[data-testid="stSidebar"] button[kind="secondary"] {background: transparent; border:1px solid transparent; color:#334155;}
[data-testid="stSidebar"] button[kind="secondary"]:hover {background:#e2eaf4; color:#1d4ed8;}
[data-testid="stMetric"] {background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding: .85rem 1rem;}
[data-testid="stMetricValue"] {font-size:1.65rem; font-weight:650;}
[data-testid="stMetricLabel"] p {font-size:.84rem; color:#475569;}
h1 {font-size:2.15rem!important; line-height:1.22!important; letter-spacing:-.035em; padding:.15rem 0 .45rem!important;}
h2 {font-size:1.45rem!important; letter-spacing:-.02em;} h3 {font-size:1.16rem!important;}
[data-testid="stMarkdownContainer"] p {line-height:1.65;}
[data-testid="stCaptionContainer"] p {line-height:1.5; color:#64748b;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#e2e8f0;}
@media(max-width: 700px){.block-container {padding:4rem 1rem 2rem;} h1 {font-size:1.65rem!important;}}
</style>""",
    unsafe_allow_html=True,
)

@st.cache_resource
def bundle():
    return load()

@st.cache_data
def data():
    return (pd.read_csv(ROOT / 'data/processed/jobs.csv'),
        pd.read_csv(ROOT / 'results/metrics.csv'),
        json.loads((ROOT / 'results/summary.json').read_text()))

@st.cache_data
def result(name):
    return pd.read_csv(ROOT / 'results' / f'{name}.csv')

@st.cache_data
def result_profiles():
    return pd.read_csv(ROOT / 'data/processed/balance_augmentation.csv')
b = bundle()
d, m, s = data()
opts = [k for k in b['models'] if k != 'Majority baseline'] + ['Hybrid']
best = b['best']
r = m[(m.model == best) & (m.split == 'test')].iloc[0]
pages = ['Overview', 'EDA', 'Preprocessing', 'Training & Testing', 'Prediction']
labels = dict(zip(pages,
    ['01  Overview', '02  Explore data', '03  Preprocessing', '04  Training & Testing', '05  Career prediction']))
if 'page' not in st.session_state:
    st.session_state.page = 'Overview'

def go(page):
    st.session_state.page = page

def heading(k, title, subtitle):
    st.caption(f'{pages.index(k) + 1:02d} / 05  ·  {labels[k][4:]}')
    st.title(title)
    st.caption(subtitle)

def footer(page):
    i = pages.index(page)
    st.divider()
    a, c = st.columns(2)
    if i:
        a.button('← ' + pages[i - 1], on_click=go, args=(pages[i - 1],), key='prev_' + page, width="stretch")
    if i < 4:
        c.button(pages[i + 1] + ' →', on_click=go, args=(pages[i + 1],), key='next_' + page, width="stretch")
with st.sidebar:
    st.markdown('### SkillBridge')
    st.caption('Career intelligence · Indonesia')
    st.divider()
    st.caption('RESEARCH WORKFLOW')
    for item in pages:
        st.button(labels[item],
            key='nav_' + item,
            type='primary' if st.session_state.page == item else 'secondary',
            width="stretch",
            on_click=go,
            args=(item,))
    st.divider()
    st.caption('SUSTAINABLE DEVELOPMENT GOALS')
    st.markdown('**04** Quality Education  \n**08** Decent Work & Economic Growth')
    st.divider()
    st.caption('MLSC 2026 · PenaklukFCMobile')
page = st.session_state.page
if page == 'Overview':
    heading(page, 'SkillBridge', 'Rekomendasi karier dan pengembangan keterampilan berbasis data lowongan Indonesia.')
    a, c, e = st.columns(3)
    a.metric('Profil lokal', f'{len(d):,}')
    c.metric('Keluarga karier', len(s['classes']))
    e.metric('Macro F1 · test', f'{r.macro_f1:.2%}')
    with st.container(border=True):
        st.subheader('Temukan arah karier Anda')
        st.write('Pilih keterampilan yang sudah dimiliki, jelajahi bidang karier, dan susun prioritas pembelajaran yang relevan.')
        st.button('Mulai rekomendasi karier', type='primary', on_click=go, args=('Prediction',))
    st.subheader('Latar belakang dan relevansi')
    st.write(
        'Ketidaksesuaian keterampilan dengan kebutuhan industri menyulitkan mahasiswa, '
        'fresh graduate, dan pencari kerja dalam menentukan jalur pengembangan kompetensi. '
        'SkillBridge memetakan keterampilan dari lowongan historis Indonesia untuk membantu '
        'eksplorasi karier digital.'
    )
    st.markdown('[Kementerian Ketenagakerjaan RI](https://www.kemnaker.go.id/)')
    st.subheader('Tujuan penelitian')
    st.markdown(
        '1. Mengidentifikasi keluarga karier berdasarkan profil keterampilan.\n'
        '2. Membandingkan model ML, hybrid dan baseline serta strategi class imbalance.\n'
        '3. Menghasilkan rekomendasi skill dan simulasi cakupan kompetensi.'
    )
    st.subheader('Sustainable Development Goals')
    a, c = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown('**SDG 4 · Quality Education**')
            st.caption('Target 4.4')
            st.write('Mendukung pengembangan keterampilan teknis dan vokasional melalui analisis skill gap dan prioritas belajar.')
            st.markdown('[Tujuan 4](https://sdgs.un.org/goals/goal4)')
    with c:
        with st.container(border=True):
            st.markdown('**SDG 8 · Decent Work and Economic Growth**')
            st.caption('Target 8.5')
            st.write('Mendukung eksplorasi jalur karier dan kesiapan kompetensi menuju pekerjaan produktif dan layak.')
            st.markdown('[Tujuan 8](https://sdgs.un.org/goals/goal8)')
    st.subheader('Tim PenaklukFCMobile')
    for col, name in zip(st.columns(3),
        ['Christian Caesar', 'Gerald Amadeo Passover Sutanto', 'Nicholas Driyadis Tjoe']):
        with col:
            with st.container(border=True):
                st.write(name)
elif page == 'EDA':
    heading(page,
        'Exploratory Data Analysis',
        'Jelajahi profil lokal, kandidat augmentasi, distribusi kelas, dan keterampilan.')
    aug = result_profiles()
    local = d.assign(dataset='Profil lokal')
    extra = aug.assign(dataset='Augmentasi minoritas', split='Training candidate')
    all_profiles = pd.concat([local, extra], ignore_index=True)
    with st.container(border=True):
        a, c = st.columns(2)
        scope = a.selectbox('Dataset', ['Profil lokal', 'Augmentasi minoritas', 'Gabungan kandidat'], key='eda_scope')
        base = {'Profil lokal': local, 'Augmentasi minoritas': extra, 'Gabungan kandidat': all_profiles}[scope]
        source = c.selectbox('Sumber data', ['Semua sumber'] + sorted(base.source.unique()), key='eda_source_' + scope)
        a, c = st.columns(2)
        selected = base if source == 'Semua sumber' else base[base.source.eq(source)]
        field = a.selectbox('Keluarga karier',
            ['Semua bidang'] + sorted(selected.career.unique()),
            key='eda_field_' + scope + '_' + source)
        subset = selected if field == 'Semua bidang' else selected[selected.career.eq(field)]
        top = c.slider('Jumlah skill teratas', 5, 30, 15, key='eda_top')
        st.caption('Kandidat augmentasi hanya digunakan dalam eksperimen training; model utama dan test set lokal tidak berubah.')
    counts = subset.career.value_counts()
    tokens = subset.skills.str.split('|').explode()
    a, c, e = st.columns(3)
    a.metric('Profil terpilih', f'{len(subset):,}')
    c.metric('Keluarga karier', subset.career.nunique())
    e.metric('Skill unik', tokens.nunique())
    t1, t2, t3, t4 = st.tabs(['Distribusi karier', 'Keterampilan', 'Perbandingan dataset', 'Sumber & data explorer'])
    with t1:
        st.subheader('Distribusi keluarga karier')
        st.bar_chart(counts, horizontal=True)
        table = counts.rename('Profil').to_frame()
        table['Proporsi'] = table.Profil / len(subset)
        st.dataframe(table.style.format({'Proporsi': '{:.1%}'}), width="stretch")
        st.caption('Proporsi sampel yang diproses, bukan estimasi distribusi pekerjaan nasional.')
    with t2:
        st.subheader('Keterampilan paling sering muncul')
        freq = tokens.value_counts().head(top).rename('Profil')
        st.bar_chart(freq, horizontal=True)
        st.dataframe(freq.to_frame().assign(**{'Proporsi profil': freq / len(subset)}).style.format({'Proporsi profil': '{:.1%}'}),
            width="stretch")
        st.download_button('Unduh frekuensi skill', freq.to_csv(), 'skill_frequency.csv', 'text/csv')
    with t3:
        st.subheader('Profil lokal dan tambahan minoritas')
        comp = pd.crosstab(all_profiles.career,
            all_profiles.dataset).reindex(columns=['Profil lokal', 'Augmentasi minoritas'],
            fill_value=0)
        st.bar_chart(comp, horizontal=True)
        st.dataframe(comp.assign(Total=comp.sum(axis=1)), width="stretch")
        a, c = st.columns(2)
        a.metric('Profil lokal', f'{len(local):,}')
        c.metric('Kandidat tambahan', f'{len(extra):,}')
        st.subheader('Distribusi training per metode')
        bd = result('balance_training_distribution')
        method = st.selectbox('Metode balancing', bd.method.unique(), key='eda_method')
        st.bar_chart(bd[bd.method.eq(method)].set_index('career').training_rows, horizontal=True)
        st.caption('Distribusi training setelah resampling dapat mencakup profil berulang.')
    with t4:
        st.subheader('Sumber profil terpilih')
        st.bar_chart(subset.source.value_counts(), horizontal=True)
        audit_tab, aug_tab = st.tabs(['Audit sumber utama', 'Audit augmentasi'])
        with audit_tab:
            st.dataframe(result('data_audit'), hide_index=True, width="stretch")
        with aug_tab:
            st.dataframe(result('balance_data_audit'), hide_index=True, width="stretch")
        q = st.text_input('Cari judul pekerjaan atau keterampilan', key='eda_search')
        show = subset[['title', 'career', 'skills', 'source', 'dataset', 'split']]
        show = show[show.title.str.contains(q,
            case=False,
            regex=False,
            na=False) | show.skills.str.contains(q,
            case=False,
            regex=False,
            na=False)] if q else show
        st.caption(f'{len(show):,} profil ditemukan')
        st.dataframe(show, hide_index=True, width="stretch", height=360)
        st.download_button('Unduh profil terpilih', show.to_csv(index=False), 'selected_profiles.csv', 'text/csv')
elif page == 'Preprocessing':
    heading(page, 'From job text to skills', 'A reproducible pipeline with grouped evaluation.')
    for i, (title, desc) in enumerate([
        ('Filter digital-sector jobs', 'Retain relevant records and audit source quality.'),
        ('Assign career labels', 'Infer weak labels from titles, not from model inputs.'),
        ('Extract skills', 'Normalize bilingual skill names and aliases.'),
        ('Deduplicate and split', 'Group identical skill profiles across splits.'),
        ('Build model features', 'Fit TF-IDF inside each training fold.'),
    ], 1):
        with st.container(border=True):
            st.markdown(f'**{i:02d} · {title}**')
            st.caption(desc)
    st.subheader('Development and test distribution')
    st.dataframe(pd.crosstab(d.career, d.split), width="stretch")
    with st.expander('Source quality audit'):
        st.dataframe(result('data_audit'), hide_index=True)
        st.caption('Labels are derived from job titles; international profiles are used only in augmentation experiments.')
elif page == 'Training & Testing':
    heading(page, 'Model evaluation', 'Pretrained models; default selected using four-fold grouped development CV.')
    model = st.selectbox('Select algorithm', opts, index=opts.index(best), key='evaluation_model')
    row = m[(m.model == model) & (m.split == 'test')].iloc[0]
    report = result('class_report')
    report = report[report.model == model]
    bal = report.recall.mean()
    cols = st.columns(4)
    for col, n, v in zip(cols,
        ['Accuracy', 'Macro F1', 'Balanced accuracy', 'Top-3 hit'],
        [row.accuracy, row.macro_f1, bal, row.top3_hit]):
        col.metric(n, f'{v:.1%}')
    t1, t2, t3, t4, t5 = st.tabs(['Model comparison',
        'Per-class results',
        'Skill recommendation',
        'Class imbalance',
        'Statistical analysis'])
    with t1:
        test = m[m.split == 'test'].set_index('model')
        st.bar_chart(test[['accuracy', 'macro_f1']])
        st.dataframe(test.drop(columns='split').style.format('{:.3f}'), width="stretch")
    with t2:
        st.caption('Balanced accuracy is mean per-class recall. Support is the number of test profiles.')
        st.dataframe(report.drop(columns='model').style.format({'precision': '{:.1%}', 'recall': '{:.1%}', 'f1-score': '{:.1%}', 'support': '{:.0f}'}),
            hide_index=True,
            width="stretch")
        st.subheader('Confusion matrix')
        st.dataframe(pd.read_csv(ROOT / 'results' / f'cm_{model}.csv', index_col=0), width="stretch")
        st.caption('Rows: actual title-derived family. Columns: predicted family.')
    with t3:
        ge = result('gap_evaluation')
        v3 = result('gap_v3_evaluation')
        a, c, e = st.columns(3)
        a.metric('Hit@5', f'{v3.hit5.mean():.1%}')
        c.metric('Hit@10', f'{v3.hit10.mean():.1%}')
        e.metric('Global baseline Hit@5', f'{ge.global_hit.mean():.1%}')
        st.caption('One observed skill hidden. Retrospective evaluation, not hiring or learning success.')
    with t4:
        st.subheader('Balancing experiments')
        st.caption('Ten scenarios on the same retrospective local holdout. Resampling is applied only to training folds.')
        bm = result('balance_metrics')
        bs = json.loads((ROOT / 'results/balance_summary.json').read_text())
        st.info('CV winner: ' + bs['selected'] + ' | Balanced alternative: ' + bs['alternative'])
        chosen_split = st.selectbox('Evaluation split', ['grouped CV', 'test'], key='balance_split')
        view = bm[bm.split.eq(chosen_split)].set_index('method')
        st.bar_chart(view[['macro_f1', 'balanced_accuracy']])
        st.dataframe(view.drop(columns='split').style.format('{:.2%}'), width="stretch")
        st.subheader('Minority augmentation')
        st.dataframe(result('balance_data_audit'), hide_index=True)
        st.write(bs['augmentation_counts'])
        selected_method = st.selectbox('Training distribution', bm.method.unique(), key='balance_method')
        dist = result('balance_training_distribution')
        st.bar_chart(dist[dist.method.eq(selected_method)].set_index('career').training_rows)
        st.caption('Counts reflect resampling, not unique profiles or national demand.')
        st.code('python -m src.balance', language='bash')
    with t5:
        st.subheader('Paired group bootstrap')
        st.caption('1,000 paired group resamples; 95% unadjusted intervals. Positive values favor the reference. Intervals spanning zero do not establish a difference.')
        source = st.selectbox('Comparison', ['Classifiers', 'Balancing methods'])
        metric = st.selectbox('Metric', ['macro_f1', 'balanced_accuracy', 'accuracy'])
        ci = result('paired_model_ci' if source == 'Classifiers' else 'paired_balance_ci')
        ci = ci[ci.metric.eq(metric)]
        st.dataframe(ci, hide_index=True, width='stretch')
        st.subheader('Nested grouped cross-validation')
        st.caption('Four outer folds, three inner folds on development profiles; historical data and title-derived labels.')
        st.dataframe(result('nested_cv'), hide_index=True, width='stretch')
        st.subheader('Label quality')
        st.dataframe(result('profile_quality_summary'), hide_index=True, width='stretch')
        st.caption('Human label review is separate from agreement with title-derived labels.')
    with st.expander('Advanced evaluation and reproducibility'):
        st.code('python -m src.train', language='bash')
        for f in ['tuning',
            'robustness',
            'bootstrap_ci',
            'gap_tuning',
            'gap_v3_tuning',
            'similarity_audit',
            'label_audit',
            'specialization_support',
            'source_holdout']:
            st.markdown('**' + {'gap_v3_tuning': 'Neighborhood ranking tuning'}.get(f,
                f.replace('_', ' ').title()) + '**')
            st.dataframe(result(f), hide_index=True)
    st.button('Try the model in Prediction', on_click=go, args=('Prediction',), type='primary')
else:
    heading(page, 'Find your career direction', 'Select your skills, explore a career, and build a learning plan.')
    presets = {'Custom': [],
        'Web': ['HTML', 'CSS', 'JavaScript', 'React', 'Git'],
        'Data': ['Python', 'SQL', 'Pandas', 'Statistics', 'Power BI'],
        'Infrastructure': ['Linux', 'Docker', 'Networking'],
        'Design': ['Figma', 'Adobe Photoshop'],
        'Marketing': ['SEO', 'Google Analytics', 'Social media']}
    with st.container(border=True):
        st.subheader('1. Your skills')
        preset = st.selectbox('Quick-start profile', list(presets))
        known = st.multiselect('Select skills you already have',
            b['vocabulary'],
            default=[x for x in presets[preset] if x in b['vocabulary']],
            key='skills_' + preset)
        with st.expander('Additional input and model settings'):
            text = st.text_input('Additional skills (optional)')
            model = st.selectbox('Algorithm', opts, index=opts.index(best), key='prediction_model')
            if model == 'Balancing experiment':
                st.caption('Moderate class weighting alternative; higher minority recall, lower macro F1 in development CV.')
        known = sorted(set(known + extract(text)) & set(b['vocabulary']))
    ranked = recommend(b, known, model)
    if ranked.empty:
        st.info('Choose a quick-start profile or select at least one recognized skill.')
        st.stop()
    if len(known) < 3:
        st.caption('Tambahkan lebih banyak keterampilan untuk memperkaya profil.')
    st.subheader('2. Your recommended career fields')
    for col, row in zip(st.columns(3), ranked.head(3).itertuples(index=False, name=None)):
        with col:
            with st.container(border=True):
                st.markdown(f'**{row[0]}**')
                st.metric('Ranking score', f'{row[1]:.1%}')
                st.caption(f'Skill coverage: {row[2]:.1%}')
    st.caption('Scores are not hiring probabilities or skill proficiency. Coverage uses retained local development profiles, not all job postings.')
    career = st.selectbox('Career to explore', ranked.Career.tolist(), key='career_' + str(tuple(known)) + '_' + model)
    g = gap(b, career, known)
    missing = learning_rank(b, career, known)
    st.subheader('3. Build your learning plan')
    t1, t2, t3, t4 = st.tabs(['Skill gap', 'Specializations', 'Learning simulator', 'Learning topics'])
    with t1:
        a, c = st.columns(2)
        with a:
            st.markdown('**Matched skills**')
            st.dataframe(g[g.Have][['Skill', 'Demand share']].style.format({'Demand share': '{:.1%}'}),
                hide_index=True,
                width="stretch")
        with c:
            st.markdown('**Top skills to learn**')
            st.dataframe(missing.head(8)[['Skill', 'Demand share', 'Learning priority score']].style.format({'Demand share': '{:.1%}', 'Learning priority score': '{:.3f}'}),
                hide_index=True,
                width="stretch")
    with t2:
        specs = b['specializations'].get(career, {})
        if specs:
            specialization_table = pd.DataFrame([
                {'Specialization': k, 'Coverage': overlap({k: v}, known, [k])[0]}
                for k, v in specs.items()
            ]).sort_values('Coverage', ascending=False)
            st.dataframe(
                specialization_table.style.format({'Coverage': '{:.1%}'}),
                hide_index=True, width='stretch',
            )
        else:
            st.info('No supported specialization for this field.')
        st.caption('Descriptive skill coverage, not a separately validated classifier.')
    with t3:
        p = b['profiles'][career]
        new = st.multiselect('Skills you plan to learn',
            [skill for skill in missing.Skill if skill in p],
            key='sim_' + career + '_' + str(tuple(known)))
        before = overlap({career: p}, known, [career])[0]
        after = overlap({career: p}, known + new, [career])[0]
        a, c = st.columns(2)
        a.metric('Current coverage', f'{before:.1%}')
        c.metric('Simulated coverage', f'{after:.1%}', f'{(after - before) * 100:.1f} pp')
        st.progress(float(min(1, after)))
        st.caption('Coverage simulation is not a prediction of employment success.')
    with t4:
        st.caption('Historical curriculum topics matched to your missing skills; not verified live courses or enrollment links.')
        topics = match_topics(missing)
        if topics.empty:
            st.info('No matching learning topic in the retained catalog.')
        else:
            st.dataframe(topics, hide_index=True, width='stretch')
    with st.expander('Why this recommendation?'):
        st.dataframe(explain(b, known, career, model), hide_index=True)
        st.caption('Change in ranking score when a skill is removed; not causal evidence.')
    st.download_button('Export career ranking', ranked.to_csv(index=False), 'skillbridge_ranking.csv', 'text/csv')
footer(page)
