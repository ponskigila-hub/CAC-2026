import calendar
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
from src.forecast import ROOT,FEATURES,HORIZONS,eligible,forecast_path,load

st.set_page_config(page_title='FoodShock Indonesia',page_icon='🌾',layout='wide')
st.markdown('''<style>.block-container{max-width:1120px;padding-top:2rem;} [data-testid="stMetric"]{background:#f2f6f3;border:1px solid #dce8df;border-radius:12px;padding:16px;} h1,h2,h3{letter-spacing:-.025em;} </style>''',unsafe_allow_html=True)
@st.cache_resource
def assets(): return load()
@st.cache_data
def reports(): return pd.read_csv(ROOT/'results/forecast_metrics.csv'),pd.read_csv(ROOT/'results/shock_metrics.csv')
@st.cache_data
def raw_audit():return pd.read_csv(ROOT/'data/raw/source_manifest.csv')
d,b=assets();metrics,risk=reports();q=risk[(risk.phase.str.startswith('Test'))&(risk.model==b['risk_selected'])&(risk.horizon.astype(str)=='All')].iloc[0];latest=eligible(d).sort_values('date').groupby('series',as_index=False).tail(1)
PAGES=['Overview','EDA','Preprocessing','Training & Testing','Prediction']
if 'page' not in st.session_state:st.session_state.page='Overview'
with st.sidebar:
    st.title('FoodShock')
    st.caption('INDONESIA · SDG 2.c')
    st.radio('Research workflow',PAGES,key='page')
    st.divider();st.caption('Data historis 2015–2025\n\nPrediksi bulanan, bukan harian.')
def go_prediction():st.session_state.page='Prediction'

def rupiah(x):return 'Rp'+f'{x:,.0f}'.replace(',','.')
def title(a,b):st.title(a);st.caption(b)

def overview():
    title('FoodShock Indonesia','Estimasi harga pangan dan risiko lonjakan harga pada bulan mendatang.')
    st.button('Mulai prediksi',type='primary',on_click=go_prediction)
    st.subheader('Latar belakang')
    st.write('Lonjakan harga pangan mengurangi keterjangkauan pangan bagi rumah tangga, khususnya yang memiliki anggaran terbatas. Pemantauan harga yang tepat waktu membantu analis mengidentifikasi pasar yang membutuhkan perhatian. FoodShock menggunakan pola historis harga retail dalam rupiah untuk memperkirakan harga dan risiko lonjakan pada bulan yang dipilih pengguna. Solusi ini mendukung SDG 2 (Zero Hunger), khususnya target 2.c mengenai informasi pasar dan volatilitas harga pangan.')
    a,c,e,f=st.columns(4);a.metric('Observasi bulanan',f'{len(d):,}');c.metric('Pasar',d.market_id.nunique());e.metric('Produk',d.commodity_id.nunique());f.metric('Horizon diuji','1–36 bulan')
    st.subheader('Rumusan masalah');st.write('Dapatkah pola historis harga pangan Indonesia 2015–2025 digunakan untuk memprediksi harga pada berbagai horizon dan mengidentifikasi potensi kenaikan bulanan minimal 5%?')
    st.subheader('Tujuan');st.write('Menghasilkan estimasi harga, perubahan harga bulanan, dan skor risiko shock pada bulan target; membandingkan model dengan baseline melalui evaluasi temporal; serta menyajikan hasil secara sederhana bagi analis pasar.')
    st.info('Dataset terakhir berada pada 2025. Prediksi masa depan bersifat kondisional; perubahan kebijakan, cuaca, pasokan, dan inflasi setelah periode data belum diketahui.')

def eda():
    title('Exploratory Data Analysis','Pahami distribusi dan pola harga historis sebelum pemodelan.')
    counts=d.assign(year=d.date.dt.year).groupby('year').size();st.subheader('Cakupan data tahunan');st.bar_chart(counts.rename('Observasi'))
    st.caption('Tahun dengan cakupan rendah bukan berarti harga stabil atau tidak ada perdagangan; hanya data sumber yang terbatas.')
    a,c=st.columns(2);a.bar_chart(d.groupby('province').size().sort_values(ascending=False).head(12).rename('Observasi'));c.bar_chart(d.groupby('commodity').size().sort_values(ascending=False).head(12).rename('Observasi'))
    st.subheader('Frekuensi shock historis');known=d[d.mom1.notna()].copy();known['shock']=(known.mom1>=np.log(1.05)-1e-10).astype(int);st.line_chart(known.groupby('date').shock.mean().mul(100).rename('Shock rate (%)'));st.caption('Hanya pasangan harga bulanan berurutan; bulan tanpa label tidak dianggap non-shock.')
    st.subheader('Riwayat harga satu pasar');s=select_series('eda');h=d[d.series.eq(s)].sort_values('date');st.line_chart(h.set_index('date').price);st.caption('IDR per satuan produk asli; tidak menggabungkan satuan berbeda.')
    with st.expander('Sumber data dan contoh baris'):st.dataframe(raw_audit(),hide_index=True);st.dataframe(d[['date','province','market','commodity','unit','price']].head(20),hide_index=True)

def preprocessing():
    title('Preprocessing','Pipeline yang sama digunakan dalam notebook, training dan aplikasi.')
    st.subheader('Persiapan data');st.markdown('1. Pilih Indonesia, harga retail, mata uang IDR, komoditas pangan dan harga positif.\n2. Gabungkan observasi dalam bulan yang sama menggunakan median; pertahankan identitas pasar, produk dan satuan.\n3. Hitung momentum, volatilitas, tren dan seasonality hanya dari riwayat yang tersedia.\n4. Bentuk pasangan origin–target untuk horizon 1 sampai 36 bulan.\n5. Label shock = kenaikan harga aktual bulan target dibandingkan bulan sebelumnya ≥5%; label kosong jika bulan sebelumnya tidak tersedia.')
    a,c,e=st.columns(3);a.metric('Monthly records',f'{len(d):,}');c.metric('Eligible origins',f'{len(eligible(d)):,}');e.metric('Feature count',len(FEATURES));st.subheader('Temporal evaluation')
    st.dataframe(pd.read_csv(ROOT/'results/split_audit.csv'),hide_index=True,use_container_width=True)
    st.caption('Tanggal target, bukan tanggal origin saja, membatasi label training. Origin validasi/test harus memiliki riwayat yang cukup. Tidak ada imputasi harga hilang.')
    with st.expander('Feature engineering'):st.code('\n'.join(FEATURES),language=None);st.dataframe(d[['date','market','commodity','price','mom1','mom3','mom6','mom12','vol3','trend']].head(25),hide_index=True)

def training():
    title('Training & Testing','Bandingkan hasil model pada empat origin backtest. Model otomatis dipilih dari validasi.')
    options=[b['selected'],'Persistence','Seasonal Naive']+list(b['price_models'])+list(b.get('blends',{}));alg=st.selectbox('Model harga',options,key='train_model')
    test=metrics[metrics.phase.str.startswith('Test')];row=test[(test.model==alg)&(test.horizon.astype(str)=='All')].iloc[0];a,c,e=st.columns(3);a.metric('MAPE',f'{row.MAPE:.2f}%');c.metric('RMS log ratio',f'{row.RMSLE:.3f}');e.metric('Pasangan test',f'{int(row.n):,}')
    st.subheader('Perbandingan model harga');st.bar_chart(test[test.horizon.astype(str)=='All'].set_index('model').MAPE);st.caption('MAPE lebih kecil lebih baik. Pemilihan otomatis menggunakan MAPE validasi per kelompok horizon, bukan hasil test.')
    st.dataframe(pd.read_csv(ROOT/'results/deployment_policy.csv')[['horizon','price_model','validation_n']],hide_index=True,use_container_width=True)
    st.subheader('Shock classifier · '+b['risk_selected']);a,c,e,f=st.columns(4);a.metric('Precision',f'{q.precision:.1%}');c.metric('Recall',f'{q.recall:.1%}');e.metric('ROC-AUC',f'{q.AUC:.3f}');f.metric('False positive rate',f'{q.FPR:.1%}')
    st.caption('Ambang setiap kelompok horizon dipilih dengan FPR validasi ≤10%. Metrik test memakai ambang validasi; model deployment memakai ambang OOF historis hingga 2025 dan belum diuji pada masa depan.')
    with st.expander('Confusion matrix & performa per horizon'):
        st.dataframe(pd.DataFrame([[int(q.TN),int(q.FP)],[int(q.FN),int(q.TP)]],index=['Actual non-shock','Actual shock'],columns=['Predicted non-shock','Predicted shock']));st.dataframe(risk[risk.phase.str.startswith('Test')],hide_index=True);st.dataframe(test[test.model==alg],hide_index=True);st.dataframe(pd.read_csv(ROOT/'results/interval_coverage.csv'),hide_index=True)
    with st.expander('Advanced analysis · tuning, ablation & rolling origin'):
        for file in ['tuning_price','tuning_risk','ablation','sampling_sensitivity','rolling_origin','rolling_risk','importance','segment_metrics','deployment_policy']:
            st.markdown('**'+file.replace('_',' ').title()+'**');st.dataframe(pd.read_csv(ROOT/f'results/{file}.csv'),hide_index=True)
    with st.expander('Latih ulang backtest (opsional)'):
        st.caption('Eksperimen cutoff Desember 2022 memakai konfigurasi hasil validasi; tidak menimpa model deployment.')
        if st.button('Jalankan training dan testing',key='retrain'):
            from src.train import fit_one,evaldata,report,auto,blend,BLENDS,BASE,AUTO
            from src.forecast import pairs,predict_price
            with st.spinner('Melatih model dan mengevaluasi target 2023–2025…'):
                tr=pairs(d,'2022-12-01');te=evaldata(d,'2022-12-01',2025*12+12)
                for n,cfg in b['price_configs'].items():te[n]=predict_price(fit_one(tr,cfg),te[FEATURES],te.base)
                blend(te);auto(te,b['price_policy']);st.session_state['backtest_result']=pd.DataFrame(report(te,BASE+list(b['price_configs'])+list(BLENDS)+[AUTO],'Interactive backtest'))
        if 'backtest_result' in st.session_state:st.dataframe(st.session_state['backtest_result'].query('model == @alg'),hide_index=True)
    st.caption('Rebuild seluruh penelitian: REBUILD=True pada notebook atau python -m src.train, kemudian restart Streamlit.')

def select_series(prefix):
    pool=latest if prefix=='eda' else latest[latest.date.dt.year==2025]
    regions=sorted(pool.province.unique());rk=prefix+'_region';ck=prefix+'_commodity';sk=prefix+'_series'
    if st.session_state.get(rk) not in regions:st.session_state[rk]='JAWA BARAT' if 'JAWA BARAT' in regions else regions[0]
    loc=st.selectbox('Provinsi / cakupan',regions,key=rk);z=pool[pool.province.eq(loc)];commodities=sorted(z.commodity.unique())
    if st.session_state.get(ck) not in commodities:st.session_state[ck]='Rice (medium quality)' if 'Rice (medium quality)' in commodities else commodities[0]
    commodity=st.selectbox('Komoditas',commodities,key=ck);z=z[z.commodity.eq(commodity)].sort_values(['market','unit']);opts=z.series.tolist()
    if st.session_state.get(sk) not in opts:st.session_state[sk]=opts[0]
    names={int(r.series):f'{r.market} · {r.unit} · {r.date:%b %Y} (ID {r.series})' for r in z.itertuples()}
    return st.selectbox('Pasar dan satuan',opts,format_func=lambda x:names[x],key=sk)

def prediction():
    title('Prediksi Harga Pangan','Pilih bulan target dan pasar. Tidak perlu memasukkan harga manual.')
    a,c,e=st.columns([1,1,1]);year=int(a.number_input('Tahun target',min_value=2026,max_value=2100,value=2028,step=1));month=c.selectbox('Bulan target',range(1,13),format_func=lambda x:calendar.month_name[x]);day=e.selectbox('Tanggal target',range(1,calendar.monthrange(year,month)[1]+1),index=calendar.monthrange(year,month)[1]-1,key=f'target_day_{year}_{month}');target=pd.Timestamp(year=year,month=month,day=day)
    st.caption(f'Target: {target:%d %B %Y}. Dataset bulanan, sehingga tanggal dalam bulan yang sama menghasilkan estimasi yang sama.')
    st.caption('Pilihan pasar dibatasi pada riwayat terakhir yang memenuhi syarat di 2025. Semua series historis dapat dijelajahi di EDA.')
    s=select_series('pred');alg=st.selectbox('Model harga',[b['selected'],'Persistence','Seasonal Naive']+list(b['price_models'])+list(b.get('blends',{})),key='pred_model',help='Automatic memilih model dari validasi sesuai horizon.')
    if not st.button('Prediksi sekarang',type='primary',use_container_width=True):return
    origin,path=forecast_path(d,s,target,b,alg);r=path.iloc[-1]
    st.divider();st.subheader(f'Hasil prediksi · {r.date:%B %Y}');st.caption(f'{origin.market} · {origin.commodity} · {origin.unit} | Data acuan terakhir: {origin.date:%B %Y} | Horizon {int(r.horizon)} bulan')
    a,c,e=st.columns(3);a.metric('Estimasi harga',rupiah(r.price));c.metric('Perubahan vs bulan sebelumnya',f'{r.change:+.2%}');e.metric('Harga bulan sebelumnya',rupiah(r.price/(1+r.change)))
    if r.extrapolation:
        st.warning('Target melampaui horizon 36 bulan: hasil berupa ekstrapolasi musiman dengan tren teredam, bukan prediksi ML yang telah divalidasi. Risk score tidak tersedia.');st.write('Sinyal shock dari estimasi harga:', 'SHOCK' if r.price_implied_shock else 'NON-SHOCK')
    else:
        if r.shock:st.warning('SHOCK DIPREDIKSI · Risiko kenaikan harga ≥5%')
        else:st.success('NON-SHOCK DIPREDIKSI · Skor berada di bawah ambang alert')
        st.metric('Risk score',f'{r.risk_score:.3f} / 1.000');st.caption(f'Model harga: {r.price_model} · Classifier: {b["risk_selected"]}');st.caption('Klasifikasi risiko berdiri sendiri dan dapat berbeda dari kenaikan estimasi harga titik. Non-shock bukan jaminan harga tidak naik.')
    if np.isfinite(r.lower):
        st.caption(f'Rentang kuantil galat OOF 10–90%: {rupiah(r.lower)} – {rupiah(r.upper)} (bukan jaminan coverage).')
    h=d[d.series.eq(s)&d.date.le(origin.date)].sort_values('date').tail(36);hist=h[['date','price']].rename(columns={'price':'Harga historis'}).set_index('date');pred=pd.concat([pd.DataFrame({'date':[origin.date],'Estimasi harga':[origin.price]}),path[['date','price']].rename(columns={'price':'Estimasi harga'})]).set_index('date');st.line_chart(hist.join(pred,how='outer'))
    with st.expander('Detail prediksi'):
        st.write('Price-implied shock:',bool(r.price_implied_shock));st.write('Ambang risk score:',r.threshold);st.dataframe(path,hide_index=True);st.download_button('Unduh forecast CSV',path.to_csv(index=False),'forecast.csv','text/csv')
    st.caption('Estimasi kondisional pada pola historis hingga 2025, bukan harga aktual atau prediksi dengan informasi baru setelah 2025.')

{'Overview':overview,'EDA':eda,'Preprocessing':preprocessing,'Training & Testing':training,'Prediction':prediction}[st.session_state.page]()
