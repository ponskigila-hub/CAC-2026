# FoodShock Indonesia

**Multi-Horizon Food Price & Shock Forecasting** · Machine Learning Solution Challenge 2026 · SDG 2: Zero Hunger (target 2.c)

## 1. Latar belakang

Lonjakan harga pangan mengurangi keterjangkauan pangan bagi rumah tangga, khususnya kelompok dengan anggaran terbatas. Pemantauan harga yang tepat waktu dapat membantu analis ketahanan pangan mengidentifikasi pasar dan komoditas yang memerlukan perhatian. FoodShock menggunakan riwayat harga retail Indonesia untuk menghasilkan estimasi harga dan risiko lonjakan pada bulan target. Solusi ini mendukung tujuan SDG 2.c mengenai informasi pasar pangan dan volatilitas harga pangan ekstrem.

## 2. Perumusan masalah

1. Bagaimana memanfaatkan riwayat harga pangan Indonesia 2015–2025 untuk memprediksi harga komoditas pada berbagai horizon waktu?
2. Bagaimana memperkirakan risiko kenaikan harga minimal 5% dibandingkan bulan sebelum target, tanpa menggunakan harga aktual yang belum diketahui?
3. Bagaimana mengukur kemampuan model dibandingkan baseline sederhana dan mengendalikan false alarm melalui pemilihan threshold berbasis validasi?

## 3. Tujuan dan manfaat

- Memprediksi **harga pangan pada bulan target** pilihan pengguna, serta estimasi persentase perubahan dibandingkan bulan sebelumnya.
- Menghasilkan **risk score dan klasifikasi shock/non-shock** untuk horizon yang telah dievaluasi.
- Membandingkan performa berbagai horizon secara temporal, bukan menggunakan random train/test split.
- Memberikan informasi pendukung kepada analis pasar, bukan keputusan otomatis untuk intervensi harga.

Pengguna cukup memilih tanggal target, provinsi, produk, pasar/satuan dan model harga. Tidak perlu mengisi harga acuan atau fitur manual. Contoh: tanggal target 31 Januari 2028 menggunakan model yang dipelajari dari data 2015–2025 untuk menghasilkan estimasi Januari 2028 dan Desember 2027. Harga aktual terakhir tetap menjadi titik awal matematis untuk menyatakan perubahan relatif, **bukan satu-satunya data yang digunakan untuk melatih model**.

## 4. Dataset dan ruang lingkup

Sumber: [Global Food Prices Database (WFP), Kaggle](https://www.kaggle.com/datasets/abhishekgupta56447/global-food-prices-database-wfp), uploader `abhishekgupta56447`, penyedia asli World Food Programme. Sebelas CSV tahunan 2015–2025 yang diberikan pengguna menghasilkan 327.163 baris Indonesia. File regional terkompresi tersedia di `data/raw/indonesia.csv.gz`; checksum SHA-256 seluruh sumber asli dan jumlah baris terdapat pada `source_manifest.csv`. Kesebelas checksum asli berhasil diverifikasi terhadap file sumber.

- 327.163 observasi bulanan setelah penyaringan/agregasi; 223 ID pasar, 29 ID komoditas dan 5.673 series market–product–unit–currency.
- 287.879 origin dengan minimal empat bulan observasi berurutan; 5.652 series yang memiliki origin layak.
- 1.720 series memiliki origin terakhir yang layak pada 2025 (1.710 pada Desember). Prediction membatasi pilihan pada series tersebut agar data awal tidak terlalu usang. EDA menampilkan cakupan historis yang lebih luas.
- Data tahunan tidak seimbang; 2015, 2018 dan 2019 sangat jarang dibandingkan 2020–2023. Tidak ada pengisian harga hilang untuk menciptakan riwayat buatan.
- Harga lokal IDR; unit dan kualitas produk tidak digabung secara sembarangan. Harga sumber berlabel actual maupun aggregate dipertahankan sebagai harga retail yang tercatat, bukan dianggap semuanya merupakan transaksi individual.

Tanggal unduh, lisensi redistribusi snapshot dan tanggal rilis aktual setiap observasi belum dikonfirmasi. Pastikan izin redistribusi sebelum pengumpulan eksternal. Data bulanan diperlakukan seolah tersedia setelah bulan selesai; publication lag tidak dimodelkan.

## 5. Metodologi

### 5.1 Data preparation dan feature engineering

1. Filter IDN, IDR, retail, pangan, harga positif dan finite.
2. Agregasi median per market ID, commodity ID, unit, currency dan bulan. Pertahankan identitas produk/pasar.
3. Hitung log momentum pada lag 1, 3, 6 dan 12 bulan, volatilitas tiga bulan, tren bulanan, jumlah riwayat, fitur kalender origin/target, horizon, interaksi horizon dengan tren, serta indikator missing lag. Lag harus berjarak tepat sesuai kalender; minimal empat observasi bulanan berurutan diperlukan. Fitur lag opsional yang hilang diisi nol disertai indikator missing. Tidak mengimputasi harga aktual.
4. Bentuk training pair origin–target pada horizon 1–36 bulan. Origin training menggunakan bulan kuartal (Maret, Juni, September, Desember), maksimal 6.000 origin terpilih secara deterministik untuk efisiensi. Maksimal 90.000 pasangan digunakan pada tahap fit. Seluruh origin eligible pada cutoff evaluasi digunakan untuk backtest, tanpa sampling test. Data dari seluruh rentang sumber dapat berkontribusi pada persiapan dan pool training; tidak semua baris menjadi training pair.
5. Bentuk label shock dari harga aktual bulan target dan bulan tepat sebelumnya: `shock = 1` jika `price_target / price_previous - 1 >= 0.05`; jika bulan sebelumnya tidak tercatat, label shock tidak diketahui dan dikeluarkan dari evaluasi klasifikasi.

### 5.2 Multi-horizon price regression

Target regresi adalah `log(P_target / P_origin)`. Model menerima fitur origin serta horizon target dan kalender target. Estimasi harga diperoleh dari `P_origin * exp(predicted_log_ratio)`. Training target diclip pada [-3, 3] untuk membatasi efek outlier; output log ratio juga dibatasi [-5, 5] sebagai perlindungan numerik. Dibandingkan Ridge Regression, Histogram Gradient Boosting Regressor, Persistence, dan Seasonal Naive (fallback persistence jika referensi bulan musiman tidak tersedia).

Pemilihan model ML berdasarkan weighted MAPE validasi (bukan berdasarkan test). Model harga HGB dipilih di antara dua model ML; Persistence tetap baseline terbaik secara agregat pada backtest. Dua algoritma harga tersedia untuk pengguna. Regresi langsung menggunakan horizon sebagai fitur, bukan mengulang satu snapshot klasifikasi dari masa lalu. Prediksi perubahan harga pada target dihitung terhadap estimasi bulan sebelumnya (atau harga aktual origin jika h=1).

### 5.3 Shock risk classification

Model terpisah mempelajari label kenaikan aktual minimal 5% pada bulan target dari fitur origin dan horizon. Kandidat Logistic Regression dan Histogram Gradient Boosting Classifier dipilih berdasarkan recall pada threshold dengan FPR validasi <=10% (minimal 30 alert jika tersedia). HGB classifier terpilih; threshold 0,4453752072505741 ditetapkan dari prediksi validasi gabungan. Threshold kemudian digunakan pada backtest dan deployment refit tanpa tuning terhadap hasil test. Risk score bukan probabilitas yang dikalibrasi. Klasifikasi shock dan price-implied shock dari perubahan estimasi harga titik dapat berbeda, karena keduanya menjawab pertanyaan statistik yang berbeda. UI menampilkan klasifikasi utama dari classifier dan price-implied shock pada detail.

### 5.4 Temporal evaluation

| Tahap | Cutoff label training / origin | Target evaluasi |
|---|---|---|
| Validation A | Desember 2020 | Januari 2021–Desember 2022 |
| Validation B | Desember 2021 | Januari–Desember 2022 |
| Retrospective test | Desember 2022 | Januari 2023–Desember 2025 |
| Deployment refit | Desember 2025 | Januari 2026 dan setelahnya |

Data target training wajib tidak melampaui cutoff. Pooled validation menghitung forecast dari origin berbeda sebagai kasus berbeda walaupun beberapa bulan target berulang. Model selection dan threshold menggunakan validasi 1–24 bulan. Backtest mencakup 1–36 bulan, dengan jumlah observasi yang tidak sama pada setiap horizon akibat cakupan data. Periode 2023–2025 telah digunakan dalam eksperimen proyek sebelumnya, sehingga hasil ini merupakan **evaluasi retrospektif**, bukan klaim test set baru yang sepenuhnya untouched.

### 5.5 Ketidakpastian dan horizon panjang

Rentang empiris berbasis kuantil ke-10 dan ke-90 log residual validasi ditampilkan sebagai rentang error historis, **bukan jaminan confidence/prediction interval formal**. Untuk horizon 25–36, rentang 13–24 digunakan sebagai pendekatan dan coverage historis dilaporkan terpisah. Untuk target lebih dari 36 bulan, aplikasi melanjutkan seasonal pattern dari tahun terakhir hasil forecast dengan tren teredam. Ini merupakan ekstrapolasi yang belum divalidasi; risk score ML dan rentang empiris tidak ditampilkan. Sinyal shock pada horizon tersebut hanya dihitung dari perubahan estimasi harga. Batas tahun UI 2026–2100. Semua tanggal dalam satu bulan menghasilkan prediksi bulanan yang sama.

## 6. Hasil evaluasi

Hasil training aktual (lihat `results/forecast_metrics.csv`, `results/shock_metrics.csv`, `results/interval_coverage.csv`).

### Price forecasting (MAPE %, lebih rendah lebih baik)

| Model | Semua horizon | 1 bulan | 12 bulan | 24 bulan | 36 bulan |
|---|---:|---:|---:|---:|---:|
| Persistence | 13,86 | 5,82 | 16,91 | 15,47 | 16,42 |
| Seasonal Naive | 17,96 | 13,88 | 16,91 | 15,47 | 16,42 |
| Ridge | 20,06 | 11,04 | 15,03 | 15,07 | 14,07 |
| HGB | 16,86 | 5,88 | 14,17 | 13,49 | 12,91 |

104.875 target harga diketahui pada backtest. MAPE adalah mean absolute percentage error; RMSLE dalam CSV didefinisikan sebagai akar rata-rata kuadrat log rasio prediksi terhadap aktual. HGB lebih baik dari Persistence pada horizon 12, 24 dan 36 bulan, tetapi bukan secara agregat. Tidak ada klaim ML selalu mengalahkan baseline. Perbedaan error juga dipengaruhi coverage dan komposisi komoditas antarhorizon.

### Shock classification (102.621 label tersedia)

| Model | AP | ROC-AUC | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0,3114 | 0,6808 | 38,46% | 33,56% | 35,84% | 12,36% |
| HGB | 0,3250 | 0,6932 | 40,57% | 25,24% | 31,12% | 8,51% |

Prevalence shock test 18,70%; random/uninformative ranking memiliki expected AP mendekati prevalence. HGB memilih 11.943 alert pada label yang tersedia: TP 4.845, FP 7.098, FN 14.350, TN 76.328. HGB terpilih karena lebih baik pada sasaran recall dengan batas FPR **validasi**, bukan karena F1 test. Precision 73,995% pada horizon 1 bulan, 45,42% pada 24 bulan, 48,42% pada 36 bulan; jumlah sampel berbeda. FPR test tidak selalu di bawah 10% untuk setiap horizon. Risiko false alarm dan missed shock tetap ada.

### Contoh prediksi aktual dari frozen deployment model

Pasar Mundu, Jawa Barat, Rice (medium quality), KG. Harga tercatat Desember 2025 Rp15.000/kg. Target 31 Januari 2028 = horizon 25 bulan. HGB menghasilkan estimasi Desember 2027 Rp17.116/kg dan Januari 2028 Rp17.184/kg, perubahan +0,40%, risk score 0,061, klasifikasi non-shock. Rentang galat empiris Rp11.977–Rp22.595/kg (menggunakan pendekatan horizon 13–24). Ini adalah output model, bukan harga aktual masa depan.

## 7. Inovasi dan dampak

Satu antarmuka memungkinkan pengguna memilih tanggal target tanpa memasukkan data masa depan, membandingkan model harga, serta membaca risiko kenaikan harga dan keterbatasan horizon. Model shock memisahkan risiko kejadian ekstrem dari prediksi rata-rata harga. Output dapat menjadi bahan prioritisasi pemeriksaan oleh analis, bukan bukti manfaat intervensi atau validasi deployment nyata. Pengujian meliputi evaluasi temporal dan kontrol false alarm berbasis validation, bukan sekadar peningkatan kompleksitas model.

## 8. Struktur proyek

```text
FoodShock_Forecasting/
├── .streamlit/config.toml
├── data/
│   ├── raw/                  # Indonesia CSV + checksum manifest
│   └── processed/            # monthly.csv.gz
├── docs/                     # Methodology, results, rubric, validation
├── models/forecast.joblib    # deployment regression + risk classifier
├── notebooks/FoodShock_Indonesia.ipynb
├── results/                  # benchmark, interval, split audit, test predictions, example
├── src/
│   ├── extract.py            # ekstraksi raw global 2015–2025
│   ├── forecast.py           # shared preprocessing and inference
│   └── train.py              # training/validation/test/refit
├── tests/test_forecast.py
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

## 9. Instalasi

Gunakan Python 3.11 atau 3.12. Buka PowerShell di root folder proyek. Jika sudah memiliki virtual environment yang sesuai, aktifkan saja.

```powershell
python -m venv cac2026-venv
.\cac2026-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 10. Jupyter Notebook

```powershell
python -m jupyter lab
```

Buka `notebooks/FoodShock_Indonesia.ipynb`, pilih kernel dari `cac2026-venv`, lalu **Run All**. Notebook mengikuti lima tahap yang sama dengan Streamlit: Overview, EDA, Preprocessing, Training & Testing, Prediction. `REBUILD=False` memakai artifact dan hasil eksperimen yang disediakan; `REBUILD=True` menjalankan ulang `src.train` dari raw regional. Konfigurasi prediction default: `TARGET_DATE='2028-01-31'`, `SERIES_ID=868`, `ALGORITHM=None` (model ML terpilih).

## 11. Streamlit

```powershell
python -m streamlit run app.py
```

Buka http://localhost:8501. Pilih Prediction atau tombol Mulai prediksi. Ketik tahun target, pilih bulan/tanggal, provinsi, komoditas, pasar/satuan, dan model harga. Klik Prediksi sekarang. Tidak perlu training ulang untuk demo. Training & Testing menyediakan eksperimen backtest opsional yang tidak menimpa model deployment. Untuk rebuild resmi melalui notebook/CLI, restart Streamlit setelah selesai supaya cache model diperbarui.

## 12. Reproduksi penelitian melalui CLI

Dari root proyek, untuk mengulang training dan preprocessing dari data regional yang sudah tersedia:

```powershell
python -m src.train
python -m unittest discover -s tests -v
```

Untuk mengekstrak ulang dari 11 CSV global asli, letakkan semua file `wfp_food_prices_global_2015.csv` sampai `wfp_food_prices_global_2025.csv` dalam satu folder. Jalankan sebelum training:

```powershell
python -m src.extract "C:\path\to\raw_csvs"
python -m src.train
```

Notebook dan aplikasi menggunakan `src.forecast` dan model/CSV hasil yang sama. Hasil backtest menggunakan model cutoff 2022; frozen deployment model menggunakan data sampai 2025. Hasil keduanya tidak dicampur.

## 13. Keterbatasan

- Forecast masa depan bukan harga aktual dan tidak memasukkan kejadian setelah data 2025 (cuaca, kebijakan, supply shock, inflasi baru). Ketidakpastian meningkat dan tidak selalu monoton pada estimasi rentang empiris.
- Hanya sampai 36 bulan telah dievaluasi; horizon lebih panjang adalah ekstrapolasi. Prediksi 2028 layak dihitung dalam horizon ML jika origin masih pada 2025, namun performa masa depan tidak dijamin.
- Data Indonesia dan kualitas riwayat tidak seragam antarwilayah/produk. Data sumber tidak mencakup seluruh pasar Indonesia. Origin evaluasi dibatasi pada Desember, sedangkan training mencakup bulan kuartal; generalisasi untuk origin lain belum diuji menyeluruh.
- Persistence unggul dalam error agregat. Shock classifier memiliki recall terbatas, sehingga non-shock tidak berarti harga pasti aman.
- Threshold dipindahkan ke model refit deployment dan belum divalidasi pada outcome masa depan; FPR dapat berubah.
- Tidak ada validasi dengan pengguna nyata, audit dampak intervensi, atau jaminan redistribusi data.
- Streamlit browser end-to-end belum diuji dalam environment pengembangan; jalankan smoke test lokal sebelum submission.

## 14. Referensi

- United Nations, SDG 2: https://sdgs.un.org/goals/goal2 (khususnya target 2.c).
- World Food Programme, Global Food Prices Database, snapshot yang diberikan pengguna: https://www.kaggle.com/datasets/abhishekgupta56447/global-food-prices-database-wfp
- Scikit-learn, HistGradientBoostingRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
- Scikit-learn, HistGradientBoostingClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
- Scikit-learn, Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- Scikit-learn, LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Panduan penilaian: Guidebook Machine Learning Solution Challenge 2026, halaman 9–10; proyek difokuskan pada rubrik penyisihan (problem identification, data preparation, ML implementation, model evaluation, innovation).
