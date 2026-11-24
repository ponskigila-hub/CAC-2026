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

### 5.2 Direct multi-horizon price regression

Target regresi adalah `log(P_target/P_origin)`. Model menerima fitur origin, horizon, serta kalender target. Harga dihitung sebagai `P_origin * exp(predicted_log_ratio)`; target training diclip [-3, 3] dan output [-5, 5]. Kandidat: Ridge (alpha 1/10), tiga konfigurasi Histogram Gradient Boosting (7/15/31 leaves), Persistence, Seasonal Naive (fallback Persistence), dan shrinkage geometris HGB dengan bobot 25%, 50%, 75% pada log ratio. Shrinkage merupakan kombinasi sederhana yang membatasi koreksi harga terhadap baseline, bukan model tambahan yang kompleks.

Konfigurasi Ridge/HGB dipilih menurut MAPE validasi gabungan. Mode Automatic memilih kandidat dengan MAPE validasi terbaik per kelompok horizon 1–3, 4–6, 7–12, 13–36. Validasi awal hanya mencapai 24 bulan sehingga kebijakan 13–36 ditentukan dari bukti 13–24. Pengguna dapat memilih model harga tertentu secara eksplisit. Perubahan bulanan dihitung dari dua estimasi berturut-turut pada path; pada mode Automatic perubahan model antar kelompok dapat memengaruhi perubahan estimasi. Shock classifier berdiri sendiri.

### 5.3 Shock risk classification

Target = kenaikan aktual bulan target terhadap bulan sebelumnya ≥5%. Logistic Regression (C 0.1/1) dan tiga konfigurasi HGB diuji. Ambang berbeda untuk kelompok horizon 1–3, 4–6, 7–12, 13–36: maksimum recall di bawah batas FPR 10% pada validasi, minimal 30 alert apabila feasible; fallback tanpa alert bila tidak feasible. Model dipilih berdasarkan recall agregat pada kebijakan tersebut. Hasil test tidak digunakan untuk pemilihan konfigurasi, model harga atau threshold evaluasi retrospektif. Risk score bukan probabilitas terkalibrasi; FPR masa depan tidak dijamin. False alarm di antara alert = 1 − precision, berbeda dari FPR.

### 5.4 Rolling-origin evaluation

| Tahap | Cutoff label training / origin | Target maksimum |
|---|---|---|
| Validation | Des 2020, Jun 2021, Des 2021, Jun 2022 | Des 2022 |
| Retrospective test | Des 2022, Jun 2023, Des 2023, Jun 2024 | Des 2025 |
| Deployment refit | Des 2025 | Prediksi setelah origin |

Label training tidak boleh melewati cutoff. Pada setiap origin, model dilatih hanya dengan label yang tersedia hingga cutoff. Prediksi dari beberapa origin dapat memiliki bulan target sama, sehingga n adalah jumlah pasangan origin–target, bukan jumlah kejadian independen. Evaluasi ini retrospektif: data 2023–2025 sudah pernah dilihat dalam eksperimen versi sebelumnya. Origin Juni/Desember menambah cakupan dibanding hanya Desember, tetapi belum mewakili seluruh musim. Semua horizon 1–36 diuji secara gabungan, meskipun horizon panjang hanya berasal dari origin lebih awal.

Tuning: 5 konfigurasi regresi dan 5 klasifikasi, masing-masing pada 4 origin validasi. Feature ablation HGB standard: momentum+horizon (8), +volatilitas/tren (12), full (17). Sensitivitas cap pasangan training 30.000 vs 90.000 di origin Jun 2022. Permutation importance classifier pada sampel validasi terakhir (AP; 3 pengulangan). Analisis per provinsi dan komoditas tersedia. Tidak mengklaim seluruh fitur meningkatkan performa regresi.

### 5.5 Deployment dan ketidakpastian

Setelah evaluasi selesai, model dilatih ulang dengan pool label sampai Des 2025. Threshold deployment dan kuantil log error 10–90% diperbarui menggunakan prediksi out-of-fold historis validasi + backtest sampai 2025. Ini memperluas bukti untuk horizon 25–36, tetapi masih merupakan transfer dari model historis ke model refit, bukan kalibrasi independen terhadap artifact deployment. Tidak ada test setelah 2025. Metrik retrospektif tetap memakai threshold validasi awal, bukan threshold baru. Rentang empiris bukan jaminan interval prediksi formal. Di atas 36 bulan, aplikasi memakai ekstrapolasi musiman dengan tren teredam, tanpa risk score ML dan tanpa rentang empiris. Tahun target 2026–2100; semua tanggal di bulan sama memberi hasil yang sama.

## 6. Hasil evaluasi

Hasil rolling backtest aktual: 224.336 pasangan harga dan 217.627 pasangan dengan label shock. Fixed HGB blend 25% mencapai MAPE 12,28% vs Persistence 12,63%; Automatic yang dipilih dari validasi mencapai 13,68% dan belum mengalahkan Persistence. HGB shock classifier: AP 0,3191, ROC-AUC 0,7012, precision 38,33%, recall 22,22%, FPR 7,64%. FPR horizon 1–3 bulan masih 12,62%. Pada origin Desember 2022 yang identik dengan versi sebelumnya, FPR meningkat dari 8,51% menjadi 10,26% sambil recall meningkat dari 25,24% menjadi 28,58%. Oleh karena itu, tidak ada klaim bahwa false alarm turun pada perbandingan yang setara.

Contoh aktual: Pasar Mundu, beras medium, Januari 2028, Automatic: Rp15.528/kg, perubahan +0,11%, risk score 0,0602, Non-Shock, rentang empiris Rp13.702–21.342/kg. Bukan harga aktual.

Lihat `docs/RESULTS.md` untuk tabel hasil dan interpretasi serta `results/` untuk CSV lengkap.

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

Buka `notebooks/FoodShock_Indonesia.ipynb`, pilih kernel dari `cac2026-venv`, lalu **Run All**. Notebook mengikuti lima tahap yang sama dengan Streamlit: Overview, EDA, Preprocessing, Training & Testing, Prediction. `REBUILD=False` memakai artifact dan hasil eksperimen yang disediakan; `REBUILD=True` menjalankan ulang `src.train` dari raw regional. Konfigurasi prediction default: `TARGET_DATE='2028-01-31'`, `SERIES_ID=868`, `ALGORITHM=None` (mode Automatic (by horizon)).

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

Notebook dan aplikasi menggunakan `src.forecast` dan model/CSV hasil yang sama. Hasil backtest menggunakan empat model cutoff 2022–2024; frozen deployment model menggunakan data sampai 2025. Hasil keduanya tidak dicampur.

## 13. Keterbatasan

- Forecast masa depan bukan harga aktual dan tidak memasukkan kejadian setelah data 2025 (cuaca, kebijakan, supply shock, inflasi baru). Ketidakpastian meningkat dan tidak selalu monoton pada estimasi rentang empiris.
- Hanya sampai 36 bulan telah dievaluasi; horizon lebih panjang adalah ekstrapolasi. Prediksi 2028 layak dihitung dalam horizon ML jika origin masih pada 2025, namun performa masa depan tidak dijamin.
- Data Indonesia dan kualitas riwayat tidak seragam antarwilayah/produk. Data sumber tidak mencakup seluruh pasar Indonesia. Origin evaluasi mencakup Juni dan Desember, sedangkan training mencakup bulan kuartal; generalisasi untuk origin lain belum diuji menyeluruh.
- Model ML belum tentu mengungguli Persistence. Shock classifier memiliki recall terbatas, sehingga non-shock tidak berarti harga pasti aman.
- Threshold deployment diadaptasi dari prediksi OOF hingga 2025 dan belum divalidasi pada outcome masa depan; FPR dapat berubah.
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
