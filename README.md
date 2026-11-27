# SkillBridge

**Career Recommendation and Personalized Skill-Gap Analysis for Indonesia's Digital Job Market**

SkillBridge membantu mahasiswa, fresh graduate, dan pencari kerja mengeksplorasi karier digital berdasarkan keterampilan yang dimiliki, memahami keterampilan yang belum tercakup, dan menentukan prioritas belajar. Sistem menggunakan machine learning dan profil keterampilan dari lowongan historis Indonesia.

**Tim: PenaklukFCMobile**

| Anggota tim |
|---|
| Christian Caesar |
| Gerald Amadeo Passover Sutanto |
| Nicholas Driyadis Tjoe |

**Machine Learning Solution Challenge (MLSC) 2026**

## 1. Quick start

### Live demo (Streamlit)

**Deployment URL:** `https://skillbridge-penaklukfcmobile.streamlit.app/`

Placeholder: replace this URL with the actual public Streamlit deployment link after deployment. The local setup below is available independently of the hosted demo.

### Dua jalur penggunaan

| Jalur | Entry point | Tujuan |
|---|---|---|
| Jupyter Notebook | `notebooks/SkillBridge.ipynb` | Penelitian end-to-end: EDA, preprocessing, eksperimen, interpretasi dan prediksi yang dapat diedit. |
| Streamlit | `app.py` | Demonstrasi interaktif: navigasi lima tahap, pilihan model, analisis keterampilan dan simulasi belajar. |

Kedua jalur menggunakan pipeline `src/`, data dan model pretrained yang sama; bukan dua model yang dilatih terpisah. Notebook secara default membaca artefak penelitian; opsi training ulang tersedia. Pilih salah satu jalur setelah instalasi environment berikut.

**Prasyarat:** Python 3.11–3.13 (disarankan 3.13; scikit-learn 1.8.0), terminal, dan koneksi internet saat instalasi dependency. Tidak diperlukan GPU, API key, database, atau training ulang untuk demo.

Ekstrak arsip, masuk ke folder `SkillBridge` yang berisi `app.py`, lalu jalankan:

### Windows (PowerShell)

```powershell
python -m venv cac2026-venv
.\cac2026-venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Jika aktivasi diblokir oleh execution policy, gunakan interpreter virtual environment secara langsung:

```powershell
.\cac2026-venv\Scripts\python.exe -m pip install -r requirements.txt
.\cac2026-venv\Scripts\python.exe -m streamlit run app.py
```

### Linux / macOS

```bash
python3 -m venv cac2026-venv
source cac2026-venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Buka http://localhost:8501 jika browser tidak terbuka otomatis. Model pretrained dan hasil evaluasi sudah disertakan. Setelah instalasi, inferensi lokal tidak memerlukan internet.

### Jalur Jupyter Notebook

Setelah instalasi dependency, dari root proyek jalankan:

```bash
python -m ipykernel install --user --name cac2026-venv --display-name "Python (SkillBridge)"
python -m jupyter lab
```

Buka `notebooks/SkillBridge.ipynb`, pilih kernel **Python (SkillBridge)**, lalu **Run All**. Ubah `USER_SKILLS`, `MODEL_NAME`, `TARGET_CAREER`, dan `SKILLS_TO_LEARN` pada bagian Prediction untuk mencoba profil berbeda. `RETRAIN = False` adalah default; ubah ke `True` jika ingin menjalankan ulang eksperimen dari raw data. Training ulang memerlukan waktu lebih lama daripada demo.

## 2. Panduan demo untuk penilai

| Menu | Isi dan cara menggunakan |
|---|---|
| Overview | Latar belakang, tujuan, SDG dan ringkasan hasil. Klik **Explore my career** untuk demo langsung. |
| EDA | Distribusi keluarga karier, keterampilan dan sumber data. Perhatikan ketimpangan kelas. |
| Preprocessing | Pipeline ekstraksi skill, label, deduplikasi dan pembagian data. |
| Training & Testing | Pilih model; bandingkan accuracy, Macro F1, balanced accuracy, confusion matrix dan eksperimen balancing. |
| Prediction | Pilih preset **Data**, atau masukkan Python, SQL, Pandas, Statistics dan Power BI. Lihat tiga rekomendasi, pilih bidang, lalu eksplorasi skill gap, spesialisasi, learning simulator dan penjelasan skor. |

Navigasi menggunakan sidebar dan tombol Previous/Next. Pada Prediction, pengaturan algoritma berada di **Additional input and model settings**. Default adalah Hybrid; `Balancing experiment` adalah alternatif moderate class weighting. Hasil ranking dapat diunduh sebagai CSV.

## 3. Latar belakang dan tujuan

Ketidaksesuaian keterampilan dengan kebutuhan industri menyulitkan pencari kerja menentukan jalur pengembangan kompetensi. Judul lowongan yang bervariasi juga membuat pencarian berbasis keyword saja kurang informatif.

Pertanyaan penelitian:

1. Seberapa baik profil skill mengidentifikasi keluarga pekerjaan digital?
2. Apakah hybrid ML dan weighted skill overlap mengungguli baseline?
3. Bagaimana class weighting, resampling dan selective augmentation memengaruhi performa kelas minoritas?
4. Apakah rekomendasi berbasis kemiripan profil memperbaiki pemulihan skill yang disembunyikan?

### Relevansi studi kasus Indonesia

Mahasiswa dan fresh graduate dapat memiliki keterampilan lintas bidang (misalnya Python dan SQL) tanpa mengetahui jalur karier yang paling sesuai. Persyaratan dalam iklan lowongan tersebar di berbagai platform dan istilah yang tidak seragam. SkillBridge mengubah informasi tersebut menjadi rekomendasi keluarga karier, gap keterampilan dan prioritas belajar yang dapat ditelusuri. Target penelitian adalah kategori pekerjaan dari lowongan, bukan status diterima kerja. Konteks kelembagaan: [Kementerian Ketenagakerjaan RI](https://www.kemnaker.go.id/). Penelitian tidak mengestimasi prevalensi skill mismatch nasional.

### Sustainable Development Goals (SDGs)

| Tujuan dan target | Keterkaitan solusi | Bukti evaluasi |
|---|---|---|
| [SDG 4 – Quality Education](https://sdgs.un.org/goals/goal4), target 4.4 | Meningkatkan jumlah pemuda dan orang dewasa yang memiliki keterampilan relevan, termasuk keterampilan teknis dan vokasional, untuk pekerjaan, pekerjaan layak dan kewirausahaan. Skill gap dan prioritas belajar membantu mengidentifikasi topik kompetensi digital. | Masked-skill recovery mengukur kemampuan menemukan kembali keterampilan dalam lowongan, bukan peningkatan kemampuan TIK aktual (indikator 4.4.1). |
| [SDG 8 – Decent Work and Economic Growth](https://sdgs.un.org/goals/goal8), target 8.5 | Mendukung pekerjaan penuh dan produktif serta pekerjaan layak bagi semua, termasuk pemuda dan penyandang disabilitas, dan upah setara untuk pekerjaan bernilai setara. Rekomendasi karier membantu eksplorasi kebutuhan kompetensi industri. | Klasifikasi keluarga karier dan ranking diukur pada lowongan historis; tidak mengukur tingkat pengangguran, upah, kesetaraan akses atau kualitas pekerjaan (indikator 8.5.1/8.5.2). |

Kontribusi praktis adalah dukungan pengambilan keputusan belajar dan eksplorasi karier. Dampak aktual pada indikator SDG belum diuji dan tidak diklaim sebagai hasil penelitian.

## 4. Dataset

Model utama menggunakan tiga sumber lokal: Job Description and Salary in Indonesia, Web Developer Dataset, dan Glints Indonesia. Setelah filtering dan deduplikasi, terdapat 2.838 profil lokal: 2.271 development dan 567 test. Vocabulary terdiri dari 111 keterampilan pada development set.

| Keluarga karier | Profil lokal |
|---|---:|
| Software Development | 2.065 |
| Digital Marketing | 248 |
| Infrastructure & Security | 219 |
| Data & AI | 168 |
| Design | 138 |
| Total | 2.838 |

Eksperimen selective augmentation memperoleh 1.315 kandidat profil minoritas: 620 Infrastructure & Security, 596 Data & AI, 99 Design. Tidak ada tambahan Software Development. Kandidat internasional hanya digunakan untuk eksperimen training, bukan sebagai pengukuran kebutuhan pasar kerja Indonesia. Distribusi data yang diproses bukan distribusi nasional. Seluruh sumber, peran data dan ketentuan lisensi dijelaskan dalam [Data Sources](docs/DATA_SOURCES.md).

### Eksplorasi interaktif di Streamlit

Pada menu EDA, pilih profil lokal, kandidat augmentasi minoritas, atau gabungan kandidat. Filter sumber dan keluarga karier berlaku pada distribusi, frekuensi skill, dan pencarian profil. Tab perbandingan menampilkan distribusi lokal vs tambahan serta jumlah baris training berdasarkan metode balancing. CSV hasil filter dapat diunduh.

### Inventaris sumber dan kualitas data

| Sumber (tautan asli) | Isi / jumlah raw | Penggunaan |
|---|---|---|
| [Job Description and Salary in Indonesia](https://www.kaggle.com/datasets/canggih/jog-description-and-salary-in-indonesia) | 34.746 lowongan; 5.374 lolos filter awal bidang digital | Training lokal utama |
| [Knowledge and Skill Sets of Web Developer](https://data.mendeley.com/datasets/5shmcs4g7f/1) | 2.649 judul dan kebutuhan skill | Training lokal |
| [Data Science Jobs in Indonesia](https://zenodo.org/records/18162371) | 166 baris Glints utama; file Glints dan JobStreet tambahan | Training lokal dan kandidat augmentasi minoritas |
| [IT Job Posts Descriptions](https://www.kaggle.com/datasets/mscgeorges/itjobpostdescriptions) | 10.000 deskripsi IT | Eksperimen augmentasi internasional |
| [IT Job Description Role Classification](https://data.mendeley.com/datasets/btf562prk5/1) | 10.461 deskripsi; 2.102 kandidat minoritas sebelum filter | Eksperimen augmentasi selektif; Software Development baru dikeluarkan |
| [Data Science Job Postings & Skills](https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills) | 12.217 metadata pada file yang tersedia | Tidak untuk training: tidak ada deskripsi/skill |
| [Mapping the Digital Skills Gap](https://zenodo.org/records/21803444) | 15.247 lowongan dan 3.892 kursus | Audit pola template; kursus sebagai referensi, tidak untuk training |
| [IT Job Skill Requirements](https://data.mendeley.com/datasets/zzy8v7ggkz/3) | 776 kalimat | Audit vocabulary |

| Sumber lokal | Setelah filter awal | Label dikenali | Skill dikenali | Unik dalam sumber |
|---|---:|---:|---:|---:|
| Indonesia Kaggle | 5.374 | 2.763 | 2.600 | 1.762 |
| Web Developer | 2.649 | 2.222 | 1.903 | 1.163 |
| Glints Indonesia | 166 | 78 | 74 | 53 |

Deduplikasi lintas sumber menghasilkan 2.838 pasangan label–profil skill lokal. Nilai missing ditangani sebagai teks kosong; baris tanpa label atau keterampilan yang dikenali dikeluarkan. Deskripsi berpola sangat seragam tidak dipakai untuk training; hal ini merupakan kehati-hatian kualitas, bukan bukti bahwa data tersebut sintetis. Audit dan hash input tersedia pada `results/data_audit.csv`, `results/balance_data_audit.csv`, dan `results/input_manifest.csv`. Lisensi tambahan masih harus diverifikasi sebelum redistribusi; lihat [Data Sources](docs/DATA_SOURCES.md).

## 5. Metodologi

```text
Lowongan historis -> Filter dan audit -> Label keluarga dari judul
                  -> Ekstraksi skill -> Deduplikasi dan grouped split
                  -> TF-IDF -> Grouped CV -> Model pretrained
Input skill -> Hybrid ranking -> Top-3 keluarga karier
            -> Skill gap -> Prioritas belajar -> Simulasi cakupan
```

Judul pekerjaan hanya digunakan untuk membentuk label; input model adalah token keterampilan. TF-IDF di-fit pada setiap training fold. Empat-fold stratified grouped cross-validation memilih konfigurasi berdasarkan Macro F1. Hybrid menggabungkan 50% skor Naive Bayes dan 50% normalized weighted skill overlap. Ranking spesialisasi bersifat deskriptif, bukan classifier tersendiri.

Sepuluh skenario class imbalance dibandingkan, termasuk class weighting, random oversampling, random undersampling, selective augmentation dan kombinasi keduanya. Resampling diterapkan pada training saja. Rekomendasi skill menggunakan neighborhood berbobot Jaccard kuadrat yang dipilih berdasarkan development CV. Detail: [Methodology](docs/METHODOLOGY.md).

### Preprocessing dan feature engineering

1. Filter bidang digital berdasarkan fungsi pekerjaan pada sumber Kaggle Indonesia. Normalisasi nilai kosong dan teks untuk audit duplikasi.
2. Bentuk weak label lima keluarga karier dengan aturan regex judul. Judul, gaji, lokasi, pendidikan, dan asal sumber tidak dimasukkan sebagai fitur.
3. Ekstrak keterampilan bilingual dengan kamus alias dan batas token; misalnya `JS` → `JavaScript`, `postgres` → `PostgreSQL`. Aturan khusus membedakan React Native dari React dan SQL Server dari SQL. Terdapat 111 skill yang muncul pada development set.
4. Representasikan setiap skill sebagai token deterministik (`skill_` + UTF-8 hex) supaya nama multi-kata dan simbol seperti C++ tidak pecah; urutkan dan deduplikasi token per profil. TF-IDF memetakan token ke vektor sparse berbobot, dengan fitting hanya pada training fold.
5. Deduplikasi berdasarkan teks normalisasi dalam sumber dan pasangan label–vektor skill; kelompokkan vektor skill identik agar tidak bocor antar split. Development 2.271, test 567. Label konflik dan near-duplicate nonidentik masih mungkin ada.

### Pemilihan algoritma dan hyperparameter tuning

| Algoritma | Alasan | Ruang tuning | Konfigurasi terpilih |
|---|---|---|---|
| Majority baseline | Kontrol terhadap dominasi kelas | Prior kelas | Prior kelas |
| Weighted skill overlap | Baseline transparan tanpa classifier | Profil top-25 skill per keluarga | Normalisasi overlap |
| Multinomial Naive Bayes | Cocok untuk fitur sparse nonnegatif, efisien | alpha = 0,1; 0,5; 1 | alpha = 0,1 |
| Logistic Regression | Baseline linear diskriminatif | C = 1; 3; 10, class_weight balanced | C = 1 |
| Linear SVM | Pemisahan kelas pada ruang fitur berdimensi tinggi | C = 0,3; 1; 3, class_weight balanced | C = 0,3 |
| Random Forest | Pembanding nonlinear untuk interaksi skill | 60 trees; min_samples_leaf = 1; 3; balanced | 60 trees, leaf = 1 |
| Hybrid | Menggabungkan pola classifier dan kesesuaian skill | alpha = 0; 0,25; 0,5; 0,75; 1 | Naive Bayes + overlap, alpha = 0,5 |

Empat-fold StratifiedGroupKFold (shuffle, seed 43) menghasilkan prediksi out-of-fold; konfigurasi dipilih berdasarkan pooled Macro F1. Model final di-fit pada development set. Hybrid: `score = 0.5 × score_NB + 0.5 × normalized_overlap`. Skor SVM dinormalisasi dari decision function, bukan probabilitas terkalibrasi. Model disimpan sebagai `models/bundle.joblib` dan dimuat oleh kedua antarmuka. Hasil tuning lengkap: `results/tuning.csv`.

Moderate weighting menggunakan bobot akar inverse-frequency `sqrt(N / (K × N_c))`, dinormalisasi. Oversampling dan undersampling dilakukan hanya pada training fold. Model alternatif dan profil tambahan tidak mengganti profil demand lokal. Neighborhood skill ranking memanfaatkan kemiripan Jaccard kuadrat; lima kebijakan dibandingkan dengan grouped CV, dan neighborhood memperoleh validation Hit@5 46,31%.

## 6. Hasil evaluasi

### Definisi metrik

| Metrik | Interpretasi |
|---|---|
| Accuracy | Proporsi label keluarga yang benar pada ranking pertama. |
| Precision / Recall per kelas | Ketepatan prediksi suatu kelas / proporsi anggota kelas yang berhasil ditemukan. |
| Macro F1 | Rata-rata F1 tiap kelas tanpa pembobotan jumlah sampel; penting untuk kelas minoritas. |
| Balanced accuracy | Rata-rata recall tiap kelas, mengurangi pengaruh kelas mayoritas. |
| Top-3 Hit | Proporsi label aktual yang muncul dalam tiga rekomendasi dari lima keluarga. |
| MRR / NDCG@3 | Menilai posisi label aktual pada ranking; posisi lebih tinggi diberi nilai lebih baik. |
| Skill Hit@5 / Hit@10 | Proporsi skill yang disembunyikan dan muncul kembali dalam lima/sepuluh rekomendasi. |

### Perbandingan classifier

| Model                      |   Accuracy |   Macro F1 |   Top-3 Hit |
|:---------------------------|-----------:|-----------:|------------:|
| Naive Bayes                |     0.9259 |     0.8504 |      1.0000 |
| Logistic Regression        |     0.8836 |     0.8141 |      0.9965 |
| Linear SVM                 |     0.9206 |     0.8621 |      0.9982 |
| Random Forest              |     0.9153 |     0.8336 |      0.9965 |
| Majority baseline          |     0.7284 |     0.1686 |      0.8924 |
| International augmentation |     0.9083 |     0.8183 |      0.9982 |
| Skill overlap              |     0.8131 |     0.7331 |      0.9683 |
| Hybrid                     |     0.9330 |     0.8723 |      0.9965 |
| Balancing experiment       |     0.9259 |     0.8654 |      0.9965 |

Nilai tabel classifier berada pada skala 0–1. Berikut perbandingan metode balancing pada skala persen.


| Model | Accuracy | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|
| Hybrid (default) | 93,30% | 87,23% | 87,97% |
| Moderate class weighting | 92,59% | 86,54% | 90,42% |
| Selective augmentation | 92,42% | 86,31% | 90,37% |
| Class weighting | 88,54% | 82,41% | 91,47% |
| Random oversampling | 89,59% | 83,20% | 91,10% |
| Random undersampling | 87,65% | 80,63% | 89,32% |

Hasil pada 567 profil test lokal. Hybrid unggul pada development Macro F1 (81,33% dibanding 79,93% untuk moderate weighting), sehingga tetap menjadi default. Moderate weighting meningkatkan recall Infrastructure & Security dari 72,09% menjadi 83,72%, dengan trade-off precision dan accuracy. Majority baseline accuracy adalah 72,84%; karena itu Macro F1, balanced accuracy dan evaluasi per kelas menjadi penting.

Hybrid Top-3 Hit Rate 99,65%, MRR 0,9644 dan NDCG@3 0,9721. Top-3 mencakup tiga dari lima keluarga, bukan tiga lowongan individual. Audit kemiripan skill dengan development pada ambang Jaccard 0,8 menyisakan 512 profil test dengan accuracy 92,77%. Evaluasi ini bersifat retrospektif, bukan test independen baru.

Rekomendasi skill end-to-end (542 profil dengan satu skill disembunyikan): Hit@5 44,65%, Hit@10 64,39%, dibanding global-frequency baseline Hit@5 26,20%. Ini proxy pemulihan keterampilan, bukan ukuran keberhasilan belajar.

### Interpretasi per kelas

| career                    |   precision |   recall |   f1-score |   support |
|:--------------------------|------------:|---------:|-----------:|----------:|
| Data & AI                 |      0.8378 |   0.9394 |     0.8857 |   33.0000 |
| Design                    |      0.8000 |   0.8571 |     0.8276 |   28.0000 |
| Digital Marketing         |      0.9020 |   0.9200 |     0.9109 |   50.0000 |
| Infrastructure & Security |      0.8378 |   0.7209 |     0.7750 |   43.0000 |
| Software Development      |      0.9636 |   0.9613 |     0.9624 |  413.0000 |

Software Development mencakup 72,8% profil lokal, sehingga accuracy saja tidak cukup. Infrastructure & Security memiliki recall terendah (72,09%, 43 sampel test). Moderate weighting memperbaiki recall minoritas tetapi menurunkan Macro F1; karena itu Hybrid tetap menjadi default. Pada audit source-held-out, accuracy Kaggle Indonesia 88,37%, Web Developer 94,97%, dan Glints 71,43% (hanya tujuh sampel). Saat separuh skill dipertahankan, accuracy 90,48% dan Macro F1 82,17%. Group bootstrap 100 replikasi menghasilkan interval empiris 95% accuracy 91,07–95,38% dan Macro F1 82,71–90,83%. Seluruh audit menggunakan data historis, bukan validasi pengguna independen.

Hasil lengkap tersedia dalam `results/metrics.csv`, `results/class_report.csv`, `results/balance_metrics.csv`, `results/balance_class_report.csv` dan notebook.

### Cakupan interpretasi hasil

- Dataset historis, tidak mewakili seluruh pasar kerja Indonesia; label diturunkan dari judul dan belum divalidasi manusia secara independen.
- Ketimpangan kelas, kemiripan nonidentik dan temporal drift masih mungkin memengaruhi hasil. Test set digunakan kembali untuk audit retrospektif.
- Skor ranking dan coverage bukan probabilitas diterima bekerja, ukuran proficiency, atau alat penolakan kandidat. Tidak ada ground truth hasil hiring, belajar atau kecocokan pencari kerja nyata.
- Model joblib hanya boleh dimuat dari sumber tepercaya.
- Ketentuan atribusi dan redistribusi sumber data tersedia pada [Data Sources](docs/DATA_SOURCES.md).

## 7. Notebook, reproduksi dan deployment

```bash
python -m jupyter lab
```

Buka `notebooks/SkillBridge.ipynb`, pilih kernel virtual environment, lalu **Run All**. Notebook membaca data, model dan hasil eksperimen yang sama dengan Streamlit.

Untuk mereproduksi training (opsional, jalankan dari root):

```bash
python -m src.train
python -m src.balance
python -m unittest discover -s tests -v
```

Jalankan kedua perintah training secara berurutan untuk membangun kembali seluruh bundle, termasuk model balancing. Restart Streamlit setelah training ulang. Panduan server, deployment dan troubleshooting: [Deployment](docs/DEPLOYMENT.md). Checklist pengujian: [Validation](docs/VALIDATION.md).

## 8. Struktur proyek

```text
SkillBridge/
├── app.py                 # Streamlit entry point
├── requirements.txt       # Python dependencies
├── .streamlit/            # UI theme
├── data/raw/              # Source datasets for reproduction
├── data/processed/        # Model-ready profiles
├── models/                # Pretrained model bundle
├── notebooks/             # Research workflow
├── results/               # Experiment tables and audits
├── src/                   # Preprocessing, training and inference
├── tests/                 # Automated tests
```
