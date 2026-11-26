# SkillBridge AI
**Machine learning for career-family recommendation and personalized skill-gap exploration in Indonesia's digital job market.**

## Latar belakang
Pencari kerja dan mahasiswa perlu menghubungkan keterampilan yang dimiliki dengan kebutuhan industri. Informasi lowongan tersebar dan judul pekerjaan sering tumpang tindih. SkillBridge memanfaatkan profil keterampilan yang disebutkan dalam lowongan historis untuk mengidentifikasi beberapa keluarga karier dan menyusun prioritas pengembangan keterampilan. Aplikasi bukan alat penyeleksi kandidat dan tidak memprediksi diterima bekerja, gaji, atau lowongan aktif.

## Rumusan masalah dan tujuan
1. Apakah profil skill dapat mengidentifikasi keluarga pekerjaan digital berdasarkan label yang diturunkan dari judul lowongan?
2. Apakah hybrid ML dan weighted skill overlap mengungguli baseline?
3. Apakah pemanfaatan kemunculan skill bersama meningkatkan pemulihan keterampilan yang disembunyikan dari profil test?
4. Bagaimana rekomendasi dan simulasi cakupan skill dapat dijelaskan secara transparan?

Kontribusi: rekomendasi tiga keluarga karier, eksplorasi spesialisasi deskriptif, ranking skill berbasis frekuensi dan co-occurrence, penjelasan sensitivitas skor individual, serta simulasi penambahan skill. Mendukung SDG 4.4 (keterampilan relevan dengan pekerjaan) dan SDG 8.5 (pekerjaan produktif dan layak); dampak terhadap indikator SDG belum diukur. Referensi: [SDG 4](https://sdgs.un.org/goals/goal4), [SDG 8](https://sdgs.un.org/goals/goal8), [Kemnaker](https://www.kemnaker.go.id/).

## Data dan ruang lingkup
Tiga sumber lokal: lowongan Kaggle Indonesia, Web Developer dan Glints. Sumber IT internasional diuji sebagai augmentation, tidak menjadi model default. Sumber metadata LinkedIn tidak memiliki deskripsi atau skill; corpus Digital Skills Gap memiliki pola teks berulang sehingga tidak digunakan untuk training; corpus kursus hanya inventaris referensi; IT sentences untuk audit vocabulary. Lihat `docs/DATA_SOURCES.md` dan `results/data_audit.csv`. Dataset tambahan Kaggle dan Glints memerlukan verifikasi lisensi redistribusi sebelum dikumpulkan/dipublikasikan. Arsip ini memuat salinan file yang diberikan pengguna untuk reproduksi lokal; bukan pernyataan izin redistribusi.

2.838 profil lokal unik, 2.271 development, 567 test. Lima keluarga: Data & AI, Design, Digital Marketing, Infrastructure & Security, Software Development. Label diturunkan dari judul menggunakan aturan eksplisit, bukan label hasil rekrutmen yang diverifikasi. Kategori yang tumpang tindih pada V1 digabung menjadi keluarga yang lebih konsisten. Cakupan 111 skill yang ditemukan pada development. Spesialisasi hanya ditampilkan jika memiliki minimal 10 profil development; bukan classifier spesialisasi yang tervalidasi.

## Metode
- Filter bidang digital pada sumber Kaggle Indonesia; ekstraksi skill bilingual menggunakan regex; hapus duplikasi persis per keluarga dan profil skill. Kelompokkan berdasarkan profil skill identik, termasuk profil berlabel berbeda.
- Holdout berkelompok, kemudian 4-fold stratified grouped CV pada development. TF-IDF dan estimator di-fit dalam setiap fold. Judul lowongan digunakan untuk label saja; input classifier hanya token skill. Baseline majority dan weighted skill overlap.
- Uji Naive Bayes, Logistic Regression, Linear SVM, Random Forest, hybrid, dan augmentation internasional. Konfigurasi dipilih berdasarkan pooled Macro F1 dari prediksi out-of-fold development. Holdout tidak dipakai untuk memilih model atau alpha, tetapi hasilnya retrospektif karena sumber lama telah dieksplorasi.
- Hybrid terpilih: 0,5 skor Naive Bayes + 0,5 normalized weighted skill overlap. Bobot skill berasal dari frekuensi profil development lokal. Skor ranking bukan probabilitas diterima kerja. Linear SVM menggunakan softmax margin untuk ranking (bukan probabilitas terkalibrasi).
- Prioritas belajar: uji campuran frekuensi dan conditional co-occurrence skill pada development CV, pilih co-occurrence penuh; gunakan frekuensi jika bukti pasangan tidak tersedia. Kandidat dibatasi 25 skill teratas per keluarga. Evaluasi end-to-end pada test setelah satu skill disembunyikan.
- Explainability: perubahan skor ketika satu skill dihapus, bukan sebab-akibat. Simulator: perubahan cakupan bobot skill (aritmetis), bukan prediksi keberhasilan belajar.

## Hasil utama
| Model | Accuracy test | Macro F1 test | Top-3 hit test |
|---|---:|---:|---:|
| Majority | 72,84% | 16,86% | 89,24% |
| Skill overlap | 81,31% | 73,31% | 96,83% |
| Naive Bayes | 92,59% | 85,04% | 100,00% |
| Logistic Regression | 88,36% | 81,41% | 99,65% |
| Linear SVM | 92,06% | 86,21% | 99,82% |
| Random Forest | 91,53% | 83,36% | 99,65% |
| International augmentation | 90,83% | 81,83% | 99,82% |
| Hybrid (default) | 93,30% | 87,23% | 99,65% |

Default CV accuracy 90,62%, Macro F1 81,33%. Test MRR 0,9644, NDCG@3 0,9721. Group bootstrap (100 resamples), percentile interval for test accuracy 91,07–95,38% and Macro F1 82,71–90,83%; descriptive uncertainty, not future guarantee. Test has 413 Software Development profiles out of 567; majority baseline accuracy is already high. Top-3 over five broad families is an easier task than Top-3 over eight fine categories. Do not compare V1 and V2 accuracy as a direct like-for-like improvement.

Masked skill recovery on 542 test profiles: global 26,20%; predicted-family frequency 35,79%; predicted-family co-occurrence 42,44%; oracle field co-occurrence 44,65%. Oracle is diagnostic, not deployment. With roughly half the skills retained (rounded up), test accuracy 90,48%, Macro F1 82,17%. Source-held-out audit uses fixed hybrid configuration and removes each source from development: Indonesia Kaggle accuracy 88,37% (361 test profiles); Web Developer 94,97% (199); Glints 71,43% (7, one class, not representative).

## Menjalankan (Windows PowerShell)
Gunakan Python 3.11–3.13. Ekstrak ZIP dan buka folder ini di terminal.
```powershell
python -m venv cac2026-venv
.\cac2026-venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Jika execution policy memblokir aktivasi, gunakan `cac2026-venv\Scripts\python.exe -m pip install -r requirements.txt` dan `cac2026-venv\Scripts\python.exe -m streamlit run app.py`.

Notebook: `python -m jupyter lab`, buka `notebooks/SkillBridge.ipynb`. Run All untuk audit, visualisasi, evaluasi dan inferensi dari model yang disertakan. Training penuh opsional: `python -m src.train` (mereproduksi preprocessing, CV, final fit, gap tuning dan source audit; jalankan dari root). Restart Streamlit setelah training ulang karena model dicache. Tes: `python -m unittest discover -s tests -v`.

## Navigasi
Overview → EDA → Preprocessing → Training & Testing → Prediction. Model sudah dilatih; Training & Testing memilih dan membandingkan model tersimpan, tidak menjalankan training ulang di browser. Prediction: pilih skill atau contoh profil, lihat tiga keluarga, pilih bidang eksplorasi, prioritas skill, simulator, penjelasan individual dan spesialisasi yang didukung. Export CSV ranking tersedia.

## Keterbatasan dan kepatuhan
- Cakupan tidak mewakili semua lowongan Indonesia; data lintas periode dan sumber, bukan survei representatif atau data lowongan aktif. Profil dideduplikasi sehingga demand share bukan proporsi semua lowongan.
- Label lemah dari judul; 57 grup skill identik memiliki label keluarga berbeda. Near-duplicate nonidentik, overlap sumber, dan temporal drift masih mungkin; belum ada audit label manusia atau holdout temporal baru.
- Tidak ada ground truth kecocokan kandidat, keberhasilan belajar atau hiring. Evaluasi masked recovery adalah proxy. Skill gap masih terbatas (Hit@5 44,65%).
- Model dan notebook diuji secara lokal; Streamlit tidak tersedia di lingkungan pengujian sehingga browser smoke test belum dilakukan. Pastikan UI berjalan di komputer tim.
- Konfirmasi lisensi setiap sumber tambahan dan ketentuan penggunaan AI dengan panitia. Tim perlu memahami dan memverifikasi kode, data, dan interpretasi sebelum submission. Jangan menggunakan rekomendasi untuk menolak kandidat.

## Struktur
`data/raw` sumber yang diberikan; `data/processed/jobs.csv` profil model; `src` pipeline bersama; `models` bundle pretrained; `results` hasil eksperimen; `notebooks` notebook; `tests` tes otomatis; `docs` metodologi, atribusi, validasi. Tidak ada virtual environment di dalam paket.

## Audit sensitivitas dan rekomendasi keterampilan
Klasifikasi tetap menggunakan model Hybrid dan split yang sama (accuracy 93,30%, Macro F1 87,23%). Audit Jaccard atas himpunan skill: tanpa profil test dengan kemiripan >=0,8 terhadap development, tersisa 512 profil (accuracy 92,77%, Macro F1 86,36%); ambang 0,7 menyisakan 416 (91,83%, 86,14%). Ini audit sensitivitas input, bukan deduplikasi teks asli atau test independen. Ada 57 grup ambigu (127 profil). Template review 90 profil pada results/label_review_template.csv masih kosong, tidak diklaim sebagai validasi manusia.

Rekomendasi belajar membandingkan lima kebijakan pada 4-fold grouped development CV, dengan keluarga karier diprediksi dari skill tersisa. Neighborhood berbobot Jaccard kuadrat terpilih (CV Hit@5 46,31%). Pada 542 profil test yang sama: Hit@5 44,65%, Hit@10 64,39%, MRR 0,3199. Dibanding metode co-occurrence sebelumnya 42,44%, peningkatan Hit@5 adalah 2,21 poin persentase. Kandidat mencakup seluruh skill dengan dukungan pada keluarga development; simulator tetap memakai profil coverage top-25 agar definisi coverage konsisten. Hasil retrospektif; belum ada test baru atau bukti dampak pengguna.

## Class imbalance experiments

The original Hybrid remains the default. Ten scenarios use the original 2,271-profile development split and 567-profile local test set. The extra role dataset contributes only non-software-development roles; additional Indonesian JobStreet and Glints files are filtered similarly. Exact skill profiles overlapping any original profile and conflicting augmentation profiles are excluded. 1,315 novel candidate profiles remain (620 Infrastructure & Security, 596 Data & AI, 99 Design). The duplicate first Glints upload is not added again. Original local demand and learning profiles are unchanged.

Moderate class weighting (square root of inverse class frequency) is the highest-development-Macro-F1 alternative to the original Hybrid. It improves balanced accuracy on the reused test from 87.97% to 90.42%, while accuracy falls from 93.30% to 92.59% and Macro F1 from 87.23% to 86.54%. The original Hybrid wins development Macro F1 (81.33% vs 79.93%); therefore the alternative is optional, not a replacement. Select `Balancing experiment` in Training & Testing or Prediction. Skill recovery metrics continue to describe the original default only.

Run `python -m src.balance` after `python -m src.train` to regenerate the experiment and optional pretrained model. See the Class imbalance tab and `results/balance_*.csv`. Results are retrospective, not independent validation. Exact profile exclusion does not guarantee removal of all near-duplicate descriptions. The extra international source is a candidate training source only, not evidence of local market demand. Its accompanying README identifies CC BY-SA 4.0; retain attribution and applicable share-alike terms. Other raw-source redistribution permissions still require verification before submission. Source: https://data.mendeley.com/datasets/btf562prk5/1 and https://zenodo.org/records/18162371.
