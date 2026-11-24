# Evidence map — babak penyisihan

| Rubrik resmi | Bobot | Bukti | Keterbatasan |
|---|---:|---|---|
| Perumusan Permasalahan | 15% | README §§1–3, Overview, notebook, SDG 2.c | Target 5% definisi operasional; stakeholder belum diuji |
| Pengolahan dan Kualitas Data | 20% | Raw regional, manifest, EDA, exact calendar, sampling sensitivity, ablation | Cakupan tidak merata, lisensi dan publication lag belum diverifikasi |
| Implementasi Machine Learning | 25% | 5 konfigurasi regresi dan 5 classifier pada 4 origin validasi, shrinkage, model selection dan threshold per horizon, importance, frozen deployment, shared inference | Transfer threshold ke refit, sampling, tidak semua origin kalender diuji |
| Evaluasi Model | 20% | 4 validation + 4 retrospective origins; baseline, metrics per horizon/segment, coverage, error comparisons | Automatic kalah dari Persistence secara agregat; false alarm/recall masih terbatas; target kalender berulang; tidak ada independent post-2025 test |
| Inovasi dan Dampak Solusi | 20% | Target tanggal bebas, harga dan risiko, grafik, model comparison, notebook–Streamlit parity, export | Manfaat dunia nyata belum diukur; ketidakpastian horizon panjang |

Bobot mengikuti guidebook, bukan nilai resmi. Presentasi dan Q&A tidak termasuk penilaian penyisihan.
