# Evidence map — babak penyisihan

| Rubrik resmi | Bobot | Bukti dalam proyek | Keterbatasan |
|---|---:|---|---|
| Perumusan Permasalahan | 15% | README §1–3, Overview, notebook Overview; SDG 2.c dan target kenaikan 5% | Target 5% definisi operasional, bukan standar universal; kebutuhan stakeholder belum diuji |
| Pengolahan dan Kualitas Data | 20% | Raw regional, manifest checksum, EDA, shared preparation, feature definitions, missing and temporal handling | Cakupan tahun/pasar timpang, lisensi dan publication lag belum diverifikasi |
| Implementasi Machine Learning | 25% | Ridge vs HGB price regression; logistic vs HGB risk classification; training cutoff, feature/horizon, frozen deployment, code | Sampling untuk efisiensi; generalisasi origin non-kuartal belum diuji |
| Evaluasi Model | 20% | Validation, backtest 1–36, MAPE/RMSLE, AP/AUC, confusion matrix, FPR policy, interval coverage, baselines | Persistence unggul agregat; recall terbatas; test retrospektif dan horizon >36 belum divalidasi |
| Inovasi dan Dampak Solusi | 20% | Target tanggal bebas, harga dan risiko, model comparison, notebook–Streamlit parity, export | Manfaat dunia nyata belum diukur; uncertainty tinggi pada long horizon |

Bobot mengikuti guidebook, bukan nilai resmi. Tidak mengklaim seluruh poin sudah terpenuhi secara maksimal. Presentasi dan Q&A tidak termasuk penilaian penyisihan.
