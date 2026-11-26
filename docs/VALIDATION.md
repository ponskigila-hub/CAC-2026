# Validasi

- Periksa hasil `results/metrics.csv`, `results/class_report.csv`, `results/gap_evaluation.csv`, `results/source_holdout.csv`, `results/robustness.csv`, `results/bootstrap_ci.csv`.
- Jalankan `python -m unittest discover -s tests -v`.
- Notebook dieksekusi ulang dengan nbconvert; hasil disimpan dalam notebook.
- Uji manual Streamlit: buka lima halaman, pilih semua model, coba profil kosong, pilih preset Data/Web/Infrastructure/Design/Marketing, coba skill tambahan, simulator dan download CSV. Pastikan hasil yang sama dengan notebook untuk input/model yang sama.
- Streamlit tidak tersedia di lingkungan pengujian; browser smoke test belum dilakukan.
- Sebelum submit, verifikasi lisensi data tambahan dan ketentuan penggunaan AI/orisinalitas; tim harus memahami seluruh metode.

- 19 automated tests passed; notebook executed successfully after neighborhood ranking integration. Full classifier retraining not repeated in this edition: existing V2 classifiers retained. New gap policy fitted and audited.
