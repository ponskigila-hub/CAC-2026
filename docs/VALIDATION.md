# Validation

- Seluruh 11 SHA-256 manifest sumber asli cocok dengan CSV pengguna.
- Training ulang selesai; hasil benchmark, prediksi test dan frozen deployment disertakan.
- Automated tests: 15/15 lulus (unique series-month, harga positif, fitur finite, target Jan 2028, tanggal dalam satu bulan, formula perubahan, threshold, far future, reject historical target, variasi horizon, kontrol FPR validation, target cutoff, future invariance, MAPE parity, interval ordering).
- Notebook dan Streamlit mengimpor fungsi `forecast_path` yang sama. Notebook memakai hasil frozen deployment dan benchmark CSV; tidak ada angka evaluasi hardcoded dalam kode grafik.
- Streamlit tidak tersedia pada runtime pemeriksaan; browser smoke test belum dilakukan. Tidak mengklaim UI sudah teruji secara end-to-end.
