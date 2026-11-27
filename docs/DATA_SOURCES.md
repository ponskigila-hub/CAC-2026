# Sumber dan penggunaan data

| File lokal | Sumber yang dirujuk | Status penggunaan |
|---|---|---|
| all.csv.zip | https://www.kaggle.com/datasets/canggih/jog-description-and-salary-in-indonesia | Utama; filter digital dari 34.746 baris |
| JobsDataset.csv.zip | https://www.kaggle.com/datasets/mscgeorges/itjobpostdescriptions | Eksperimen augmentation internasional |
| job_postings.csv.zip | https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills | Metadata saja; dikeluarkan dari ML |
| glints_job_data_with_company.csv | https://zenodo.org/records/18162371 | Sumber lokal pendukung |
| web/Dataset_Web Developer.xlsx | https://data.mendeley.com/datasets/5shmcs4g7f/1 | Sumber lokal |
| it/* | https://data.mendeley.com/datasets/zzy8v7ggkz/3 | Audit vocabulary |
| digital/* | https://zenodo.org/records/21803444 | Audit template dan inventaris kursus; tidak digunakan dalam ML |

Sumber dipasangkan dengan file berdasarkan nama dan struktur yang diberikan pengguna. Lisensi setiap sumber tambahan belum berhasil diverifikasi secara independen; halaman yang dapat diakses tidak menyediakan metadata lisensi spesifik yang cukup. Sebelum redistribusi, periksa halaman dataset, persyaratan situs asli dan izin yang berlaku. Salinan raw dalam paket ini untuk reproduksi lokal pengguna. Tidak mengklaim autentisitas setiap iklan secara independen. Hash SHA-256 dalam manifest mengidentifikasi salinan input, bukan bukti autentisitas atau izin.

## Selective augmentation

| File / folder | Source | Purpose |
|---|---|---|
| `balancing/IT Job Description Role Classification Dataset wit.zip` | https://data.mendeley.com/datasets/btf562prk5/1 | International minority-role candidates; source README identifies CC BY-SA 4.0. Original README and attribution retained in source ZIP. |
| `balancing/glints_job_data_with_company2.csv`, `glints_job_data_with_company3.csv`, `job_list_with_description.csv`, `job_list_with_description2.csv` | https://zenodo.org/records/18162371 | Local minority-role candidates; the original first Glints file is not added twice. |

Software Development candidates are excluded from augmentation. Exact profiles matching any original local profile and conflicting augmentation profiles are excluded. See `results/balance_data_audit.csv`. Additional source license terms require verification before redistribution or public hosting.
