# Dataset provenance

## Primary source

- Dataset: Global Food Prices Database (WFP).
- Original data provider: World Food Programme (WFP).
- Distribution: Kaggle, uploader `abhishekgupta56447`.
- Source URL: https://www.kaggle.com/datasets/abhishekgupta56447/global-food-prices-database-wfp
- Exact download date, raw snapshot checksum, dataset version and license: not recorded in the supplied archive. Refer to the original listing and applicable terms before redistribution.

## Included data

The package includes the derived ASEAN model-ready panel (`data/foodshock_asean_model_ready.csv`), with 27,950 observations across seven countries and five commodity groups, and coverage summaries. The original raw CSV is not included. Dataset, model and result hashes are recorded in `data/v9_manifest.json`. No synthetic records are included in model training or benchmark testing. The regional gap is calculated as `pct_change_1m - asean_comm_mom_1m`.

## Reproduction

Use the packaged panel with `python src/run_v9.py --trials 25`. For raw reconstruction, obtain the corresponding Kaggle snapshot, check its schema, and prepare a CSV with `countryiso3,date,market,commodity,pricetype,priceflag,unit,price,usdprice`. Run `python src/prepare_data.py --raw /path/to/raw.csv --out-dir data_rebuilt`. Exact raw-to-panel reproduction has not been independently verified. A different source vintage may yield different rows; rerun the complete benchmark if replacing the panel.

## Additional references

- WFP food-price portal: https://data.humdata.org/ (background reference, not the declared download source).
- UN SDG 2, target 2.c: https://sdgs.un.org/goals/goal2
- WFP Alert for Price Spikes (ALPS) and ASEAN Food Security Information System (AFSIS): conceptual motivation only; FoodShock uses its own +5% threshold.
