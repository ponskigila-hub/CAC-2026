# Deployment guide

## Local runtime

Use Python 3.11–3.13 with the project requirements. Run commands from the directory containing `app.py`. The quick start in the root README covers Windows, Linux and macOS. Streamlit reads the bundled model, processed profiles and experiment CSV/JSON files. It does not train on startup. Do not remove files under `models/`, `results/`, `data/processed/` or `src/` for a demo.

## Server runtime

After verifying source redistribution permissions, deploy the project to a private server with Python 3.13, install `requirements.txt` in an isolated virtual environment and run:

```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
```

Use a reverse proxy with HTTPS and access controls where required. Configure the hosting platform to expose the assigned port. For managed Streamlit hosting, use `app.py` as the entry point and `requirements.txt` as the dependency file; select a compatible Python version. Public deployment is conditional on dataset permissions. No hosting deployment has been verified.

## Troubleshooting

| Symptom | Action |
|---|---|
| `ModuleNotFoundError` | Activate the correct environment and run `python -m pip install -r requirements.txt`. |
| PowerShell activation blocked | Use the virtual environment's `python.exe` directly as described in README. |
| Model load/version error | Use the specified scikit-learn version; recreate the environment if necessary. Only load trusted model artifacts. |
| Missing CSV or model | Extract the complete archive and preserve its directory structure. |
| Port 8501 occupied | Run `python -m streamlit run app.py --server.port=8502`. |
| Old results after retraining | Stop and restart Streamlit. |
| Empty prediction | Select recognized skills or use a quick-start profile. |

## Reproducibility

Run `python -m src.train` followed by `python -m src.balance`. This regenerates the primary and alternative models and experiment artifacts. Raw data is needed for reproduction but not for the normal inference workflow. Do not publish raw or derived data without checking the applicable licenses. Keep source attribution intact.
