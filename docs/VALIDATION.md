# Validation

- Training from regional raw CSV was executed, with four validation and four retrospective origins, then refit through December 2025.
- 21/21 automated tests passed: calendar alignment, feature invariance, model and horizon policy selection, threshold constraints on validation, reported metric consistency, deployment inference, future target, and interval ordering.
- Notebook executed successfully (5/5 code cells).
- Notebook and Streamlit share the `forecast_path` function, model bundle and CSVs. No hardcoded benchmark values are used in Streamlit metrics.
- Browser end-to-end testing of this version has not been performed because Streamlit is not installed in the execution environment. Run locally before submission.
