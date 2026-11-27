# Validation and acceptance checklist

## Automated checks

Run `python -m unittest discover -s tests -v` from the project root. The tests cover inference, profile grouping, skill ranking and balancing artifacts. Run the notebook using its project environment to reproduce displayed results.

## Manual acceptance tests

- Open all five pages through the sidebar and Previous/Next buttons.
- Check model selection, metrics, per-class results and the Class imbalance tab.
- Try Data, Web, Infrastructure, Design and Marketing presets and empty input.
- Check specialization exploration, skill gap, simulator, explanation and CSV export.
- Compare the same input and algorithm in notebook and Streamlit.
- Verify source licenses, attribution and competition AI/originality requirements before submission.

Browser-level Streamlit testing and independent user testing are not yet verified. Human label review template in `results/label_review_template.csv` is unfilled. Reused test metrics are retrospective. The notebook and app use the same saved artifacts.
