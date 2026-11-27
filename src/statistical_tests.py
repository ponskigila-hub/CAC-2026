"""Paired group-bootstrap on the historical local holdout (exploratory)."""
import numpy as np
import pandas as pd
from .core import ROOT


def metrics(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    true = cm.sum(axis=-1)
    pred = cm.sum(axis=-2)
    f1 = np.divide(2 * tp, true + pred, out=np.zeros_like(tp, dtype=float), where=(true + pred) != 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp, dtype=float), where=true != 0)
    return {'accuracy': tp.sum(axis=-1) / cm.sum(axis=(-2, -1)),
            'macro_f1': f1.mean(axis=-1), 'balanced_accuracy': recall.mean(axis=-1)}


def paired(predictions, groups, reference, n_boot=1000, seed=42):
    """Use identical group draws for all models; CIs are unadjusted percentile intervals."""
    classes = sorted(predictions.career.unique())
    k = len(classes)
    codes = {c: i for i, c in enumerate(classes)}
    _, g = np.unique(np.asarray(groups), return_inverse=True)
    ng = g.max() + 1
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(ng, np.full(ng, 1 / ng), size=n_boot)
    y = predictions.career.map(codes).to_numpy()
    values = {}
    for name in predictions.columns.difference(['id', 'career', 'source'], sort=False):
        p = predictions[name].map(codes).to_numpy()
        cm = np.zeros((ng, k, k), dtype=np.int64)
        np.add.at(cm, (g, y, p), 1)
        values[name] = (metrics(cm.sum(0)), metrics((weights @ cm.reshape(ng, -1)).reshape(n_boot, k, k)))
    rows = []
    for name, (point, draws) in values.items():
        if name == reference:
            continue
        for metric, value in point.items():
            delta = values[reference][1][metric] - draws[metric]
            lo, hi = np.quantile(delta, [.025, .975])
            rows.append({'reference': reference, 'comparison': name, 'metric': metric,
                         'difference_pp': 100 * (values[reference][0][metric] - value),
                         'ci_low_pp': 100 * lo, 'ci_high_pp': 100 * hi,
                         'excludes_zero': bool(lo > 0 or hi < 0), 'replicates': n_boot})
    return pd.DataFrame(rows)


def run():
    out = ROOT / 'results'
    jobs = pd.read_csv(ROOT / 'data/processed/jobs.csv').set_index('id')
    for filename, ref, target in [('test_predictions', 'Hybrid', 'paired_model_ci'),
                                  ('balance_test_predictions', 'Original Hybrid', 'paired_balance_ci')]:
        p = pd.read_csv(out / f'{filename}.csv')
        assert p.id.is_unique and jobs.loc[p.id, 'split'].eq('test').all()
        paired(p, jobs.loc[p.id, 'group'], ref).to_csv(out / f'{target}.csv', index=False)
    group_counts = jobs.groupby('group').career.nunique()
    audit = pd.DataFrame([{'profiles': len(jobs), 'ambiguous_groups': int((group_counts > 1).sum()),
                           'ambiguous_profiles': int(jobs.group.map(group_counts).gt(1).sum()),
                           'at_most_two_skills': int(jobs.skills.str.split('|').str.len().le(2).sum())}])
    audit.to_csv(out / 'profile_quality_summary.csv', index=False)


if __name__ == '__main__':
    run()
