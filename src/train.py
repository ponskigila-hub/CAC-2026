import json
from collections import Counter
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.base import clone
from .core import ROOT, skills, features, label, score, overlap, recommend, role
R = ROOT / 'data/raw'
O = ROOT / 'results'

def prepare():
    a = pd.read_csv(R / 'all.csv.zip', sep='|')
    w = pd.read_excel(R / 'web/Dataset_Web Developer.xlsx')
    g = pd.read_csv(R / 'glints_job_data_with_company.csv')
    j = pd.read_csv(R / 'JobsDataset.csv.zip')
    l = pd.read_csv(R / 'job_postings.csv.zip')
    digital = pd.read_csv(R / 'digital/job_postings.csv')
    it = pd.read_csv(R / 'it/job_sentences.csv')
    courses = pd.read_csv(R / 'digital/course_descriptions.csv')
    local = a[a.job_function.str.contains('Komputer|Digital Marketing|Seni / Desain Kreatif', case=False, na=False)]
    frames = []
    audit = []
    for name, d, t, x in [('Indonesia Kaggle', local, 'job_title', 'job_description'),
        ('Web Developer', w, 'Job_Title', 'Required Skills'),
        ('Glints Indonesia', g, 'title', 'skills'),
        ('International IT', j, 'Job Title', 'Description')]:
        z = pd.DataFrame({'title': d[t].fillna(''), 'text': d[x].fillna('')})
        z['career'] = z.title.map(label)
        z = z[z.career.notna()].copy()
        mapped = len(z)
        z['skills'] = z.text.map(skills)
        z['x'] = z.skills.map(features)
        z = z[z.x.ne('')].copy()
        recognized = len(z)
        z['norm'] = z.text.str.lower().str.replace('\\W+', ' ', regex=True).str.strip()
        z['source'] = name
        z = z.drop_duplicates(['career', 'norm']).drop_duplicates(['career', 'x'])
        frames.append(z)
        audit.append({'source': name,
            'raw': len(d),
            'mapped': mapped,
            'recognized': recognized,
            'unique': len(z),
            'usage': 'local' if name != 'International IT' else 'augmentation candidate'})
        print(audit[-1], flush=True)
    d = pd.concat(frames[:3], ignore_index=True).drop_duplicates(['career', 'x']).reset_index(drop=True)
    d['group'] = d.x
    d['id'] = np.arange(len(d))
    ext = frames[3]
    ext = ext[~ext.x.isin(d.x)].copy()
    audit.extend([{'source': 'LinkedIn metadata', 'raw': len(l), 'usage': 'excluded: no skills or descriptions'},
        {'source': 'Digital gap jobs', 'raw': len(digital), 'usage': 'excluded: template concerns'},
        {'source': 'Digital gap courses', 'raw': len(courses), 'usage': 'reference only'},
        {'source': 'IT sentences', 'raw': len(it), 'usage': 'vocabulary audit only'}])
    pd.DataFrame(audit).to_csv(O / 'data_audit.csv', index=False)
    pd.DataFrame(Counter((s for t in it.text for s in skills(t))).most_common(),
        columns=['skill', 'count']).to_csv(O / 'it_vocabulary.csv',
        index=False)
    return (d, ext)

def pipe(e):
    return make_pipeline(TfidfVectorizer(lowercase=False), clone(e))

def profiles(d):
    return {c: {s: n / len(g) for s, n in Counter((s for ss in g.skills for s in ss)).most_common(25)} for c,
        g in d.groupby('career')}

def measure(y, p, classes):
    order = np.argsort(-p, axis=1)
    rank = np.array([list(classes[o]).index(v) + 1 for o, v in zip(order, y)])
    return {'accuracy': accuracy_score(y, classes[order[:, 0]]),
        'macro_f1': f1_score(y, classes[order[:, 0]], average='macro', zero_division=0),
        'top3_hit': (rank <= 3).mean(),
        'mrr': np.mean(1 / rank),
        'ndcg_at3': np.mean(np.where(rank <= 3, 1 / np.log2(rank + 1), 0))}

def ovmat(prof, ss, cls):
    a = np.array([overlap(prof, s, cls) for s in ss])
    return np.divide(a, a.sum(1, keepdims=True), out=np.full_like(a, 1 / len(cls)), where=a.sum(1, keepdims=True) > 0)

def split_local_data():
    """Build the grouped local split and development folds."""
    d, ext = prepare()
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(d, d.career, d.group))
    dev = d.iloc[tr]
    test = d.iloc[te]
    d['split'] = 'test'
    d.loc[tr, 'split'] = 'development'
    assert not set(dev.group) & set(test.group)
    cls = np.array(sorted(d.career.unique()))
    print('split', len(dev), len(test), d.career.value_counts().to_dict(), flush=True)
    folds = list(StratifiedGroupKFold(4, shuffle=True, random_state=43).split(dev, dev.career, dev.group))
    return d, ext, dev, test, cls, folds


def tune_classifiers(dev, cls, folds):
    """Choose classifier settings and hybrid weight using development folds."""
    candidates = {'Naive Bayes': [MultinomialNB(alpha=v) for v in [0.1, 0.5, 1]],
        'Logistic Regression': [LogisticRegression(C=v, max_iter=500, class_weight='balanced') for v in [1, 3, 10]],
        'Linear SVM': [LinearSVC(C=v, class_weight='balanced') for v in [0.3, 1, 3]],
        'Random Forest': [RandomForestClassifier(n_estimators=60, min_samples_leaf=v, random_state=42, n_jobs=2, class_weight='balanced') for v in [1, 3]],
        'Majority baseline': [DummyClassifier(strategy='prior')]}
    tuning = []
    chosen = {}
    oofs = {}
    for name, opts in candidates.items():
        best = -1
        for est in opts:
            oof = np.zeros((len(dev), len(cls)))
            fold_metrics = []
            for a, b in folds:
                fit, val = (dev.iloc[a], dev.iloc[b])
                m = pipe(est).fit(fit.x, fit.career)
                oof[b] = score(m, val.x)
                fold_metrics.append(measure(val.career, oof[b], cls)['macro_f1'])
            met = measure(dev.career, oof, cls)
            tuning.append({'model': name, 'parameters': str(est), **met, 'fold_macro_f1_std': np.std(fold_metrics)})
            if met['macro_f1'] > best:
                best = met['macro_f1']
                chosen[name] = est
                oofs[name] = oof
        print('tuned', name, best, flush=True)
    base = max((k for k in chosen if k != 'Majority baseline'),
        key=lambda k: measure(dev.career, oofs[k], cls)['macro_f1'])
    ov = np.zeros_like(oofs[base])
    for a, b in folds:
        ov[b] = ovmat(profiles(dev.iloc[a]), dev.iloc[b].skills, cls)
    oofs['Skill overlap'] = ov
    alpha = max([0, 0.25, 0.5, 0.75, 1],
        key=lambda a: measure(dev.career, a * oofs[base] + (1 - a) * ov, cls)['macro_f1'])
    oofs['Hybrid'] = alpha * oofs[base] + (1 - alpha) * ov
    for a in [0, 0.25, 0.5, 0.75, 1]:
        tuning.append({'model': 'Hybrid',
            'parameters': f'alpha={a}',
            **measure(dev.career, a * oofs[base] + (1 - a) * ov, cls)})
    return chosen, oofs, base, alpha, ov, tuning


def fit_final_models(dev, test, ext, cls, folds, chosen, oofs, base, alpha, ov):
    """Fit the selected configurations on development profiles only."""
    external_oof = np.zeros_like(ov)
    for a, b in folds:
        fit = dev.iloc[a]
        aug = pd.concat([fit, ext])
        m = pipe(chosen[base]).fit(aug.x, aug.career)
        external_oof[b] = score(m, dev.iloc[b].x)
    oofs['International augmentation'] = external_oof
    best = max((k for k in oofs if k not in ['Majority baseline', 'Skill overlap']),
        key=lambda k: measure(dev.career, oofs[k], cls)['macro_f1'])
    models = {k: pipe(e).fit(dev.x, dev.career) for k, e in chosen.items()}
    aug = pd.concat([dev, ext])
    models['International augmentation'] = pipe(chosen[base]).fit(aug.x, aug.career)
    prof = profiles(dev)
    pr = {k: score(m, test.x) for k, m in models.items()}
    pr['Skill overlap'] = ovmat(prof, test.skills, cls)
    pr['Hybrid'] = alpha * pr[base] + (1 - alpha) * pr['Skill overlap']
    return best, models, prof, pr


def build_bundle(dev, test, cls, oofs, pr, best, models, prof, base, alpha):
    """Persist classifier predictions, confusion matrices and inference bundle."""
    rows = []
    pred = test[['id', 'career', 'source']].copy()
    report = []
    for k, p in pr.items():
        rows.append({'model': k, 'split': 'grouped CV', **measure(dev.career, oofs[k], cls)})
        rows.append({'model': k, 'split': 'test', **measure(test.career, p, cls)})
        y = cls[p.argmax(1)]
        pred[k] = y
        cr = classification_report(test.career, y, labels=cls, output_dict=True, zero_division=0)
        report.extend(({'model': k, 'career': c, **cr[c]} for c in cls))
        pd.DataFrame(confusion_matrix(test.career, y, labels=cls), index=cls, columns=cls).to_csv(O / f'cm_{k}.csv')
    sd = dev.copy()
    sd['specialization'] = sd.title.map(role)
    sd = sd[sd.specialization.notna()]
    specs = {c: profiles(g.rename(columns={'career': 'field', 'specialization': 'career'})) for c,
        g in sd.groupby('career')}
    bundle = {'best': best,
        'models': models,
        'base': base,
        'alpha': alpha,
        'profiles': prof,
        'specializations': specs,
        'vocabulary': sorted(set((s for ss in dev.skills for s in ss)))}
    joblib.dump(bundle, ROOT / 'models/bundle.joblib', compress=3)
    return rows, pred, report, bundle


def evaluate_additional(test, dev, cls, bundle, best, base, alpha, models, prof, pr):
    """Evaluate historical skill recovery, masking robustness and bootstrap intervals."""
    gc = Counter((s for ss in dev.skills for s in ss))
    gaprows = []
    for r in test.itertuples():
        if len(r.skills) < 2:
            continue
        hidden = str(np.random.default_rng(42 + r.id).choice(r.skills))
        known = set(r.skills) - {hidden}
        rec = recommend(bundle, list(known), best)
        if rec.empty:
            continue
        c = rec.iloc[0]['Career']
        globalrank = [s for s, n in gc.most_common() if s not in known][:5]
        oracle = [s for s in prof[r.career] if s not in known][:5]
        actual = [s for s in prof[c] if s not in known][:5]
        gaprows.append({'id': r.id,
            'hidden': hidden,
            'actual_field': r.career,
            'predicted_field': c,
            'global_hit': int(hidden in globalrank),
            'oracle_hit': int(hidden in oracle),
            'end_to_end_hit': int(hidden in actual)})
    pd.DataFrame(gaprows).to_csv(O / 'gap_evaluation.csv', index=False)
    robust = []
    for fraction in [0.5, 0.75, 1]:
        ss = [list(np.random.default_rng(42 + i).choice(s, max(1, int(np.ceil(len(s) * fraction))), replace=False)) for i,
            s in enumerate(test.skills)]
        x = [features(s) for s in ss]
        p = score(models[base if best == 'Hybrid' else best], x)
        p = alpha * p + (1 - alpha) * ovmat(prof, ss, cls) if best == 'Hybrid' else p
        robust.append({'retained_fraction': fraction, **measure(test.career, p, cls)})
    pd.DataFrame(robust).to_csv(O / 'robustness.csv', index=False)
    rng = np.random.default_rng(42)
    groups = [np.array(v) for v in test.reset_index(drop=True).groupby('group').indices.values()]
    boot = []
    yy = test.career.to_numpy()
    p = pr[best]
    for _ in range(100):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boot.append(measure(yy[idx], p[idx], cls))
    pd.DataFrame(boot).quantile([0.025, 0.975]).to_csv(O / 'bootstrap_ci.csv')


def train():
    d, ext, dev, test, cls, folds = split_local_data()
    chosen, oofs, base, alpha, ov, tuning = tune_classifiers(dev, cls, folds)
    best, models, prof, pr = fit_final_models(dev, test, ext, cls, folds, chosen, oofs, base, alpha, ov)
    rows, pred, report, bundle = build_bundle(dev, test, cls, oofs, pr, best, models, prof, base, alpha)
    evaluate_additional(test, dev, cls, bundle, best, base, alpha, models, prof, pr)
    pd.DataFrame(rows).to_csv(O / 'metrics.csv', index=False)
    pd.DataFrame(tuning).to_csv(O / 'tuning.csv', index=False)
    pd.DataFrame(report).to_csv(O / 'class_report.csv', index=False)
    pred.to_csv(O / 'test_predictions.csv', index=False)
    d.assign(skills=d.skills.map('|'.join)).drop(columns=['text', 'norm']).to_csv(ROOT / 'data/processed/jobs.csv',
        index=False)
    (O / 'summary.json').write_text(json.dumps({
        'best': best, 'base': base, 'alpha': alpha,
        'development': len(dev), 'test': len(test), 'external': len(ext),
        'classes': cls.tolist(), 'vocabulary': len(bundle['vocabulary']),
    }, indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
if __name__ == '__main__':
    train()
    from .enhance import run
    run()
    from .source_audit import run as audit_sources
    audit_sources()
    from .quality import run as quality_audit
    quality_audit()
    from .gap_rank import run as gap_v3
    gap_v3()
