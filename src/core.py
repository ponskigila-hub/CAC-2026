import json
from pathlib import Path
import re, joblib, pandas as pd, numpy as np
ROOT = Path(__file__).resolve().parents[1]
ALIASES = json.loads((ROOT / 'config/skill_aliases.json').read_text(encoding='utf-8'))
PATTERNS = {k: re.compile('(?<![\\w])(?:' + '|'.join((re.escape(v) for v in sorted(vals, key=len, reverse=True))) + ')(?![\\w])', re.I) for k,
    vals in ALIASES.items()}

def extract(text):
    text = str(text)
    found = {k for k, p in PATTERNS.items() if p.search(text)}
    if 'React Native' in found and (not re.search('\\breact\\b(?!\\s+native)', text, re.I)):
        found.discard('React')
    if 'SQL Server' in found and (not re.search('\\bsql\\b(?!\\s+server)', text, re.I)):
        found.discard('SQL')
    return sorted(found)

def features(skills):
    return ' '.join(('skill_' + s.encode('utf-8').hex() for s in sorted(set(skills))))

def role(title):
    t = str(title).lower()
    if re.search('bangkok|singapore based|relocation|malaysia based', t):
        return None
    for name, p in [('Full-stack', 'full[\\s-]*stack'),
        ('Front-end', 'front[\\s-]*end|angular|react(?! native)|vue[.\\s]'),
        ('Back-end', 'back[\\s-]*end'),
        ('Mobile', '\\bmobile\\b|\\bandroid\\b|\\bios\\b|flutter|react native'),
        ('Digital marketing', 'digital market|digital advertis|seo specialist|social media specialist|performance market'),
        ('Graphic design', 'graphic design|desain grafis|desainer grafis|visual design'),
        ('Web development', 'web developer|web programmer|wordpress developer'),
        ('Software engineering',
            'software (engineer|developer)|\\bprogrammer\\b|java developer|\\.net developer|dotnet developer|php developer|golang developer|it developer|c# developer')]:
        if re.search(p, t):
            return name
    return None
rules = json.loads((ROOT / "config/role_rules.json").read_text(encoding="utf-8"))

def label(t):
    for c, p in rules:
        if re.search(p, str(t), re.I):
            return c
    return None
extra = {'Data analysis': 'data analytics|analisis data',
    'Statistics': 'statistic|statistik',
    'Data visualization': 'data visuali[sz]ation|visualisasi data',
    'Hadoop': '\\bhadoop\\b',
    'Scikit-learn': 'scikit.learn|sklearn',
    'LLM': 'large language model|\\bllm\\b',
    'Cloud computing': 'cloud computing',
    'Network security': 'network security|firewall',
    'Windows Server': 'windows server',
    'CCNA': '\\bccna\\b',
    'Active Directory': 'active directory',
    'Adobe XD': 'adobe xd',
    'Canva': '\\bcanva\\b',
    'Digital advertising': 'digital advertising|iklan digital',
    'Content creation': 'content creation|content creator',
    'R programming': '\\br programming\\b|\\br language\\b',
    'Data warehouse': 'data warehous',
    'IT troubleshooting': 'troubleshooting',
    'Virtualization': 'virtuali[sz]ation|vmware',
    'Automation': 'automation|otomasi'}

def skills(t):
    return sorted(set(extract(t)) | {k for k, p in extra.items() if re.search(p, str(t), re.I)})
ALIASES.update({k: [] for k in extra})

def load():
    return joblib.load(ROOT / 'models/bundle.joblib')

def score(m, x):
    if hasattr(m, 'predict_proba'):
        return m.predict_proba(x)
    v = m.decision_function(x)
    v = np.exp(v - v.max(axis=1, keepdims=True))
    return v / v.sum(axis=1, keepdims=True)

def overlap(profiles, skills, classes):
    s = set(skills)
    return np.array([sum((v for k, v in profiles[c].items() if k in s)) / max(sum(profiles[c].values()),
        1e-12) for c in classes])

def recommend(bundle, skills, model=None):
    known = sorted(set(skills) & set(bundle['vocabulary']))
    name = model or bundle['best']
    m = bundle['models'][bundle['base'] if name == 'Hybrid' else name]
    classes = m.classes_
    if not known:
        return pd.DataFrame(columns=['Career', 'Model score', 'Skill coverage'])
    p = score(m, [features(known)])[0]
    o = overlap(bundle['profiles'], known, classes)
    if name in ('Hybrid', 'Balancing experiment'):
        p = bundle['alpha'] * p + (1 - bundle['alpha']) * (o / o.sum() if o.sum() else np.ones(len(o)) / len(o))
    return pd.DataFrame({'Career': classes, 'Model score': p, 'Skill coverage': o}).sort_values('Model score',
        ascending=False,
        ignore_index=True)

def gap(bundle, career, skills):
    return pd.DataFrame([{'Skill': s, 'Demand share': v, 'Have': s in skills} for s, v in bundle['profiles'][career].items()]).sort_values('Demand share',
        ascending=False,
        ignore_index=True)

def explain(bundle, skills, career, model=None):
    base = recommend(bundle, skills, model).set_index('Career')['Model score'][career]
    rows = []
    for s in skills:
        r = recommend(bundle, [k for k in skills if k != s], model)
        rows.append({'Skill': s,
            'Score change when removed': base - r.set_index('Career')['Model score'][career] if len(r) else np.nan})
    return pd.DataFrame(rows).sort_values('Score change when removed', ascending=False)
SPEC_RULES = json.loads((ROOT / "config/specialization_rules.json").read_text(encoding="utf-8"))

def specialization(career, title):
    for name, p in SPEC_RULES.get(career, []):
        if re.search(p, str(title), re.I):
            return name
    return None

def learning_rank(bundle, career, known):
    """Rank missing skills using selected development-only policy; coverage is separate."""
    if 'gap_matrix' in bundle:
        from .gap_rank import ranking
        return ranking(bundle, career, known)
    profile = bundle['profiles'][career]
    known = set(known)
    policy = bundle.get('gap_policy', 'frequency')
    rows = []
    for skill, freq in profile.items():
        if skill in known:
            continue
        value = freq
        if policy != 'frequency':
            counts = bundle['cooccurrence'][career]
            den = bundle['skill_counts'][career]
            cond = [counts.get(tuple(sorted((s, skill))),
                0) / max(den.get(s, 0),
                1) for s in known if den.get(s,
                0) >= 3]
            conditional = float(np.mean(cond)) if cond else freq
            w = float(policy)
            value = w * conditional + (1 - w) * freq
        rows.append({'Skill': skill, 'Demand share': freq, 'Learning priority score': value})
    return pd.DataFrame(rows,
        columns=['Skill', 'Demand share', 'Learning priority score']).sort_values('Learning priority score',
        ascending=False,
        ignore_index=True)
