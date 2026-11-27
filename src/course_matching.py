"""Conservative topic matching from historical curriculum metadata, not live course offers."""
import pandas as pd
from .core import ROOT, skills

CATALOG = ROOT / 'data/processed/learning_topics.csv'


def build():
    raw = pd.read_csv(ROOT / 'data/raw/digital/course_descriptions.csv')
    cols = ['course_title', 'technology_focus']
    topics = raw[cols].dropna().drop_duplicates().sort_values(cols).copy()
    topics['Skills'] = (topics.course_title + ' ' + topics.technology_focus).map(lambda x: '|'.join(skills(x)))
    topics = topics[topics.Skills.ne('')].rename(columns={'course_title': 'Learning topic', 'technology_focus': 'Technology focus'})
    topics.to_csv(CATALOG, index=False)
    recognized = set('|'.join(topics.Skills).split('|'))
    hidden = pd.read_csv(ROOT / 'results/gap_evaluation.csv').hidden
    audit = {
        'raw_rows': len(raw),
        'unique_titles': raw.course_title.nunique(),
        'unique_focus_patterns': raw.technology_focus.nunique(),
        'unique_recognized_topic_focus_pairs': len(topics),
        'recognized_skills': len(recognized),
        'masked_profiles': len(hidden),
        'masked_skills_covered': int(hidden.isin(recognized).sum()),
    }
    pd.DataFrame([audit]).to_csv(ROOT / 'results/course_audit.csv', index=False)
    return topics


def match_topics(missing, catalog=None, limit=10):
    """Rank exact normalized matches by missing-skill priority, never recommend unrelated topics."""
    catalog = pd.read_csv(CATALOG) if catalog is None else catalog
    priority = dict(zip(missing.Skill, missing['Learning priority score']))
    rows = []
    for row in catalog.to_dict('records'):
        matched = sorted(set(str(row['Skills']).split('|')) & set(priority))
        if matched:
            rows.append({'Learning topic': row['Learning topic'], 'Matched missing skills': ', '.join(matched),
                         'Priority score': sum(priority[s] for s in matched)})
    columns = ['Learning topic', 'Matched missing skills', 'Priority score']
    return pd.DataFrame(rows, columns=columns).sort_values(['Priority score', 'Learning topic'], ascending=[False, True]).drop_duplicates('Learning topic').head(limit).reset_index(drop=True)


if __name__ == '__main__':
    build()
