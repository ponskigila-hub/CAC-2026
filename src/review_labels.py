"""Independent double-review workflow. Blank or disputed labels never count as truth."""
import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from .core import ROOT

REVIEW = ROOT / 'review'
LABELS = ['Data & AI', 'Design', 'Digital Marketing', 'Infrastructure & Security', 'Software Development']


def generate(include_text=False):
    REVIEW.mkdir(exist_ok=True)
    jobs = pd.read_csv(ROOT / 'data/processed/jobs.csv')
    sample = jobs[jobs.split.eq('test')].groupby('career', group_keys=False).sample(n=18, random_state=91)
    sample = sample.sample(frac=1, random_state=92)[['id','source','title','skills']]
    if include_text:
        from .train import prepare
        original, _ = prepare()
        check = original.set_index('id').loc[jobs.id]
        assert check.x.tolist() == jobs.x.tolist(), 'Profile alignment changed.'
        sample = sample.merge(original[['id', 'text']], on='id', validate='one_to_one')
        sample = sample.rename(columns={'text': 'original_evidence'})
    for name in ['reviewer_a','reviewer_b']:
        path = REVIEW / f'{name}.csv'
        if path.exists():
            existing = pd.read_csv(path, keep_default_na=False)
            if existing.human_label.ne('').any() or existing.evidence.ne('').any():
                continue
            if not include_text or 'original_evidence' in existing:
                continue
        sample.assign(human_label='', evidence='').to_csv(path,index=False)


def evaluate():
    a,b = [pd.read_csv(REVIEW / f'{n}.csv',keep_default_na=False) for n in ['reviewer_a','reviewer_b']]
    if not a.id.is_unique or not b.id.is_unique or set(a.id)!=set(b.id):
        raise ValueError('Reviewer IDs must be unique and identical.')
    j=a[['id','human_label']].merge(b[['id','human_label']],on='id',suffixes=('_a','_b'))
    allowed=set(LABELS)|{'Uncertain','Other',''}
    if not set(j.human_label_a).union(j.human_label_b)<=allowed:
        raise ValueError('Use a documented label, Uncertain or Other.')
    complete=j[j.human_label_a.ne('')&j.human_label_b.ne('')]
    if complete.empty:
        print('No paired human reviews completed. No scores calculated.');return
    consensus=complete[complete.human_label_a.eq(complete.human_label_b)&complete.human_label_a.isin(LABELS)]
    result={'sample':len(j),'paired_completed':len(complete),'agreement':(complete.human_label_a==complete.human_label_b).mean(), 'cohen_kappa':cohen_kappa_score(complete.human_label_a,complete.human_label_b),'usable_consensus':len(consensus)}
    if len(consensus):
        p=pd.read_csv(ROOT/'results/test_predictions.csv');z=consensus.merge(p,on='id',validate='one_to_one')
        result.update({'regex_vs_consensus_accuracy':accuracy_score(z.human_label_a,z.career),'hybrid_vs_consensus_accuracy':accuracy_score(z.human_label_a,z.Hybrid),'hybrid_vs_consensus_macro_f1':f1_score(z.human_label_a,z.Hybrid,labels=LABELS,average='macro',zero_division=0)})
    pd.DataFrame([result]).to_csv(REVIEW/'review_summary.csv',index=False)
    print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');parser.add_argument('--with-text', action='store_true');args=parser.parse_args()
    evaluate() if args.evaluate else generate(include_text=args.with_text)
