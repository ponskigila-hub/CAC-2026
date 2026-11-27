import unittest
import numpy as np
import pandas as pd
from src.core import ROOT, load, learning_rank
from src.statistical_tests import paired
from src.course_matching import match_topics


class ReviewImprovements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = pd.read_csv(ROOT / 'data/processed/jobs.csv').set_index('id')
        cls.pred = pd.read_csv(ROOT / 'results/test_predictions.csv')
        cls.balance = pd.read_csv(ROOT / 'results/balance_test_predictions.csv')

    def test_balancing_prediction_alignment(self):
        self.assertTrue(self.pred.id.is_unique)
        self.assertEqual(set(self.pred.id), set(self.balance.id))
        self.assertTrue(self.jobs.loc[self.pred.id, 'split'].eq('test').all())
        aligned = self.balance.set_index('id').loc[self.pred.id]
        self.assertTrue(np.array_equal(aligned['Original Hybrid'], self.pred.Hybrid))
        self.assertTrue(np.array_equal(aligned['Moderate class weighting'], self.pred['Balancing experiment']))

    def test_paired_zero(self):
        p = self.pred[['id', 'career', 'source', 'Hybrid']].copy()
        p['Same'] = p.Hybrid
        out = paired(p, self.jobs.loc[p.id, 'group'], 'Hybrid', n_boot=30)
        self.assertTrue((out[['difference_pp','ci_low_pp','ci_high_pp']]==0).all().all())
        self.assertFalse(out.excludes_zero.any())

    def test_paired_effect_matches_report(self):
        p = pd.read_csv(ROOT/'results/paired_model_ci.csv')
        report = pd.read_csv(ROOT/'results/metrics.csv').query('split == "test"').set_index('model')
        for row in p[p.metric.eq('macro_f1')].itertuples():
            expected = 100*(report.loc['Hybrid','macro_f1']-report.loc[row.comparison,'macro_f1'])
            self.assertAlmostEqual(row.difference_pp, expected)
        self.assertTrue((p.ci_low_pp<=p.ci_high_pp).all())
        self.assertTrue(p.replicates.eq(1000).all())

    def test_nested_development_only(self):
        p = pd.read_csv(ROOT/'results/nested_cv_predictions.csv')
        self.assertTrue(p.id.is_unique)
        self.assertEqual(set(p.id),set(self.jobs[self.jobs.split.eq('development')].index))
        table = pd.read_csv(ROOT/'results/nested_cv.csv')
        self.assertEqual(len(table),5)
        self.assertAlmostEqual(table.iloc[-1].accuracy,(p.prediction==p.career).mean())

    def test_topic_matching(self):
        missing = learning_rank(load(),'Data & AI',['Python','SQL'])
        r=match_topics(missing)
        self.assertFalse(r['Learning topic'].duplicated().any())
        self.assertTrue(r['Priority score'].is_monotonic_decreasing)
        for ss in r['Matched missing skills']:
            self.assertTrue(set(ss.split(', '))<=set(missing.Skill))
        self.assertTrue(match_topics(missing.iloc[:0]).empty)

    def test_profile_audit(self):
        q=pd.read_csv(ROOT/'results/profile_quality_summary.csv').iloc[0]
        self.assertEqual(q.ambiguous_profiles,127)
        self.assertEqual(q.at_most_two_skills,449)

    def test_blind_review(self):
        a,b=[pd.read_csv(ROOT/'review'/f'{n}.csv') for n in ['reviewer_a','reviewer_b']]
        self.assertEqual(set(a.id),set(b.id))
        self.assertEqual(len(a),90)
        self.assertNotIn('career',a.columns)
        self.assertNotIn('Hybrid',a.columns)
        self.assertTrue(self.jobs.loc[a.id,'split'].eq('test').all())
