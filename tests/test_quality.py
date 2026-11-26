import unittest,pandas as pd
from src.core import ROOT,load,learning_rank
class QualityTest(unittest.TestCase):
 def test_audit(self):
  a=pd.read_csv(ROOT/'results/similarity_audit.csv');self.assertEqual(a.iloc[0].retained_test,567);self.assertTrue(a.retained_test.is_monotonic_decreasing)
 def test_policy(self):
  b=load();a=pd.read_csv(ROOT/'results/gap_v3_tuning.csv');self.assertEqual(b['gap_v3_policy'],a.loc[a.cv_end_to_end_hit5.idxmax(),'policy'])
 def test_rank(self):
  a=learning_rank(load(),'Data & AI',['Python','SQL']);self.assertFalse(a.Skill.isin(['Python','SQL']).any());self.assertTrue(a['Learning priority score'].is_monotonic_decreasing)
 def test_recovery(self):
  a=pd.read_csv(ROOT/'results/gap_v3_evaluation.csv');self.assertEqual(len(a),542);self.assertTrue((a.hit10>=a.hit5).all())
