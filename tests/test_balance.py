import unittest
import pandas as pd
from src.core import ROOT,load,recommend
class BalanceTests(unittest.TestCase):
    def test_no_overlap_or_software_aug(self):
        a=pd.read_csv(ROOT/'data/processed/balance_augmentation.csv');d=pd.read_csv(ROOT/'data/processed/jobs.csv');self.assertFalse(set(a.x)&set(d.x));self.assertNotIn('Software Development',set(a.career))
    def test_candidate_predicts(self):
        r=recommend(load(),['Python','SQL','Pandas'],'Balancing experiment');self.assertEqual(len(r),5);self.assertAlmostEqual(r['Model score'].sum(),1)
    def test_cv_selection(self):
        m=pd.read_csv(ROOT/'results/balance_metrics.csv');v=m[m.split.eq('grouped CV')];self.assertEqual(v.sort_values('macro_f1',ascending=False).iloc[0].method,'Original Hybrid')
