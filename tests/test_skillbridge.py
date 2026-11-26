import unittest,json
import numpy as np,pandas as pd
from src.core import ROOT,load,recommend,gap,skills,features,explain,learning_rank,specialization
class SkillBridgeTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.b=load();cls.d=pd.read_csv(ROOT/'data/processed/jobs.csv');cls.m=pd.read_csv(ROOT/'results/metrics.csv')
 def test_split(self):
  a=self.d[self.d.split=='development'];b=self.d[self.d.split=='test'];self.assertFalse(set(a.group)&set(b.group))
 def test_unique(self):self.assertFalse(self.d.duplicated(['career','x']).any())
 def test_alias(self):self.assertIn('JavaScript',skills('javascript'));self.assertIn('Statistics',skills('statistik'))
 def test_empty(self):self.assertTrue(recommend(self.b,[]).empty)
 def test_all_models(self):
  for m in list(self.b['models'])+['Hybrid']:
   r=recommend(self.b,['Python','SQL','Git'],m);self.assertEqual(len(r),len(self.b['profiles']));self.assertAlmostEqual(r['Model score'].sum(),1);self.assertTrue(r['Model score'].is_monotonic_decreasing)
 def test_gap(self):
  r=recommend(self.b,['Python','SQL']);c=r.iloc[0].Career;g=gap(self.b,c,['Python','SQL']);self.assertTrue((g.Have==g.Skill.isin(['Python','SQL'])).all())
 def test_explain(self):self.assertEqual(len(explain(self.b,['Python','SQL'],'Data & AI')),2)
 def test_selection(self):
  r=self.m[(self.m.split=='grouped CV')&~self.m.model.isin(['Majority baseline','Skill overlap'])];self.assertAlmostEqual(r.loc[r.model==self.b['best'],'macro_f1'].iloc[0],r.macro_f1.max())
 def test_metrics(self):
  r=self.m[self.m.split=='test'];self.assertTrue(((r[['accuracy','macro_f1','top3_hit','mrr','ndcg_at3']]>=0)&(r[['accuracy','macro_f1','top3_hit','mrr','ndcg_at3']]<=1)).all().all())
 def test_profiles(self):self.assertTrue(all(0<v<=1 for p in self.b['profiles'].values() for v in p.values()))
 def test_cv(self):self.assertEqual(set(self.m[self.m.split=='test'].model),set(self.m[self.m.split=='grouped CV'].model))
 def test_learning_rank(self):
  r=learning_rank(self.b,'Data & AI',['Python','SQL']);self.assertFalse(r.Skill.isin(['Python','SQL']).any());self.assertTrue(r['Learning priority score'].is_monotonic_decreasing)
 def test_specializations(self):
  s=pd.read_csv(ROOT/'results/specialization_support.csv');self.assertTrue((s.development_profiles>=10).all());self.assertEqual(specialization('Data & AI','Data Scientist'),'Data science')
 def test_gap_policy(self):
  t=pd.read_csv(ROOT/'results/gap_tuning.csv');self.assertAlmostEqual(t.loc[t.policy.astype(str)==self.b['gap_policy'],'cv_oracle_hit_at5'].iloc[0],t.cv_oracle_hit_at5.max())
 def test_source_holdout(self):
  t=pd.read_csv(ROOT/'results/source_holdout.csv');self.assertEqual(t.test_profiles.sum(),len(self.d[self.d.split=='test']))
if __name__=='__main__':unittest.main()
