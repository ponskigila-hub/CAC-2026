import unittest
import numpy as np
import pandas as pd
from src.forecast import *
from src.train import choose_threshold

class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d,cls.b=load();z=eligible(cls.d);cls.series=int(z[(z.province=='JAWA BARAT')&(z.commodity=='Rice (medium quality)')&(z.date==pd.Timestamp('2025-12-01'))].series.iloc[0])
    def test_unique_months(self):self.assertFalse(self.d.duplicated(KEY+['date']).any())
    def test_prices_positive(self):self.assertTrue((self.d.price>0).all())
    def test_origin_features_finite(self):self.assertTrue(np.isfinite(features(eligible(self.d).head(500),25)).all().all())
    def test_monthly_forecast_jan_2028(self):
        o,p=forecast_path(self.d,self.series,'2028-01-31',self.b);self.assertEqual(p.iloc[-1].date,pd.Timestamp('2028-01-01'));self.assertEqual(p.iloc[-1].horizon,25);self.assertFalse(p.iloc[-1].extrapolation);self.assertTrue(np.isfinite(p.iloc[-1].risk_score))
    def test_same_month_dates(self):
        _,a=forecast_path(self.d,self.series,'2028-01-01',self.b);_,b=forecast_path(self.d,self.series,'2028-01-31',self.b);pd.testing.assert_frame_equal(a,b)
    def test_price_change(self):
        o,p=forecast_path(self.d,self.series,'2028-01-31',self.b);np.testing.assert_allclose(p.change,p.price.to_numpy()/np.r_[o.price,p.price.to_numpy()[:-1]]-1)
    def test_risk_rule(self):
        _,p=forecast_path(self.d,self.series,'2028-01-31',self.b);self.assertEqual(p.iloc[-1].shock,p.iloc[-1].risk_score>=p.iloc[-1].threshold)
    def test_far_future(self):
        _,p=forecast_path(self.d,self.series,'2035-02-28',self.b);self.assertTrue(p.iloc[-1].extrapolation);self.assertTrue(pd.isna(p.iloc[-1].risk_score));self.assertTrue(p.iloc[-1].price>0)
    def test_old_target_rejected(self):
        with self.assertRaises(ValueError):forecast_path(self.d,self.series,'2025-12-31',self.b)
    def test_horizon_aware(self):
        _,p=forecast_path(self.d,self.series,'2028-01-31',self.b,'Histogram Gradient Boosting');self.assertGreater(p.price.nunique(),1)
    def test_threshold_fpr(self):
        rng=np.random.default_rng(42);y=rng.integers(0,2,1000);scores=rng.random(1000);t=choose_threshold(y,scores);self.assertLessEqual(((scores>=t)&(y==0)).sum()/(y==0).sum(),.1)
    def test_training_label_cutoff(self):
        sub=self.d[self.d.series.isin(self.d.series.unique()[:20])];p=pairs(sub,'2022-12-01');self.assertTrue((p.target_m<=2022*12+12).all());self.assertTrue((p.target_m-p.origin_m==p.h).all())
    def test_no_future_feature_dependency(self):
        raw=pd.read_csv(ROOT/'data/raw/indonesia.csv.gz',low_memory=False);raw=raw[raw.market_id==raw.market_id.iloc[0]];old=raw[pd.to_datetime(raw.date)<='2019-12-31'];full=prepare(raw);pre=prepare(old);cols=['market_id','commodity_id','unit','date','mom1','mom3','mom6','mom12','vol3','history','trend'];a=full[full.date<='2019-12-31'][cols].reset_index(drop=True);pd.testing.assert_frame_equal(a,pre[cols].reset_index(drop=True))
    def test_backtest_report_consistency(self):
        p=pd.read_csv(ROOT/'results/test_predictions.csv.gz');m=pd.read_csv(ROOT/'results/forecast_metrics.csv');r=m[(m.phase.str.startswith('Test'))&(m.model==self.b['selected'])&(m.horizon.astype(str)=='All')].iloc[0];self.assertAlmostEqual(np.abs(p.prediction/p.actual-1).mean()*100,r.MAPE,places=8)
    def test_interval_order(self):
        _,p=forecast_path(self.d,self.series,'2028-01-31',self.b);self.assertTrue((p.lower<=p.upper).all())
    def test_auto_model_selection(self):
        _,p=forecast_path(self.d,self.series,'2028-01-31',self.b)
        self.assertEqual(p.price_model.tolist(),[self.b['price_policy'][str(horizon_group(h))] for h in p.horizon])
        for _,r in p.iterrows():self.assertEqual(r.threshold,self.b['thresholds'][str(horizon_group(r.horizon))])
    def test_validation_bucket_fpr(self):
        m=pd.read_csv(ROOT/'results/shock_metrics.csv');v=m[(m.phase=='Validation pooled')&m.horizon.isin(GROUPS)]
        self.assertTrue((v.FPR<=.100001).all())
    def test_backtest_shock_report(self):
        p=pd.read_csv(ROOT/'results/test_predictions.csv.gz').dropna(subset=['shock']);m=pd.read_csv(ROOT/'results/shock_metrics.csv');r=m[(m.phase.str.startswith('Test'))&(m.model==self.b['risk_selected'])&(m.horizon=='All')].iloc[0]
        self.assertEqual(int(((p.shock==1)&p.alert).sum()),int(r.TP));self.assertEqual(int(((p.shock==0)&p.alert).sum()),int(r.FP));self.assertTrue((p.alert==(p.risk_score>=p.threshold)).all())
    def test_auto_price_matches_selected_model(self):
        _,p=forecast_path(self.d,self.series,'2028-01-31',self.b)
        for name in p.price_model.unique():
            _,q=forecast_path(self.d,self.series,'2028-01-31',self.b,name);np.testing.assert_allclose(p.loc[p.price_model==name,'price'],q.loc[p.price_model==name,'price'])
    def test_deployment_policy_matches_validation_selection(self):
        m=pd.read_csv(ROOT/'results/forecast_metrics.csv');v=m[m.phase.eq('Validation pooled') & m.model.ne(self.b['selected'])]
        for g in GROUPS:self.assertEqual(self.b['price_policy'][g],v[v.horizon.eq(g)].sort_values('MAPE').iloc[0].model)
    def test_rolling_audit(self):
        audit=pd.read_csv(ROOT/'results/split_audit.csv');self.assertEqual(len(audit[audit.phase.eq('Validation')]),4);self.assertEqual(len(audit[audit.phase.eq('Retrospective test')]),4)
if __name__=='__main__':unittest.main()
