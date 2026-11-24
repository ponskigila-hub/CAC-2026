"""Integration tests for the exact V9 submission artifacts; no network or UI needed."""
from __future__ import annotations
import unittest, tempfile, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.models import (validate_panel,split_panel,train_and_evaluate,make_bundle,predict_two_stage,_month_block_cal_split)
from core.utils import FEATURE_COLS,CAT_COLS,price_change,shock_event,alert_mask,classify_shock,calibration_offset
from core.features import scenario_features,SCENARIO_LAGS
from core.model_store import load_bundle,verify_results
from src.prepare_data import rebuild_raw
DATA=ROOT/'data'

class TestV9(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=verify_results(); cls.panel=validate_panel(pd.read_csv(DATA/'foodshock_asean_model_ready.csv'))
        cls.lookup=cls.panel.set_index(['series_id','date']); cls.oos=pd.read_csv(DATA/'v9_oos_predictions.csv',parse_dates=['date'])
        cls.cat=cls.oos[cls.oos.model=='CatBoost'].copy();cls.holdout=pd.read_csv(DATA/'v9_official_holdout_predictions.csv',parse_dates=['date'])

    def test_panel(self):
        d=self.panel; self.assertEqual(len(d),self.manifest['dataset_rows']);self.assertEqual(d.shape[1],39)
        self.assertFalse(d.duplicated(['series_id','date']).any());self.assertEqual(int(d.isna().sum().sum()),0)
        np.testing.assert_allclose(d.country_vs_asean_mom,d.pct_change_1m-d.asean_comm_mom_1m,atol=1e-12)
        np.testing.assert_array_equal(d.shock_label,shock_event(price_change(d.usdprice,d.target_price_usd_t1)))

    def test_purged_splits(self):
        for c in self.manifest['cutoffs']:
            cutoff=pd.Timestamp(c);fit,cal,test,_=split_panel(self.panel,cutoff,cutoff+pd.DateOffset(months=12))
            self.assertLess(fit.date.max()+pd.DateOffset(months=1),cal.date.min())
            self.assertLess(cal.date.max()+pd.DateOffset(months=1),test.date.min())
            tune,val=_month_block_cal_split(fit,.2);self.assertLess(tune.date.max()+pd.DateOffset(months=1),val.date.min())
            self.assertEqual(len(test),len(self.cat[self.cat.fold==c]))
        self.assertFalse(self.cat.duplicated(['series_id','date']).any());self.assertEqual(len(self.cat),self.manifest['oos_rows'])

    def assert_replay(self,bundle,rows):
        for row in rows.itertuples(index=False):
            f=self.lookup.loc[(row.series_id,row.date)].to_dict();o=predict_two_stage(bundle,f)
            np.testing.assert_allclose([o['y_hat'],o['y_p10'],o['y_p90'],o['shock_proba']],[row.predicted,row.lower,row.upper,row.score],atol=1e-8,rtol=1e-8)
            self.assertEqual(o['alert'],bool(row.alert)); self.assertEqual(bundle['threshold'],row.threshold)
            self.assertEqual(o['alert'],classify_shock(o['change'],o['shock_proba'],bundle['threshold'])[0]=='FOOD SHOCK ALERT')
            self.assertLessEqual(o['y_p10'],o['y_p90'])

    def test_all_fold_model_replays(self):
        for c in self.manifest['cutoffs']:
            b=load_bundle(c,verify_data=False);rows=self.cat[self.cat.fold==c].sample(8,random_state=42);self.assert_replay(b,rows)

    def test_official_model_replay(self):
        b=load_bundle();self.assert_replay(b,self.holdout.sample(12,random_state=42)); self.assertEqual(b['params'],self.manifest['official_params']); self.assertEqual(b['qcal'],self.manifest['official_qcal'])

    def test_aggregate_evaluation(self):
        c=self.cat; from sklearn.metrics import precision_score,recall_score
        # Vector thresholds are fold-specific, never the mean threshold.
        pred=alert_mask(price_change(c.current,c.predicted),c.score,c.threshold)
        np.testing.assert_array_equal(pred,c.alert.astype(bool));np.testing.assert_array_equal(shock_event(price_change(c.current,c.actual)),c.shock)
        shock=pd.read_csv(DATA/'model_eval_shock_classification.csv').set_index('method').loc['Hybrid_Operational_Alert']
        self.assertAlmostEqual(precision_score(c.shock,pred),shock.Precision);self.assertAlmostEqual(recall_score(c.shock,pred),shock.Recall)
        self.assertAlmostEqual(((c.actual>=c.lower)&(c.actual<=c.upper)).mean(),shock.Overall_PICP_P10_P90)
        reg=pd.read_csv(DATA/'model_evaluation_results.csv').set_index('model');self.assertAlmostEqual(np.sqrt(((c.actual-c.predicted)**2).mean()),reg.loc['CatBoost','RMSE_micro']);self.assertAlmostEqual(np.sqrt(((c.actual-c.current)**2).mean()),reg.loc['Persistence (y_t)','RMSE_micro'])
        hp=pd.read_csv(DATA/'hyperparameter_tuning_summary.csv');self.assertEqual(len(hp),35);self.assertEqual(int(hp.trials.sum()),3*len(self.manifest['cutoffs'])*self.manifest['trials_per_tuned_model'])

    def test_threshold_boundary(self):
        self.assertFalse(bool(shock_event(price_change(1.,1.05))));self.assertTrue(bool(shock_event(price_change(1.,1.05001))))
        self.assertEqual(classify_shock(.05,.1,.7)[0],'WASPADA / ELEVATED RISK');self.assertEqual(classify_shock(2.,.1,.7)[0],'FOOD SHOCK ALERT');self.assertTrue(bool(alert_mask(0.,.7,.7)))
        self.assertEqual(calibration_offset([0,1,2,3,4]),4.)

    def test_scenario_features(self):
        ref=self.panel.iloc[0].to_dict(); h={i:float(2+i*.1) for i in SCENARIO_LAGS};row=scenario_features(ref,3.,h,.02,'2024-06-01')
        self.assertAlmostEqual(row['rolling_mean_6'],np.mean([h[i] for i in range(1,7)]));self.assertAlmostEqual(row['rolling_std_6'],np.std([h[i] for i in range(1,7)],ddof=1));self.assertAlmostEqual(row['pct_change_1m'],h[1]/h[2]-1);self.assertAlmostEqual(row['pct_change_3m'],h[1]/h[4]-1);self.assertAlmostEqual(row['pct_change_12m'],h[1]/h[13]-1);self.assertAlmostEqual(row['country_vs_asean_mom'],row['pct_change_1m']-.02)
        self.assertAlmostEqual(row['fx_pct_1m'],ref['fx_pct_1m']);self.assertAlmostEqual(row['usdprice'],3.)
        with self.assertRaises(ValueError): scenario_features(ref,0.,h,0.,'2024-06-01')
        with self.assertRaises(ValueError): predict_two_stage(load_bundle(verify_data=False),{'usdprice':1.})

    def test_custom_residual_and_direct(self):
        # Fast real training for the interactive code path; not a reported benchmark.
        for mode in ('residual','direct'):
            o=train_and_evaluate(self.panel,algorithm='CatBoost',target_mode=mode,n_estimators=20,max_depth=3)
            b=make_bundle(o); r=o['test_rows'].iloc[0].to_dict();p=predict_two_stage(b,r)
            self.assertAlmostEqual(p['y_hat'],o['y_pred'][0]);self.assertAlmostEqual(p['shock_proba'],o['shock_proba'][0]);self.assertEqual(p['alert'],bool(o['alerts'][0]))
            if mode=='direct': self.assertIsNone(p['y_p10']);self.assertIsNone(p['y_p90']);self.assertTrue(np.isnan(o['metrics']['PICP_P10_P90']))
            else: self.assertAlmostEqual(p['y_p10'],o['y_lo'][0]);self.assertAlmostEqual(p['y_p90'],o['y_hi'][0])

    def test_raw_rebuild_fixture(self):
        # Synthetic fixture only, never used in competition metrics.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); dates=pd.date_range('2015-01-01',periods=70,freq='MS'); v=1+np.arange(len(dates))*.01
            pd.DataFrame(dict(countryiso3='IDN',date=dates,market='Fixture',commodity='Rice',pricetype='Retail',priceflag='actual',unit='KG',price=v*10000,usdprice=v)).to_csv(path/'raw.csv',index=False)
            rebuild_raw(path/'raw.csv',path);d=validate_panel(pd.read_csv(path/'foodshock_asean_model_ready.csv'));self.assertGreater(len(d),0);self.assertEqual(d.date.min(),dates[13]);self.assertEqual(d.date.max(),dates[-2]);self.assertAlmostEqual(d.iloc[0].lag_1,v[12])

if __name__=='__main__':unittest.main(verbosity=2)
