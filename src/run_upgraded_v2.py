#!/usr/bin/env python3
"""FoodShock V4 — fold-safe encoding, month-block inner val, conformal quantiles, low-FAR threshold."""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "figures"
RAW_CANDIDATES = [Path("/home/workdir/attachments/asean_wfp_2015_2025.csv"), ROOT / "asean_wfp_2015_2025.csv"]
RS = 42
np.random.seed(RS)
BASE = ["usdprice","lag_1","lag_2","lag_3","lag_6","lag_12","rolling_mean_3","rolling_mean_6","rolling_std_6",
        "rolling_min_6","rolling_max_6","price_to_mean6","cv_6","pct_change_1m","pct_change_3m","pct_change_12m","month_sin","month_cos"]
MACRO = ["fx_pct_1m","fx_pct_3m","fx_vol_6m","asean_comm_mom_1m","country_vs_asean_mom"]
CATS = ["countryiso3","commodity_group","pricetype"]
FEATS = BASE + MACRO
SHOCK_PCT = 0.05
MIN_PREC_OPS = 0.20  # operational precision floor


def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    d = (np.abs(y_true)+np.abs(y_pred))/2.0; m = d!=0
    return float(np.mean(np.abs(y_true[m]-y_pred[m])/d[m])*100) if m.any() else 0.0


def diebold_mariano_panel(e1, e2, year_month, max_lag=3):
    from scipy import stats
    df = pd.DataFrame({"e1":np.asarray(e1,float),"e2":np.asarray(e2,float),"ym":pd.Series(year_month).astype(str).values}).dropna()
    df["di"] = df["e1"]**2 - df["e2"]**2
    d_t = df.groupby("ym")["di"].mean().sort_index().values
    n = len(d_t)
    if n < 8: return np.nan, np.nan
    mean_d = float(d_t.mean()); gamma0 = float(np.var(d_t, ddof=1)); hac = gamma0
    for lag in range(1, min(max_lag, n-1)+1):
        w = 1.0 - lag/(min(max_lag,n-1)+1)
        hac += 2*w*float(np.cov(d_t[lag:], d_t[:-lag], ddof=1)[0,1])
    if hac <= 0: return np.nan, np.nan
    stat = mean_d / np.sqrt(hac/n)
    return float(stat), float(2*(1-stats.norm.cdf(abs(stat))))


def build_implied_fx(panel):
    raw = next((p for p in RAW_CANDIDATES if p.exists()), None)
    if raw is not None:
        print("FX from raw", raw)
        r = pd.read_csv(raw, usecols=["countryiso3","date","price","usdprice"], low_memory=False)
        r["date"]=pd.to_datetime(r["date"]); r["year_month"]=r["date"].dt.to_period("M")
        r = r[(r["usdprice"]>0)&(r["price"]>0)]; r["implied_fx"]=r["price"]/r["usdprice"]
        fx = r.groupby(["countryiso3","year_month"])["implied_fx"].median().reset_index().sort_values(["countryiso3","year_month"])
        fx["fx_pct_1m"]=fx.groupby("countryiso3")["implied_fx"].pct_change(1, fill_method=None)
        fx["fx_pct_3m"]=fx.groupby("countryiso3")["implied_fx"].pct_change(3, fill_method=None)
        fx["fx_vol_6m"]=fx.groupby("countryiso3")["fx_pct_1m"].transform(lambda s: s.shift(1).rolling(6,min_periods=3).std())
        return fx
    print("FX fallback: precomputed columns in model-ready panel")
    p = panel.copy(); p["date"]=pd.to_datetime(p["date"]); p["year_month"]=p["date"].dt.to_period("M")
    if set(MACRO[:3]).issubset(p.columns):
        return p.groupby(["countryiso3","year_month"], as_index=False)[MACRO[:3]].mean()
    fx = p[["countryiso3","year_month"]].drop_duplicates()
    for c in MACRO[:3]: fx[c]=0.0
    return fx


def enrich_panel(mr, fx):
    df = mr.copy(); df["date"]=pd.to_datetime(df["date"]); df["year_month"]=df["date"].dt.to_period("M")
    df = df.drop(columns=[c for c in MACRO if c in df.columns], errors="ignore")
    fx2=fx.copy(); fx2["countryiso3"]=fx2["countryiso3"].astype(str); df["countryiso3"]=df["countryiso3"].astype(str)
    fx2["year_month"]=pd.PeriodIndex(fx2["year_month"].astype(str), freq="M")
    df["year_month"]=pd.PeriodIndex(df["year_month"].astype(str), freq="M")
    df = df.merge(fx2[["countryiso3","year_month"]+MACRO[:3]], on=["countryiso3","year_month"], how="left")
    if "pct_change_1m" not in df.columns:
        df = df.sort_values(["countryiso3","commodity_group","pricetype","date"])
        df["pct_change_1m"]=df.groupby(["countryiso3","commodity_group","pricetype"])["usdprice"].pct_change(1, fill_method=None)
    reg = df.groupby(["year_month","commodity_group"], as_index=False)["pct_change_1m"].mean().rename(columns={"pct_change_1m":"raw"})
    reg = reg.sort_values(["commodity_group","year_month"]); reg["asean_comm_mom_1m"]=reg.groupby("commodity_group")["raw"].shift(1)
    df = df.merge(reg[["year_month","commodity_group","asean_comm_mom_1m"]], on=["year_month","commodity_group"], how="left")
    df["country_vs_asean_mom"]=df["pct_change_1m"]-df["asean_comm_mom_1m"]
    for c in MACRO:
        if c not in df.columns: df[c]=0.0
        df[c]=df[c].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    if "target_pct_change_t1" not in df.columns:
        df["target_pct_change_t1"]=(df["target_price_usd_t1"]-df["usdprice"])/df["usdprice"].clip(1e-6)
    df["residual_from_yt"]=df["target_price_usd_t1"]-df["usdprice"]
    df["shock_label"]=(df["target_pct_change_t1"]>SHOCK_PCT).astype(int)
    return df


def walk_forward_folds(df):
    folds=[]
    for c in pd.to_datetime(["2019-06-01","2020-06-01","2021-06-01","2022-06-01","2023-06-01","2024-06-01"]):
        tr=df["date"]<c; te=(df["date"]>=c)&(df["date"]<c+pd.DateOffset(months=12))
        if tr.sum()>500 and te.sum()>50: folds.append((tr.values, te.values, str(c.date())))
    return folds


def fit_encoders(train_df):
    enc={}
    for c in CATS:
        le=LabelEncoder(); le.fit(train_df[c].astype(str)); enc[c]=le
    return enc


def transform_X(df, enc):
    out=df[FEATS].copy()
    for c in CATS:
        le=enc[c]; vals=df[c].astype(str)
        # unseen -> -1 then map to 0
        known=set(le.classes_)
        mapped=[]
        for v in vals:
            if v in known: mapped.append(int(le.transform([v])[0]))
            else: mapped.append(-1)
        out[c+"_enc"]=mapped
    return out


def month_block_split(train_df, val_ratio=0.2):
    """Split train fold by sorted unique year_month blocks (no month leakage)."""
    months = sorted(train_df["year_month"].astype(str).unique())
    if len(months) < 5:
        n=len(train_df); cut=int(n*(1-val_ratio)); return train_df.iloc[:cut], train_df.iloc[cut:]
    n_val = max(1, int(len(months)*val_ratio))
    val_months = set(months[-n_val:])
    tr = train_df[~train_df["year_month"].astype(str).isin(val_months)]
    va = train_df[train_df["year_month"].astype(str).isin(val_months)]
    if len(tr)<50 or len(va)<20:
        n=len(train_df); cut=int(n*(1-val_ratio)); return train_df.iloc[:cut], train_df.iloc[cut:]
    return tr, va


def make_reg(name):
    if name=="LightGBM":
        return lgb.LGBMRegressor(n_estimators=120, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=30, reg_lambda=1.0, random_state=RS, n_jobs=-1, verbose=-1)
    if name=="XGBoost":
        return xgb.XGBRegressor(n_estimators=120, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=RS, n_jobs=-1, verbosity=0)
    if name=="CatBoost":
        return CatBoostRegressor(iterations=120, depth=5, learning_rate=0.05, random_seed=RS, verbose=0, allow_writing_files=False)
    if name=="HistGradientBoosting":
        return HistGradientBoostingRegressor(max_iter=120, max_depth=5, learning_rate=0.05, random_state=RS)
    return RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_leaf=5, n_jobs=-1, random_state=RS)


def make_clf():
    return lgb.LGBMClassifier(n_estimators=120, max_depth=4, min_child_samples=50, reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8, learning_rate=0.03, class_weight="balanced", random_state=RS, n_jobs=-1, verbose=-1)


def select_thresholds(y_va, p_va):
    """F1-optimal thr + operational thr.
    Primary: max Recall s.t. Precision >= MIN_PREC_OPS (0.20).
    Fallback: maximize F0.5 if no thr meets precision floor.
    Returns (thr_f1, thr_ops, rule_used) where rule_used in {'precision_floor','f0.5_fallback'}.
    """
    best_f1, thr_f1 = -1.0, 0.5
    best_ops, thr_ops = -1.0, 0.5
    best_f05, thr_f05 = -1.0, 0.5
    found_floor = False
    for thr in np.linspace(0.15, 0.85, 29):
        pred = (p_va >= thr).astype(int)
        prec = precision_score(y_va, pred, zero_division=0)
        rec = recall_score(y_va, pred, zero_division=0)
        f1 = f1_score(y_va, pred, zero_division=0)
        f05 = (1.25*prec*rec/(0.25*prec+rec)) if (prec+rec)>0 else 0.0
        if f1 > best_f1: best_f1, thr_f1 = f1, thr
        if f05 > best_f05: best_f05, thr_f05 = f05, thr
        if prec >= MIN_PREC_OPS and rec > best_ops:
            best_ops, thr_ops = rec, thr
            found_floor = True
    if found_floor:
        return float(thr_f1), float(thr_ops), "precision_floor"
    return float(thr_f1), float(thr_f05), "f0.5_fallback"


def regenerate_figures(results_df, oof_best, meta, shock_pack, best_name, model_obj, X_bg):
    FIG.mkdir(exist_ok=True)
    r = results_df.sort_values("RMSE_micro")
    colors = ["#94A3B8" if "Persistence" in str(m) else "#10B981" for m in r["model"]]
    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    axes[0].barh(r["model"].astype(str), r["RMSE_micro"], color=colors); axes[0].set_title("WF RMSE"); axes[0].set_xlabel("RMSE")
    axes[1].barh(r["model"].astype(str), r["MAE_micro"], color=colors); axes[1].set_title("WF MAE"); axes[1].set_xlabel("MAE")
    plt.tight_layout(); fig.savefig(FIG/"model_comparison.png", dpi=150); plt.close(fig)

    y, pred = oof_best["y"], oof_best["pred"]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y, pred, alpha=0.18, s=9, c="#10B981")
    lims=[min(y.min(),pred.min()), max(np.quantile(y,0.99), np.quantile(pred,0.99))]
    ax.plot(lims, lims, "r--", lw=1.3)
    ax.set_title(f"{best_name}\nMAE={mean_absolute_error(y,pred):.4f} RMSE={np.sqrt(mean_squared_error(y,pred)):.4f} R2={r2_score(y,pred):.4f}")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    plt.tight_layout(); fig.savefig(FIG/"actual_vs_predicted.png", dpi=150); plt.close(fig)

    try:
        import shap
        idx = np.random.RandomState(42).choice(len(X_bg), size=min(300,len(X_bg)), replace=False)
        Xs = X_bg.iloc[idx]
        # CatBoost / LGBM TreeExplainer
        exp = shap.TreeExplainer(model_obj)
        sv = exp.shap_values(Xs)
        plt.figure(figsize=(8,5.2)); shap.summary_plot(sv, Xs, show=False, max_display=12)
        plt.tight_layout(); plt.savefig(FIG/"shap_summary.png", dpi=150, bbox_inches="tight"); plt.close("all")
        print("SHAP ok")
    except Exception as e:
        print("SHAP fallback", e)
        fig,ax=plt.subplots(); ax.text(0.5,0.5,str(e),ha="center"); ax.axis("off"); fig.savefig(FIG/"shap_summary.png"); plt.close(fig)

    proba, ysh, thr = shock_pack["proba"], shock_pack["y"], shock_pack["thr"]
    fig, axes = plt.subplots(1,2,figsize=(11,4.2))
    axes[0].hist(proba[ysh==0], bins=40, alpha=0.7, density=True, label="Non-shock", color="#94A3B8")
    axes[0].hist(proba[ysh==1], bins=40, alpha=0.7, density=True, label="Shock >+5%", color="#F43F5E")
    axes[0].axvline(thr, ls="--", color="k", label=f"Ops thr={thr:.2f}"); axes[0].legend(fontsize=8); axes[0].set_title("P(Shock) by label")
    axes[1].hist(pred-y, bins=50, color="#10B981", alpha=0.85); axes[1].set_title("OOF residual ŷ-y")
    plt.tight_layout(); fig.savefig(FIG/"shock_residual_dist.png", dpi=150); plt.close(fig)

    if meta is not None and len(meta):
        meta=meta.copy(); meta["ae"]=np.abs(meta["y_true"]-meta["y_pred"])
        by=meta.groupby("commodity_group")["ae"].mean().sort_values()
        fig,ax=plt.subplots(figsize=(8,4)); ax.bar(by.index.astype(str), by.values, color="#10B981"); ax.set_title("MAE by commodity")
        plt.xticks(rotation=15,ha="right"); plt.tight_layout(); fig.savefig(FIG/"error_by_commodity.png", dpi=150); plt.close(fig)

    panel=pd.read_csv(DATA/"foodshock_asean_model_ready.csv"); panel["date"]=pd.to_datetime(panel["date"])
    tr=panel.groupby(["date","commodity_group"])["usdprice"].mean().reset_index()
    fig,ax=plt.subplots(figsize=(10,4.2))
    for g in tr["commodity_group"].unique():
        gg=tr[tr["commodity_group"]==g]; ax.plot(gg["date"], gg["usdprice"], label=g, lw=1.3)
    ax.legend(fontsize=8); ax.set_title("Avg USD price by commodity"); plt.tight_layout(); fig.savefig(FIG/"eda_trends.png", dpi=150); plt.close(fig)
    print("figures done")


def main():
    print("="*60); print("FoodShock V4"); print("="*60)
    mr=pd.read_csv(DATA/"foodshock_asean_model_ready.csv"); mr["date"]=pd.to_datetime(mr["date"])
    fx=build_implied_fx(mr); df=enrich_panel(mr, fx)
    for c in FEATS:
        if c not in df.columns: df[c]=0.0
        df[c]=df[c].replace([np.inf,-np.inf],np.nan)
    df=df.dropna(subset=BASE+["target_price_usd_t1","residual_from_yt","shock_label","usdprice"]).copy()
    for c in MACRO: df[c]=df[c].fillna(0.0)
    df=df.sort_values(["date","countryiso3","commodity_group"]).reset_index(drop=True)
    df.to_csv(DATA/"foodshock_asean_model_ready.csv", index=False)
    print(f"panel {df.shape} shock_up={df['shock_label'].mean():.3f}")

    folds=walk_forward_folds(df)
    print("folds", len(folds))
    names=["LightGBM","XGBoost","CatBoost","HistGradientBoosting","RandomForest"]
    oof={n:{"y":[],"pred":[],"pers":[],"country":[],"ym":[]} for n in names}
    # quantiles OOS collectors
    q_lo_all, q_hi_all, q_y_all = [], [], []
    # shock collectors
    sy_all, sp_all, mom_all = [], [], []
    pred_f1_all, pred_ops_all = [], []
    thr_f1_list, thr_ops_list, thr_rule_list = [], [], []
    widths = []

    for tr_m, te_m, fname in folds:
        tr_df = df.loc[tr_m].copy(); te_df = df.loc[te_m].copy()
        enc = fit_encoders(tr_df)
        X_tr = transform_X(tr_df, enc); y_tr = tr_df["residual_from_yt"]
        X_te = transform_X(te_df, enc)
        y_abs = te_df["target_price_usd_t1"].values; y_t = te_df["usdprice"].values

        for name in names:
            m = make_reg(name); m.fit(X_tr, y_tr)
            yhat = y_t + m.predict(X_te)
            oof[name]["y"].append(y_abs); oof[name]["pred"].append(yhat)
            oof[name]["pers"].append(y_t); oof[name]["country"].append(te_df["countryiso3"].values)
            oof[name]["ym"].append(te_df["year_month"].astype(str).values)

        # --- conformalized quantiles with month-block cal ---
        tr_in, va_in = month_block_split(tr_df, val_ratio=0.2)
        enc_q = fit_encoders(tr_in)
        X_tri, y_tri = transform_X(tr_in, enc_q), tr_in["residual_from_yt"]
        X_vai, y_vai = transform_X(va_in, enc_q), va_in["residual_from_yt"]
        q10 = lgb.LGBMRegressor(objective="quantile", alpha=0.1, n_estimators=100, max_depth=4, learning_rate=0.05, random_state=RS, n_jobs=-1, verbose=-1)
        q90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, n_estimators=100, max_depth=4, learning_rate=0.05, random_state=RS, n_jobs=-1, verbose=-1)
        q10.fit(X_tri, y_tri); q90.fit(X_tri, y_tri)
        # conformal scores on validation residuals
        lo_va = q10.predict(X_vai); hi_va = q90.predict(X_vai)
        # ensure order
        lo_va2 = np.minimum(lo_va, hi_va); hi_va2 = np.maximum(lo_va, hi_va)
        # absolute residual conformity for interval expansion
        # scores: how much y exceeds bounds
        s_lo = lo_va2 - y_vai.values  # positive if y below lo
        s_hi = y_vai.values - hi_va2
        s = np.maximum(s_lo, s_hi)
        qcal = float(np.quantile(s, 0.8)) if len(s)>10 else 0.0  # ~80% target on cal
        # refit quantiles on full train fold for test
        q10.fit(X_tr, y_tr); q90.fit(X_tr, y_tr)
        lo_te = np.minimum(q10.predict(X_te), q90.predict(X_te)) - qcal
        hi_te = np.maximum(q10.predict(X_te), q90.predict(X_te)) + qcal
        # reconstruct absolute prices
        lo_abs = y_t + lo_te; hi_abs = y_t + hi_te
        q_lo_all.append(lo_abs); q_hi_all.append(hi_abs); q_y_all.append(y_abs)
        widths.append(float(np.mean(hi_abs-lo_abs)))

        # --- Stage-2 shock, month-block threshold ---
        y_sh_tr = tr_df["shock_label"]
        tr_c, va_c = month_block_split(tr_df, val_ratio=0.2)
        enc_c = fit_encoders(tr_c)
        clf = make_clf()
        clf.fit(transform_X(tr_c, enc_c), tr_c["shock_label"])
        p_va = clf.predict_proba(transform_X(va_c, enc_c))[:,1]
        thr_f1, thr_ops, thr_rule = select_thresholds(va_c["shock_label"].values, p_va)
        thr_f1_list.append(thr_f1); thr_ops_list.append(thr_ops); thr_rule_list.append(thr_rule)
        # refit on full train fold
        clf.fit(X_tr.assign(**{c: tr_df[c].values for c in []}) if False else transform_X(tr_df, fit_encoders(tr_df)), y_sh_tr)
        # simpler: use enc from full tr
        enc_full = fit_encoders(tr_df)
        clf.fit(transform_X(tr_df, enc_full), y_sh_tr)
        p_te = clf.predict_proba(transform_X(te_df, enc_full))[:,1]
        sy_all.append(te_df["shock_label"].values); sp_all.append(p_te)
        mom_all.append(te_df["pct_change_1m"].fillna(0).values)
        pred_f1_all.append((p_te>=thr_f1).astype(int)); pred_ops_all.append((p_te>=thr_ops).astype(int))
        print(f"{fname}: thr_f1={thr_f1:.3f} thr_ops={thr_ops:.3f} rule={thr_rule} qcal={qcal:.4f} width={widths[-1]:.3f}")

    # aggregate regression metrics
    rows=[]
    y0=np.concatenate(oof["LightGBM"]["y"]); p0=np.concatenate(oof["LightGBM"]["pers"]); ym0=np.concatenate(oof["LightGBM"]["ym"])
    rows.append({"model":"Persistence (y_t)","MAE_micro":mean_absolute_error(y0,p0),"RMSE_micro":float(np.sqrt(mean_squared_error(y0,p0))),"sMAPE_micro":smape(y0,p0),"R2_micro":r2_score(y0,p0),"MAE_macro":np.nan,"Persistence_MAE":mean_absolute_error(y0,p0),"Persistence_RMSE":float(np.sqrt(mean_squared_error(y0,p0))),"SS_RMSE":0.0,"DM_stat":np.nan,"DM_pvalue":np.nan})
    for name in names:
        y=np.concatenate(oof[name]["y"]); pred=np.concatenate(oof[name]["pred"]); pers=np.concatenate(oof[name]["pers"])
        countries=np.concatenate(oof[name]["country"]); ym=np.concatenate(oof[name]["ym"])
        mae=mean_absolute_error(y,pred); rmse=float(np.sqrt(mean_squared_error(y,pred)))
        rmse_p=float(np.sqrt(mean_squared_error(y,pers))); ss=1-rmse/rmse_p if rmse_p>0 else np.nan
        dm_s,dm_p=diebold_mariano_panel(y-pred, y-pers, ym)
        macro=[mean_absolute_error(y[countries==c], pred[countries==c]) for c in np.unique(countries) if (countries==c).sum()>5]
        rows.append({"model":f"{name}_WF_Residual","MAE_micro":mae,"RMSE_micro":rmse,"sMAPE_micro":smape(y,pred),"R2_micro":r2_score(y,pred),"MAE_macro":float(np.mean(macro)) if macro else np.nan,"Persistence_MAE":mean_absolute_error(y,pers),"Persistence_RMSE":rmse_p,"SS_RMSE":ss,"DM_stat":dm_s,"DM_pvalue":dm_p})
        print(f"{name}: MAE={mae:.4f} RMSE={rmse:.4f} SS={ss:.4f} DM_p={dm_p}")
    results=pd.DataFrame(rows).sort_values("RMSE_micro"); results.to_csv(DATA/"model_evaluation_results.csv", index=False)
    print(results.to_string(index=False))
    best=results[~results["model"].str.contains("Persistence")].iloc[0]["model"].replace("_WF_Residual","")
    print("best", best)

    # stratified
    meta_parts=[]
    for tr_m, te_m, _ in folds:
        meta_parts.append(df.loc[te_m, ["countryiso3","commodity_group","pricetype"]].copy())
    meta=pd.concat(meta_parts, ignore_index=True)
    meta["y_true"]=np.concatenate(oof[best]["y"]); meta["y_pred"]=np.concatenate(oof[best]["pred"])
    def ev(g):
        return pd.Series({"MAE":mean_absolute_error(g["y_true"],g["y_pred"]),"RMSE":float(np.sqrt(mean_squared_error(g["y_true"],g["y_pred"]))),"sMAPE":smape(g["y_true"],g["y_pred"]),"R2":r2_score(g["y_true"],g["y_pred"]) if len(g)>2 else np.nan,"n":len(g)})
    meta.groupby("commodity_group").apply(ev).reset_index().to_csv(DATA/"model_eval_by_commodity.csv", index=False)
    meta.groupby("countryiso3").apply(ev).reset_index().to_csv(DATA/"model_eval_by_country.csv", index=False)
    meta.groupby("pricetype").apply(ev).reset_index().to_csv(DATA/"model_eval_by_pricetype.csv", index=False)

    # PICP overall
    yq=np.concatenate(q_y_all); lo=np.concatenate(q_lo_all); hi=np.concatenate(q_hi_all)
    picp=float(np.mean((yq>=lo)&(yq<=hi))); mean_width=float(np.mean(hi-lo))
    print(f"Overall PICP={picp:.3f} MeanWidth={mean_width:.4f}")

    # shock metrics
    sy=np.concatenate(sy_all); sp=np.concatenate(sp_all); mom=np.concatenate(mom_all)
    pred_f1=np.concatenate(pred_f1_all); pred_ops=np.concatenate(pred_ops_all)
    mom_pred=(mom>SHOCK_PCT).astype(int)

    def cls(y, pred, score):
        from sklearn.metrics import confusion_matrix as _cm
        prec = float(precision_score(y, pred, zero_division=0))
        cm = _cm(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fdr = float(1.0 - prec)  # FP/(TP+FP)
        return {"Precision": prec, "Recall": float(recall_score(y, pred, zero_division=0)),
                "F1": float(f1_score(y, pred, zero_division=0)),
                "ROC_AUC": float(roc_auc_score(y, score)), "PR_AUC": float(average_precision_score(y, score)),
                "FDR": fdr, "FPR": fpr}
    m_f1=cls(sy, pred_f1, sp); m_ops=cls(sy, pred_ops, sp); m_mom=cls(sy, mom_pred, mom)
    thr_ops_mean=float(np.mean(thr_ops_list)); thr_f1_mean=float(np.mean(thr_f1_list))
    shock_df=pd.DataFrame([
        {"method":"Stage2_F1_Optimal","Mean_Train_Threshold":thr_f1_mean,**m_f1,"Base_Rate":float(sy.mean()),"Support_Shock":int(sy.sum()),"Support_Total":int(len(sy)),"Overall_PICP_P10_P90":picp,"Mean_Interval_Width":mean_width,"Shock_Definition":"target_pct_change_t1 > +0.05"},
        {"method":"Stage2_Operational_LowFAR","Mean_Train_Threshold":thr_ops_mean,**m_ops,"Base_Rate":float(sy.mean()),"Support_Shock":int(sy.sum()),"Support_Total":int(len(sy)),"Overall_PICP_P10_P90":picp,"Mean_Interval_Width":mean_width,"Shock_Definition":"target_pct_change_t1 > +0.05"},
        {"method":"Baseline_Momentum_pct_change_1m","Mean_Train_Threshold":SHOCK_PCT,**m_mom,"Base_Rate":float(sy.mean()),"Support_Shock":int(sy.sum()),"Support_Total":int(len(sy)),"Overall_PICP_P10_P90":picp,"Mean_Interval_Width":mean_width,"Shock_Definition":"target_pct_change_t1 > +0.05"},
    ])
    shock_df.to_csv(DATA/"model_eval_shock_classification.csv", index=False)
    print(shock_df.to_string(index=False))
    print("CM ops:\n", confusion_matrix(sy, pred_ops))

    # app uses operational threshold
    n_floor = sum(1 for r in thr_rule_list if r == "precision_floor")
    n_f05 = sum(1 for r in thr_rule_list if r == "f0.5_fallback")
    print(f"Threshold rules: precision_floor={n_floor} folds, f0.5_fallback={n_f05} folds")
    Path(DATA/"app_shock_threshold.txt").write_text(f"{thr_ops_mean:.4f}\n")
    with open(DATA/"shock_threshold_justification.txt","w") as f:
        f.write(
            "Shock: upward only target_pct_change_t1 > +0.05 (SDG 2.c).\n"
            "Threshold: chosen inside each train fold on month-block validation (no row-wise month split).\n"
            f"F1-optimal mean thr={thr_f1_mean:.3f}; Operational (prec>={MIN_PREC_OPS} or F0.5) mean thr={thr_ops_mean:.3f}.\n"
            f"Threshold folds: precision_floor={n_floor} f0.5_fallback={n_f05}\n"
            f"Ops metrics: P={m_ops['Precision']:.3f} R={m_ops['Recall']:.3f} F1={m_ops['F1']:.3f} FDR={m_ops.get('FDR', float('nan')):.3f} FPR={m_ops.get('FPR', float('nan')):.3f}\n"
            f"F1 metrics: P={m_f1['Precision']:.3f} R={m_f1['Recall']:.3f} F1={m_f1['F1']:.3f}\n"
            f"Momentum: P={m_mom['Precision']:.3f} R={m_mom['Recall']:.3f} PR-AUC={m_mom['PR_AUC']:.3f} ROC={m_mom['ROC_AUC']:.3f}\n"
            f"Conformalized quantile overall PICP={picp:.3f} mean width={mean_width:.4f}\n"
            "FX features: from WFP price/usdprice when raw present; else precomputed columns in model-ready CSV.\n"
        )

    # fit best model for SHAP on last fold train
    tr_m, te_m, _ = folds[-1]
    tr_df=df.loc[tr_m]; enc=fit_encoders(tr_df)
    X_tr=transform_X(tr_df, enc); y_tr=tr_df["residual_from_yt"]
    best_obj=make_reg(best); best_obj.fit(X_tr, y_tr)
    oof_best={"y":np.concatenate(oof[best]["y"]),"pred":np.concatenate(oof[best]["pred"])}
    regenerate_figures(results, oof_best, meta, {"proba":sp,"y":sy,"thr":thr_ops_mean}, f"{best}_WF_Residual", best_obj, X_tr)
    print("V4 complete")

if __name__ == "__main__":
    main()
