"""
shap_agreement_lightgbm_xgboost.py
===========================
1. SHAP agreement analysis: LightGBM vs XGBoost
   - Spearman correlation between global mean absolute SHAP rankings
   - Top-5 predictor agreement
   - County-level dominant contributor agreement
   - Pearson correlation between county-level predictions

2. Bootstrap stability analysis: 500 iterations
   - Percentage of counties retaining same dominant contributor
"""

import pandas as pd
import numpy as np
import joblib, shap
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error

np.random.seed(42)

PROCESSED = "data/processed"
TABLES    = "outputs/tables"

with open(f"{PROCESSED}/feature_list_primary_22.txt") as f:
    FEATURES = [l.strip() for l in f if l.strip()]

OUTCOME = "diabetes_prev"
train = pd.read_csv(f"{PROCESSED}/train.csv", dtype={"fips": str})
val   = pd.read_csv(f"{PROCESSED}/val.csv",   dtype={"fips": str})
test  = pd.read_csv(f"{PROCESSED}/test.csv",  dtype={"fips": str})
all_df = pd.concat([train, val, test], ignore_index=True)

lgb_model = joblib.load("models/lightgbm.pkl")
xgb_model = joblib.load("models/xgboost.pkl")

structural_primary = [
    "food_desert_index","fastfood_density","grocery_density","snap_rate",
    "food_insecurity_rate","feci","poverty_rate","median_income",
    "unemployment_rate","pct_agri_sector","pct_mfg_sector",
    "pct_foodsvc_sector","rucc_code"
]
struct_cols = [f for f in structural_primary if f in FEATURES]

# ── SHAP values for both models ───────────────────────────────
print("Computing LightGBM SHAP values...")
lgb_explainer  = shap.TreeExplainer(lgb_model,
                     feature_perturbation="tree_path_dependent")
lgb_shap       = lgb_explainer.shap_values(all_df[FEATURES])

print("Computing XGBoost SHAP values...")
xgb_explainer  = shap.TreeExplainer(xgb_model,
                     feature_perturbation="tree_path_dependent")
xgb_shap       = xgb_explainer.shap_values(all_df[FEATURES])

lgb_shap_df = pd.DataFrame(lgb_shap, columns=FEATURES)
xgb_shap_df = pd.DataFrame(xgb_shap, columns=FEATURES)

# ── 1. Global mean absolute SHAP rankings ────────────────────
print("\n" + "="*60)
print("1. GLOBAL SHAP AGREEMENT — LightGBM vs XGBoost")
print("="*60)

lgb_mean_abs = np.abs(lgb_shap).mean(axis=0)
xgb_mean_abs = np.abs(xgb_shap).mean(axis=0)

lgb_rank = pd.Series(lgb_mean_abs, index=FEATURES).rank(ascending=False)
xgb_rank = pd.Series(xgb_mean_abs, index=FEATURES).rank(ascending=False)

spearman_r, spearman_p = spearmanr(lgb_rank, xgb_rank)
print(f"Spearman rank correlation: r = {spearman_r:.4f}, p = {spearman_p:.4f}")

lgb_top5 = set(pd.Series(lgb_mean_abs, index=FEATURES).nlargest(5).index)
xgb_top5 = set(pd.Series(xgb_mean_abs, index=FEATURES).nlargest(5).index)
top5_agreement = len(lgb_top5 & xgb_top5)
print(f"Top-5 predictor agreement: {top5_agreement}/5")
print(f"  LightGBM top 5: {sorted(lgb_top5)}")
print(f"  XGBoost top 5:  {sorted(xgb_top5)}")

# ── 2. County-level dominant contributor agreement ───────────
print("\n" + "="*60)
print("2. DOMINANT CONTRIBUTOR AGREEMENT")
print("="*60)

lgb_pos = lgb_shap_df[struct_cols].clip(lower=0)
lgb_dominant = lgb_pos.idxmax(axis=1)
lgb_dominant[lgb_pos.max(axis=1)==0] = "other"

xgb_pos = xgb_shap_df[struct_cols].clip(lower=0)
xgb_dominant = xgb_pos.idxmax(axis=1)
xgb_dominant[xgb_pos.max(axis=1)==0] = "other"

n_agree = (lgb_dominant == xgb_dominant).sum()
pct_agree = n_agree / len(lgb_dominant) * 100
print(f"Counties with same dominant contributor: {n_agree}/{len(lgb_dominant)} ({pct_agree:.1f}%)")

# ── 3. County-level prediction correlation ───────────────────
print("\n" + "="*60)
print("3. PREDICTION CORRELATION")
print("="*60)

lgb_preds = lgb_model.predict(all_df[FEATURES])
xgb_preds = xgb_model.predict(all_df[FEATURES])
pearson_r, pearson_p = pearsonr(lgb_preds, xgb_preds)
print(f"Pearson correlation between predictions: r = {pearson_r:.4f}, p = {pearson_p:.6f}")
pred_diff_rmse = np.sqrt(mean_squared_error(lgb_preds, xgb_preds))
print(f"RMSE between LightGBM and XGBoost predictions: {pred_diff_rmse:.4f}")

# ── 4. Bootstrap stability analysis ──────────────────────────
print("\n" + "="*60)
print("4. BOOTSTRAP STABILITY ANALYSIS (500 iterations)")
print("="*60)

# Reference dominant contributors computed directly from LightGBM SHAP values
ref_dominant = lgb_dominant.values

n_boot      = 500
stability   = []
rng         = np.random.default_rng(42)

X_all = all_df[FEATURES].values
print(f"Running {n_boot} bootstrap iterations...")
for i in range(n_boot):
    idx = rng.choice(len(X_all), size=len(X_all), replace=True)
    shap_boot = lgb_explainer.shap_values(X_all[idx])
    shap_boot_df = pd.DataFrame(shap_boot, columns=FEATURES)
    pos_boot = shap_boot_df[struct_cols].clip(lower=0)
    dom_boot = pos_boot.idxmax(axis=1)
    dom_boot[pos_boot.max(axis=1)==0] = "other"
    pct_stable = (dom_boot.values == ref_dominant[idx]).mean() * 100
    stability.append(pct_stable)
    if (i+1) % 100 == 0:
        print(f"  Completed {i+1}/{n_boot} iterations...")

stability = np.array(stability)
print(f"\nBootstrap stability results:")
print(f"  Mean % counties retaining same dominant contributor: {stability.mean():.1f}%")
print(f"  95% CI: [{np.percentile(stability, 2.5):.1f}%, {np.percentile(stability, 97.5):.1f}%]")
print(f"  Min: {stability.min():.1f}%  Max: {stability.max():.1f}%")

# ── Save results ──────────────────────────────────────────────
results = {
    "spearman_r": round(spearman_r, 4),
    "spearman_p": round(spearman_p, 4),
    "top5_agreement": f"{top5_agreement}/5",
    "dominant_contributor_agreement_pct": round(pct_agree, 1),
    "prediction_pearson_r": round(pearson_r, 4),
    "prediction_rmse_diff": round(pred_diff_rmse, 4),
    "bootstrap_mean_stability_pct": round(stability.mean(), 1),
    "bootstrap_ci_lower": round(np.percentile(stability, 2.5), 1),
    "bootstrap_ci_upper": round(np.percentile(stability, 97.5), 1),
}
pd.DataFrame([results]).to_csv(
    f"{TABLES}/shap_stability_agreement.csv", index=False)

# Save LightGBM dominant contributors
lgb_dom_df = all_df[["fips"]].copy()
lgb_dom_df["dominant"] = lgb_dominant.values
lgb_dom_df["census_region"] = all_df["census_region"].values
lgb_dom_df.to_csv(
    f"{PROCESSED}/dominant_contributors_lgb.csv", index=False)

print("\n" + "="*60)
print("All results saved.")
print(f"  {TABLES}/shap_stability_agreement.csv")
print(f"  {PROCESSED}/dominant_contributors_lgb.csv")
print("="*60)
