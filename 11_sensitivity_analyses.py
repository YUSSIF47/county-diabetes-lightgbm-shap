"""
11_sensitivity_analyses.py
==========================
Step 11: Run all 8 sensitivity analyses and compile results.

S1: Structural predictors only
S2: Health-behaviour covariates only
S3: Observed variables only (no imputed/PCA)
S4: Alternative FARA aggregation definitions
S5: Food Environment Composite Index excluded
S6: Rurality excluded
S7: Census region fixed effects added
S8: Reduced feature model (training-set SHAP rankings)

Input:  data/processed/train.csv, val.csv, test.csv
        models/best tree model
Output: outputs/tables/sensitivity_results.csv
        outputs/figures/fig_sensitivity_comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
import joblib, json, os, copy

PROCESSED = "data/processed"
TABLES = "outputs/tables"
FIGURES = "outputs/figures"

# ── Load data and best model ──────────────────────────────────
with open(f"{PROCESSED}/best_tree_model.txt") as f:
    best_model_name = f.read().strip()

model_file = {
    "XGBoost": "models/xgboost.pkl",
    "LightGBM": "models/lightgbm.pkl",
    "Random Forest": "models/random_forest.pkl",
}[best_model_name]

train = pd.read_csv(f"{PROCESSED}/train.csv", dtype={"fips": str})
val   = pd.read_csv(f"{PROCESSED}/val.csv",   dtype={"fips": str})
test  = pd.read_csv(f"{PROCESSED}/test.csv",  dtype={"fips": str})

with open(f"{PROCESSED}/feature_list.txt") as f:
    ALL_FEATURES = [line.strip() for line in f if line.strip()]

with open(f"{PROCESSED}/feature_groups.json") as f:
    feature_groups = json.load(f)

OUTCOME = "diabetes_prev"
HEALTH_BEHAV = ["obesity_prev", "physical_inactivity_prev",
                "hypertension_prev", "smoking_prev"]
HEALTH_BEHAV = [f for f in HEALTH_BEHAV if f in train.columns]

STRUCTURAL = feature_groups.get("structural_only",
    [f for f in ALL_FEATURES if f not in HEALTH_BEHAV])

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100

def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mp   = mape(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {label}: RMSE={rmse:.4f} MAE={mae:.4f} MAPE={mp:.2f}% R2={r2:.4f}")
    return {"sensitivity_model": label, "RMSE": round(rmse,4),
            "MAE": round(mae,4), "MAPE": round(mp,2), "R2": round(r2,4)}

def retrain_best(X_tr, y_tr, X_va):
    """Retrain the best model architecture on new feature set."""
    m = joblib.load(model_file)
    m.fit(X_tr, y_tr)
    return m.predict(X_va)

results = []

# ── Main model baseline ───────────────────────────────────────
print("Main model (all features):")
main_model = joblib.load(model_file)
y_pred_main = main_model.predict(test[ALL_FEATURES].values)
results.append(evaluate(test[OUTCOME].values, y_pred_main, "Main model (all features)"))

# ── S1: Structural predictors only ───────────────────────────
print("\nS1: Structural predictors only")
s1_feats = [f for f in STRUCTURAL if f in train.columns]
y_pred_s1 = retrain_best(train[s1_feats].values, train[OUTCOME].values,
                          test[s1_feats].values)
results.append(evaluate(test[OUTCOME].values, y_pred_s1, "S1: Structural only"))

# ── S2: Health-behaviour covariates only ─────────────────────
print("\nS2: Health-behaviour only")
s2_feats = [f for f in HEALTH_BEHAV if f in train.columns]
y_pred_s2 = retrain_best(train[s2_feats].values, train[OUTCOME].values,
                          test[s2_feats].values)
results.append(evaluate(test[OUTCOME].values, y_pred_s2, "S2: Health-behaviour only"))

# ── S3: Observed variables only (no imputed, no PCA) ─────────
print("\nS3: Observed variables only (exclude feci, fdi_binary_high)")
exclude_constructed = ["feci", "fdi_binary_high", "rural_flag"]
s3_feats = [f for f in ALL_FEATURES if f not in exclude_constructed
            and f in train.columns]
y_pred_s3 = retrain_best(train[s3_feats].dropna(subset=s3_feats).index
                          .pipe(lambda idx: train.loc[idx, s3_feats].values),
                          train.loc[train[s3_feats].dropna().index, OUTCOME].values,
                          test[s3_feats].fillna(test[s3_feats].median()).values)
results.append(evaluate(test[OUTCOME].values, y_pred_s3, "S3: Observed variables only"))

# ── S4: Alternative FARA definitions ─────────────────────────
print("\nS4: Alternative FARA aggregation definitions")
for alt_col, label in [("fdi_alt_1mi",   "S4a: FDI 1-mile threshold"),
                        ("fdi_alt_10mi",  "S4b: FDI 10-mile (rural)"),
                        ("fdi_binary_high","S4c: Binary FDI (>=20%)")]:
    if alt_col in train.columns:
        s4_feats = [f if f != "food_desert_index" else alt_col
                    for f in ALL_FEATURES if f in train.columns]
        s4_feats = list(dict.fromkeys(s4_feats))  # deduplicate
        y_pred_s4 = retrain_best(
            train[s4_feats].fillna(train[s4_feats].median()).values,
            train[OUTCOME].values,
            test[s4_feats].fillna(test[s4_feats].median()).values
        )
        results.append(evaluate(test[OUTCOME].values, y_pred_s4, label))
    else:
        print(f"  {label}: column {alt_col} not found — skipped")

# ── S5: Food Environment Composite Index excluded ─────────────
print("\nS5: FECI excluded")
s5_feats = [f for f in ALL_FEATURES if f != "feci" and f in train.columns]
y_pred_s5 = retrain_best(train[s5_feats].values, train[OUTCOME].values,
                          test[s5_feats].values)
results.append(evaluate(test[OUTCOME].values, y_pred_s5, "S5: FECI excluded"))

# ── S6: Rurality excluded ─────────────────────────────────────
print("\nS6: Rurality excluded")
s6_feats = [f for f in ALL_FEATURES
            if f not in ["rucc_code", "rural_flag"] and f in train.columns]
y_pred_s6 = retrain_best(train[s6_feats].values, train[OUTCOME].values,
                          test[s6_feats].values)
results.append(evaluate(test[OUTCOME].values, y_pred_s6, "S6: Rurality excluded"))

# ── S7: Census region fixed effects ──────────────────────────
print("\nS7: Census region fixed effects")
region_dummies_train = pd.get_dummies(train["census_region"], prefix="region",
                                       drop_first=True)
region_dummies_test  = pd.get_dummies(test["census_region"],  prefix="region",
                                       drop_first=True)
# Align columns
for col in region_dummies_train.columns:
    if col not in region_dummies_test.columns:
        region_dummies_test[col] = 0

s7_train = pd.concat([train[ALL_FEATURES], region_dummies_train], axis=1)
s7_test  = pd.concat([test[ALL_FEATURES],  region_dummies_test[region_dummies_train.columns]], axis=1)
s7_feats = s7_train.columns.tolist()
y_pred_s7 = retrain_best(s7_train.values, train[OUTCOME].values, s7_test.values)
results.append(evaluate(test[OUTCOME].values, y_pred_s7, "S7: Region fixed effects"))

# ── S8: Reduced feature model (training-set SHAP only) ───────
print("\nS8: Reduced feature model")
shap_importance = pd.read_csv(f"{TABLES}/shap_global_importance.csv")
# Top 10 features by mean |SHAP| from training-set SHAP
top_feats = shap_importance.head(10)["feature"].tolist()
top_feats = [f for f in top_feats if f in train.columns]
print(f"  Top 10 training-set SHAP features: {top_feats}")
y_pred_s8 = retrain_best(train[top_feats].values, train[OUTCOME].values,
                          test[top_feats].values)
results.append(evaluate(test[OUTCOME].values, y_pred_s8, "S8: Reduced (SHAP top 10)"))

# ── Save results ──────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(f"{TABLES}/sensitivity_results.csv", index=False)
print(f"\nAll sensitivity results saved.")
print(results_df.to_string(index=False))

# ── Summary figure ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
model_labels = results_df["sensitivity_model"].tolist()
short_labels = [m.split(":")[0].strip() for m in model_labels]
rmse_vals = results_df["RMSE"].tolist()
r2_vals   = results_df["R2"].tolist()
colors = ["#2c3e50"] + ["#7f8c8d"] * (len(model_labels)-1)

axes[0].barh(short_labels[::-1], rmse_vals[::-1], color=colors[::-1])
axes[0].set_xlabel("RMSE")
axes[0].set_title("Sensitivity: RMSE\n(lower = better)")
axes[0].axvline(rmse_vals[0], color="#e74c3c", linestyle="--",
                linewidth=1.2, label="Main model")
axes[0].legend()

axes[1].barh(short_labels[::-1], r2_vals[::-1], color=colors[::-1])
axes[1].set_xlabel("R²")
axes[1].set_title("Sensitivity: R²\n(higher = better)")
axes[1].axvline(r2_vals[0], color="#e74c3c", linestyle="--",
                linewidth=1.2, label="Main model")
axes[1].legend()

plt.suptitle(f"Sensitivity Analysis Results — {best_model_name}", fontsize=12)
plt.tight_layout()
plt.savefig(f"{FIGURES}/fig_sensitivity_comparison.png", dpi=150)
plt.close()
print("Sensitivity figure saved.")
