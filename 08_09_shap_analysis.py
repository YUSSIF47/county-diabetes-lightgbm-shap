"""
08_09_shap_analysis.py
======================
Runs SHAP TreeExplainer for the validation-selected LightGBM model using all
2,957 analytic counties. The model itself is the training-partition-fitted model;
it is not refitted on the full analytic sample before SHAP computation.
"""
import os
import joblib
import numpy as np
import pandas as pd
import shap

PROCESSED = "data/processed"
TABLES = "outputs/tables"
os.makedirs(TABLES, exist_ok=True)

with open(f"{PROCESSED}/feature_list_primary_22.txt") as f:
    FEATURES = [line.strip() for line in f if line.strip()]

OUTCOME = "diabetes_prev"
train = pd.read_csv(f"{PROCESSED}/train.csv", dtype={"fips": str})
val = pd.read_csv(f"{PROCESSED}/val.csv", dtype={"fips": str})
test = pd.read_csv(f"{PROCESSED}/test.csv", dtype={"fips": str})
all_df = pd.concat([train, val, test], ignore_index=True)

model = joblib.load("models/lightgbm.pkl")
X_all = all_df[FEATURES]
y_all = all_df[OUTCOME].values

print(f"Running SHAP TreeExplainer for {len(all_df)} analytic counties...")
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(X_all)

shap_df = pd.DataFrame(shap_values, columns=FEATURES)
shap_df["fips"] = all_df["fips"].values
shap_df["census_region"] = all_df["census_region"].values
shap_df["y_true"] = y_all
shap_df["y_pred"] = model.predict(X_all)
shap_df.to_csv(f"{TABLES}/shap_county_values.csv", index=False)

mean_abs_shap = (
    pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
mean_abs_shap.to_csv(f"{TABLES}/shap_global_importance.csv", index=False)

structural_primary = [
    "food_desert_index", "fastfood_density", "grocery_density", "snap_rate",
    "food_insecurity_rate", "feci", "poverty_rate", "median_income",
    "unemployment_rate", "pct_agri_sector", "pct_mfg_sector",
    "pct_foodsvc_sector", "rucc_code"
]
struct_cols = [f for f in structural_primary if f in FEATURES]
pos_shap = shap_df[struct_cols].clip(lower=0)
shap_df["dominant"] = pos_shap.idxmax(axis=1)
shap_df.loc[pos_shap.max(axis=1) == 0, "dominant"] = "other"

dominant_df = shap_df[["fips", "census_region", "dominant"]].copy()
dominant_df.to_csv(f"{PROCESSED}/dominant_contributors_lgb.csv", index=False)

dominant_summary = (
    dominant_df.groupby(["dominant", "census_region"])
    .size()
    .reset_index(name="n_counties")
)
dominant_summary.to_csv(f"{TABLES}/dominant_contributor_summary.csv", index=False)

regional_signed = shap_df.groupby("census_region")[FEATURES].mean().round(4)
regional_signed.to_csv(f"{TABLES}/shap_regional_signed_means.csv")

print("SHAP analysis complete.")
print(f"Saved: {TABLES}/shap_county_values.csv")
print(f"Saved: {TABLES}/shap_global_importance.csv")
print(f"Saved: {PROCESSED}/dominant_contributors_lgb.csv")
