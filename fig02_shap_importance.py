"""
fig02_shap_importance.py
==================
Fig 2 — corrected to show exactly 22 primary predictors.
Removes rural_flag and fdi_binary_high (sensitivity-only variables).
"""

import pandas as pd, numpy as np, matplotlib, os, joblib, shap
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.labelsize":  12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
})

PROCESSED = "data/processed"
FIGURES   = "outputs/figures/final"
os.makedirs(FIGURES, exist_ok=True)

# 22 primary predictors only — excludes rural_flag and fdi_binary_high
PRIMARY_22 = [
    "physical_inactivity_prev","pct_white","poverty_rate","pct_black",
    "smoking_prev","snap_rate","median_income","food_insecurity_rate",
    "pct_age65plus","obesity_prev","pct_hispanic","hypertension_prev",
    "pct_college_edu","unemployment_rate","food_desert_index","feci",
    "pct_agri_sector","fastfood_density","pct_mfg_sector",
    "grocery_density","pct_foodsvc_sector","rucc_code"
]

LABELS = {
    "physical_inactivity_prev": "Physical inactivity (%)",
    "pct_white":                "White population (%)",
    "poverty_rate":             "Poverty rate (%)",
    "pct_black":                "Black population (%)",
    "smoking_prev":             "Smoking prevalence (%)",
    "snap_rate":                "SNAP participation rate (%)",
    "median_income":            "Median household income",
    "food_insecurity_rate":     "Food insecurity rate (%)",
    "pct_age65plus":            "Population aged 65 years or older (%)",
    "obesity_prev":             "Obesity prevalence (%)",
    "pct_hispanic":             "Hispanic population (%)",
    "hypertension_prev":        "Hypertension prevalence (%)",
    "pct_college_edu":          "College educated (%)",
    "unemployment_rate":        "Unemployment rate (%)",
    "food_desert_index":        "Food desert index (%)",
    "feci":                     "Food Environment Composite Index",
    "pct_agri_sector":          "Agriculture sector (%)",
    "fastfood_density":         "Fast-food density (per 1,000)",
    "pct_mfg_sector":           "Manufacturing sector (%)",
    "grocery_density":          "Grocery store density (per 1,000)",
    "pct_foodsvc_sector":       "Food service sector (%)",
    "rucc_code":                "Rural-Urban Continuum Code (RUCC)",
}

structural = [
    "food_desert_index","fastfood_density","grocery_density","snap_rate",
    "food_insecurity_rate","feci","poverty_rate","median_income",
    "unemployment_rate","pct_agri_sector","pct_mfg_sector",
    "pct_foodsvc_sector","rucc_code",
]

print("Loading data and computing SHAP...")
with open(f"{PROCESSED}/feature_list_primary_22.txt") as f:
    FEATURES = [l.strip() for l in f if l.strip()]
OUTCOME = "diabetes_prev"
train = pd.read_csv(f"{PROCESSED}/train.csv", dtype={"fips":str})
val   = pd.read_csv(f"{PROCESSED}/val.csv",   dtype={"fips":str})
test  = pd.read_csv(f"{PROCESSED}/test.csv",  dtype={"fips":str})
all_df = pd.concat([train,val,test], ignore_index=True)

model     = joblib.load("models/lightgbm.pkl")
explainer = shap.TreeExplainer(model,
                feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(all_df[FEATURES])

# Build mean abs SHAP for primary 22 only
mean_abs = pd.DataFrame({
    "feature": FEATURES,
    "v": np.abs(shap_values).mean(axis=0)
})
# Keep only primary 22
mean_abs = mean_abs[mean_abs["feature"].isin(PRIMARY_22)]
mean_abs = mean_abs.sort_values("v", ascending=False).reset_index(drop=True)

print(f"Predictors in figure: {len(mean_abs)}")
for _, row in mean_abs.iterrows():
    print(f"  {row['feature']}: {row['v']:.4f}")

colors  = ["#c0392b" if f in structural else "#2980b9"
           for f in mean_abs["feature"]]
ylabels = [LABELS.get(f, f) for f in mean_abs["feature"]]

fig, ax = plt.subplots(figsize=(13, 11))
ax.barh(ylabels[::-1], mean_abs["v"][::-1],
        color=colors[::-1], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Mean absolute SHAP value (percentage points)", fontsize=12)
ax.set_title("")
ax.legend(handles=[
    mpatches.Patch(facecolor="#c0392b", label="Structural predictors"),
    mpatches.Patch(facecolor="#2980b9",
                   label="Health-behavior, clinical, and demographic predictors"),
], loc="lower right", fontsize=11, framealpha=0.9)
ax.grid(axis="x", alpha=0.3, linewidth=0.7)
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=11)
plt.tight_layout()

fig.savefig(f"{FIGURES}/Fig2.tif", dpi=300, bbox_inches="tight",
            facecolor="white", pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(f"{FIGURES}/Fig2.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(f"{FIGURES}/Fig2.pdf", bbox_inches="tight", facecolor="white")
print("\nSaved Fig2 [TIF | PNG | PDF]")
plt.close(fig)
print("Fig 2 corrected — exactly 22 primary predictors.")
