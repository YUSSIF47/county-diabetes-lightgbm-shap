"""
fig03_shap_beeswarm.py
=================
Fig 3 — SHAP beeswarm (LightGBM primary model)
Changes:
1. Fix x-axis label to include "percentage points"
2. Simplify colorbar label to "Feature value"
3. Show top 15 of 22 primary predictors
4. PLOS-compliant export
"""

import pandas as pd, numpy as np, matplotlib, os, joblib, shap
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
}
def L(v): return LABELS.get(v, v.replace("_"," ").title())

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

# Get top 15 of primary 22 by mean abs SHAP
mean_abs = pd.DataFrame({
    "feature": FEATURES,
    "v": np.abs(shap_values).mean(axis=0)
})
mean_abs = mean_abs[mean_abs["feature"].isin(PRIMARY_22)]
mean_abs = mean_abs.sort_values("v", ascending=False).reset_index(drop=True)
top15_feats = mean_abs.head(15)["feature"].tolist()

print(f"Top 15 features for beeswarm:")
for i,f in enumerate(top15_feats,1):
    print(f"  {i:2d}. {f}")

# Extract SHAP values for top 15
idx     = [FEATURES.index(f) for f in top15_feats]
sv_top  = shap_values[:, idx]
X_top   = all_df[top15_feats].copy()
feat_names = [L(f) for f in top15_feats]

# Plot
shap.summary_plot(
    sv_top, X_top,
    feature_names=feat_names,
    show=False,
    plot_type="dot",
    max_display=15,
    plot_size=(13, 10),
    color_bar_label="Feature value",
)
fig = plt.gcf()
# Remove any auto-generated title
for ax in fig.get_axes():
    ax.set_title("")
fig.suptitle("")
for text in fig.texts:
    text.set_visible(False)
# Fix x-axis label
ax = fig.get_axes()[0]
ax.set_xlabel(
    "SHAP value for predicted diagnosed diabetes prevalence "
    "(percentage points)",
    fontsize=12)
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=11)
plt.tight_layout()

fig.savefig(f"{FIGURES}/Fig3.tif", dpi=300, bbox_inches="tight",
            facecolor="white", pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(f"{FIGURES}/Fig3.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(f"{FIGURES}/Fig3.pdf", bbox_inches="tight", facecolor="white")
print("\nSaved Fig3 [TIF | PNG | PDF]")
plt.close(fig)
print("Fig 3 complete.")
