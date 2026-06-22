"""
figS11_regional_signed_shap_heatmap.py
==============
S11 Fig: regional mean signed SHAP heatmap.
"""

import pandas as pd, numpy as np, matplotlib, os, joblib, shap
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.labelsize":  12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
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

REGION_ORDER = ["Northeast","Midwest","South","West"]

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

mean_abs = pd.DataFrame({
    "feature": FEATURES,
    "v": np.abs(shap_values).mean(axis=0)
})
mean_abs = mean_abs[mean_abs["feature"].isin(PRIMARY_22)]
mean_abs = mean_abs.sort_values("v", ascending=False).reset_index(drop=True)
top15 = mean_abs.head(15)["feature"].tolist()

shap_df = pd.DataFrame(shap_values, columns=FEATURES)
shap_df["census_region"] = all_df["census_region"].values
regional = shap_df.groupby("census_region")[top15].mean().round(4)
regional_top = regional[top15].T
regional_top.index = [L(f) for f in regional_top.index]
regional_top = regional_top[REGION_ORDER]

fig, ax = plt.subplots(figsize=(12, 11))
vmax = regional_top.abs().max().max()
im   = ax.imshow(regional_top.values, cmap="RdBu_r",
                 aspect="auto", vmin=-vmax, vmax=vmax)

ax.set_xticks(range(len(regional_top.columns)))
ax.set_xticklabels(regional_top.columns, fontsize=12)
ax.set_yticks(range(len(regional_top.index)))
ax.set_yticklabels(regional_top.index, fontsize=11)
ax.set_title("")

# Cell values — fix negative zero
for i in range(len(regional_top.index)):
    for j in range(len(regional_top.columns)):
        val = regional_top.values[i, j]
        # Fix negative zero
        display_val = 0.0 if abs(val) < 0.005 else val
        abs_norm = abs(val) / vmax
        color = "white" if abs_norm > 0.55 else "black"
        ax.text(j, i, f"{display_val:.2f}", ha="center", va="center",
                fontsize=9.5, color=color, fontweight="bold")

cbar = plt.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
cbar.ax.tick_params(labelsize=10)
cbar.set_label("Mean signed SHAP value (percentage points)", fontsize=11)

ax.set_xticks(np.arange(-0.5, len(regional_top.columns), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(regional_top.index), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)

plt.tight_layout()

fig.savefig(f"{FIGURES}/S11_regional_signed_shap_heatmap.tif", dpi=300, bbox_inches="tight",
            facecolor="white", pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(f"{FIGURES}/S11_regional_signed_shap_heatmap.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(f"{FIGURES}/S11_regional_signed_shap_heatmap.pdf", bbox_inches="tight", facecolor="white")
print("Saved S11_regional_signed_shap_heatmap [TIF | PNG | PDF]")
plt.close(fig)
print("S11 regional signed SHAP heatmap complete.")
