"""
fix_esm7_8_9_unscaled.py
========================
Regenerates ESM_7, ESM_8, ESM_9 using ORIGINAL unscaled predictor
values on x-axis and colorbar.
SHAP values remain from the scaled model — only the display values
are converted back to original units.
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

print("Loading scaled and unscaled data...")
with open(f"{PROCESSED}/feature_list_primary_22.txt") as f:
    FEATURES = [l.strip() for l in f if l.strip()]
OUTCOME = "diabetes_prev"

# Scaled data for SHAP computation
train = pd.read_csv(f"{PROCESSED}/train.csv", dtype={"fips":str})
val   = pd.read_csv(f"{PROCESSED}/val.csv",   dtype={"fips":str})
test  = pd.read_csv(f"{PROCESSED}/test.csv",  dtype={"fips":str})
all_df = pd.concat([train,val,test], ignore_index=True)

# Unscaled data for display on axes
train_us = pd.read_csv(f"{PROCESSED}/train_unscaled.csv", dtype={"fips":str})
val_us   = pd.read_csv(f"{PROCESSED}/val_unscaled.csv",   dtype={"fips":str})
test_us  = pd.read_csv(f"{PROCESSED}/test_unscaled.csv",  dtype={"fips":str})
all_us   = pd.concat([train_us,val_us,test_us], ignore_index=True)

print(f"Scaled data shape: {all_df.shape}")
print(f"Unscaled data shape: {all_us.shape}")

# Verify unscaled values look correct
for feat in ["poverty_rate","snap_rate","median_income","pct_white","food_insecurity_rate"]:
    if feat in all_us.columns:
        print(f"  {feat}: min={all_us[feat].min():.1f}, max={all_us[feat].max():.1f}, mean={all_us[feat].mean():.1f}")

print("\nComputing LightGBM SHAP values...")
model     = joblib.load("models/lightgbm.pkl")
explainer = shap.TreeExplainer(model,
                feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(all_df[FEATURES])

# ── ESM_7: Poverty rate dependence — original units ──────────
print("\nESM_7: Poverty rate dependence (original units)...")
feat      = "poverty_rate"
interact  = "median_income"
feat_idx  = FEATURES.index(feat)
x_vals    = all_us[feat].values      # original % values
c_vals    = all_us[interact].values  # original $ values
y_vals    = shap_values[:, feat_idx]

fig, ax = plt.subplots(figsize=(10,7))
sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap="coolwarm",
                alpha=0.5, s=14, edgecolors="none")
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("Median household income (USD)", fontsize=11)
cbar.ax.tick_params(labelsize=10)
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_xlabel("Poverty rate (%)", fontsize=12)
ax.set_ylabel("SHAP value for poverty rate (percentage points)", fontsize=12)
ax.set_title("")
plt.tight_layout()
fig.savefig(f"{FIGURES}/ESM_7.tif", dpi=300, bbox_inches="tight",
            facecolor="white", pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(f"{FIGURES}/ESM_7.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(f"{FIGURES}/ESM_7.pdf", bbox_inches="tight", facecolor="white")
print("  Saved ESM_7")
plt.close(fig)

# ── ESM_8: SNAP rate dependence — original units ─────────────
print("ESM_8: SNAP rate dependence (original units)...")
feat      = "snap_rate"
interact  = "food_insecurity_rate"
feat_idx  = FEATURES.index(feat)
x_vals    = all_us[feat].values
c_vals    = all_us[interact].values
y_vals    = shap_values[:, feat_idx]

fig, ax = plt.subplots(figsize=(10,7))
sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap="coolwarm",
                alpha=0.5, s=14, edgecolors="none")
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("Food insecurity rate (%)", fontsize=11)
cbar.ax.tick_params(labelsize=10)
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_xlabel("SNAP participation rate (%)", fontsize=12)
ax.set_ylabel("SHAP value for SNAP participation rate (percentage points)",
              fontsize=12)
ax.set_title("")
plt.tight_layout()
fig.savefig(f"{FIGURES}/ESM_8.tif", dpi=300, bbox_inches="tight",
            facecolor="white", pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(f"{FIGURES}/ESM_8.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(f"{FIGURES}/ESM_8.pdf", bbox_inches="tight", facecolor="white")
print("  Saved ESM_8")
plt.close(fig)

# ── ESM_9: Median income dependence — original units ─────────
print("ESM_9: Median income dependence (original units)...")
feat      = "median_income"
interact  = "pct_white"
feat_idx  = FEATURES.index(feat)
x_vals    = all_us[feat].values
c_vals    = all_us[interact].values
y_vals    = shap_values[:, feat_idx]

# Express income in thousands for readability
x_vals_k  = x_vals / 1000

fig, ax = plt.subplots(figsize=(10,7))
sc = ax.scatter(x_vals_k, y_vals, c=c_vals, cmap="coolwarm",
                alpha=0.5, s=14, edgecolors="none")
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("White population (%)", fontsize=11)
cbar.ax.tick_params(labelsize=10)
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_xlabel("Median household income (USD thousands)", fontsize=12)
ax.set_ylabel("SHAP value for median household income (percentage points)",
              fontsize=12)
ax.set_title("")
plt.tight_layout()
fig.savefig(f"{FIGURES}/ESM_9.tif", dpi=300, bbox_inches="tight",
            facecolor="white", pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(f"{FIGURES}/ESM_9.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(f"{FIGURES}/ESM_9.pdf", bbox_inches="tight", facecolor="white")
print("  Saved ESM_9")
plt.close(fig)

print("\nDone. ESM_7, ESM_8, ESM_9 regenerated with original unscaled units.")
