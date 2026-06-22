"""
fix_esm4_5_6_labels.py
======================
Fixes colorbar labels for ESM_4, ESM_5, ESM_6:
- "SHAP value — Poverty rate (%)" -> "SHAP value for poverty rate (percentage points)"
- Bottom annotation updated to "contribution toward" language
- PLOS-compliant export
"""

import pandas as pd, numpy as np, matplotlib, os, joblib, shap
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd

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
MINX,MAXX,MINY,MAXY = -125.0,-66.5,24.0,49.5

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
shap_df     = pd.DataFrame(shap_values, columns=FEATURES)
shap_df["fips"] = all_df["fips"].values

gdf = gpd.read_file("data/shapefiles/cb_2019_us_county_500k.shp")
gdf["fips"] = gdf["GEOID"].str.zfill(5)
gdf_cont = gdf[~gdf["STATEFP"].isin(
    ["02","15","72","78","60","66","69"])].copy()

MAPS = [
    ("poverty_rate",   "poverty rate",              "ESM_4"),
    ("snap_rate",      "SNAP participation rate",   "ESM_5"),
    ("median_income",  "median household income",   "ESM_6"),
]

for feat, label, esm_name in MAPS:
    print(f"{esm_name}: SHAP map {feat}...")
    gdf_shap = gdf_cont.merge(
        shap_df[["fips", feat]], on="fips", how="left")
    vmax = gdf_shap[feat].abs().quantile(0.95)

    fig, ax = plt.subplots(figsize=(16,8))
    gdf_shap.plot(
        column=feat, ax=ax, legend=True,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        missing_kwds={"color":"lightgrey"},
        legend_kwds={
            "label": f"SHAP value for {label} (percentage points)",
            "shrink": 0.5,
            "orientation": "vertical"
        })
    ax.set_xlim(MINX,MAXX); ax.set_ylim(MINY,MAXY)
    ax.axis("off"); ax.set_title("")
    ax.annotate(
        "Red = contribution toward higher predicted prevalence  |  "
        "Blue = contribution toward lower predicted prevalence",
        xy=(0.5,-0.01), xycoords="axes fraction",
        ha="center", va="top", fontsize=11, color="gray")
    plt.tight_layout()

    fig.savefig(f"{FIGURES}/{esm_name}.tif", dpi=300,
                bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression":"tiff_lzw"})
    fig.savefig(f"{FIGURES}/{esm_name}.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(f"{FIGURES}/{esm_name}.pdf",
                bbox_inches="tight", facecolor="white")
    print(f"  Saved {esm_name}")
    plt.close(fig)

print("\nDone. ESM_4, ESM_5, ESM_6 colorbar labels corrected.")
