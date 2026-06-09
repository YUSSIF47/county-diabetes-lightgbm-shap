"""
08_09_shap_analysis.py  (v2 — high-resolution figures)
=======================================================
Steps 8 & 9: Select best tree model and run full SHAP analysis.

Outputs all figures in PDF + PNG (300 DPI) + TIF (600 DPI).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib, os, json

PROCESSED = "data/processed"
FIGURES   = "outputs/figures"
TABLES    = "outputs/tables"
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(TABLES,  exist_ok=True)

# ── Figure saving utility ─────────────────────────────────────
def save_fig(fig, name, dpi_png=300, dpi_tif=600):
    path = f"{FIGURES}/{name}"
    fig.savefig(f"{path}.pdf", format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{path}.png", format="png", dpi=dpi_png, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{path}.tif", format="tiff", dpi=dpi_tif, bbox_inches="tight",
                facecolor="white")
    print(f"  Saved: {name} [.pdf | .png {dpi_png}dpi | .tif {dpi_tif}dpi]")
    plt.close(fig)

# ── Load best model ───────────────────────────────────────────
with open(f"{PROCESSED}/best_tree_model.txt") as f:
    best_model_name = f.read().strip()
print(f"Best tree model: {best_model_name}")

model_file = {
    "XGBoost":      "models/xgboost.pkl",
    "LightGBM":     "models/lightgbm.pkl",
    "Random Forest":"models/random_forest.pkl",
}[best_model_name]
model = joblib.load(model_file)

# ── Load data ─────────────────────────────────────────────────
test  = pd.read_csv(f"{PROCESSED}/test.csv",  dtype={"fips": str})
train = pd.read_csv(f"{PROCESSED}/train.csv", dtype={"fips": str})

with open(f"{PROCESSED}/feature_list.txt") as f:
    FEATURES = [line.strip() for line in f if line.strip()]

OUTCOME = "diabetes_prev"
X_test  = test[FEATURES]
y_test  = test[OUTCOME].values

# ── SHAP TreeExplainer ────────────────────────────────────────
print("\nRunning SHAP TreeExplainer...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap_df = pd.DataFrame(shap_values, columns=FEATURES)
shap_df["fips"]          = test["fips"].values
shap_df["census_region"] = test["census_region"].values
shap_df["y_true"]        = y_test
shap_df["y_pred"]        = model.predict(X_test.values)
shap_df.to_csv(f"{TABLES}/shap_county_values.csv", index=False)
print(f"SHAP values computed: {shap_df.shape}")

# ── (1) Global feature importance ────────────────────────────
mean_abs_shap = pd.DataFrame({
    "feature":       FEATURES,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

mean_abs_shap.to_csv(f"{TABLES}/shap_global_importance.csv", index=False)
print(f"\nTop 10 features:\n{mean_abs_shap.head(10).to_string()}")

# Colour code by feature group
structural = ["food_desert_index","fdi_binary_high","fastfood_density",
              "grocery_density","snap_rate","food_insecurity_rate","feci",
              "poverty_rate","median_income","unemployment_rate",
              "pct_agri_sector","pct_mfg_sector","pct_foodsvc_sector",
              "rucc_code","rural_flag"]
top20 = mean_abs_shap.head(20)
colors = ["#c0392b" if f in structural else "#2980b9"
          for f in top20["feature"]]

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1],
               color=colors[::-1], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Mean |SHAP value|", fontsize=13)
ax.set_title(f"Global Feature Importance — {best_model_name}\n"
             f"Top 20 features (red = structural determinants)",
             fontsize=13, pad=12)
legend_elements = [
    mpatches.Patch(facecolor="#c0392b", label="Structural determinants"),
    mpatches.Patch(facecolor="#2980b9", label="Health-behaviour / demographic"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
ax.tick_params(axis="y", labelsize=9)
ax.tick_params(axis="x", labelsize=10)
plt.tight_layout()
save_fig(fig, "fig_shap_global_importance")

# ── (2) SHAP beeswarm ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=FEATURES,
                  max_display=20, show=False, plot_size=None)
plt.title(f"SHAP Summary Plot — {best_model_name}", fontsize=13, pad=12)
plt.tight_layout()
save_fig(plt.gcf(), "fig_shap_beeswarm")

# ── (3) Regional SHAP profiles heatmap ───────────────────────
regional = (
    shap_df.groupby("census_region")[FEATURES]
    .mean().round(4)
)
regional.to_csv(f"{TABLES}/shap_regional_profiles.csv")

top_feats = mean_abs_shap.head(15)["feature"].tolist()
regional_top = regional[top_feats].T

fig, ax = plt.subplots(figsize=(9, 9))
im = ax.imshow(regional_top.values, cmap="RdBu_r", aspect="auto",
               vmin=-regional_top.abs().max().max(),
               vmax= regional_top.abs().max().max())
ax.set_xticks(range(len(regional_top.columns)))
ax.set_xticklabels(regional_top.columns, fontsize=11)
ax.set_yticks(range(len(regional_top.index)))
ax.set_yticklabels(regional_top.index, fontsize=9)
plt.colorbar(im, ax=ax, label="Mean SHAP value", shrink=0.8)
ax.set_title(f"Regional SHAP Profiles — Top 15 Features\n({best_model_name})",
             fontsize=13, pad=12)
plt.tight_layout()
save_fig(fig, "fig_shap_regional_heatmap")

# ── (4) Dominant-driver analysis ─────────────────────────────
structural_feats = [f for f in structural if f in FEATURES]

pos_shap = shap_df[structural_feats].clip(lower=0)
shap_df["dominant_driver"] = pos_shap.idxmax(axis=1)
shap_df.loc[pos_shap.max(axis=1) == 0, "dominant_driver"] = "None"

dominant_summary = (
    shap_df.groupby(["dominant_driver","census_region"])
    .size().reset_index(name="n_counties")
)
dominant_summary.to_csv(f"{TABLES}/dominant_driver_summary.csv", index=False)
print(f"\nDominant driver summary:\n{dominant_summary.to_string()}")

# Dominant driver bar chart by region
pivot = dominant_summary.pivot(
    index="dominant_driver", columns="census_region", values="n_counties"
).fillna(0)

fig, ax = plt.subplots(figsize=(12, 6))
pivot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white",
           linewidth=0.5)
ax.set_xlabel("Dominant Structural Driver", fontsize=12)
ax.set_ylabel("Number of counties", fontsize=12)
ax.set_title("Dominant Structural SHAP Driver by Census Region\n"
             f"({best_model_name})", fontsize=13, pad=12)
ax.legend(title="Census Region", fontsize=10)
ax.tick_params(axis="x", rotation=30, labelsize=9)
plt.tight_layout()
save_fig(fig, "fig_dominant_driver_by_region")

# ── (5) Dependence plots for top 3 structural features ────────
top3_structural = [f for f in mean_abs_shap["feature"].tolist()
                   if f in structural][:3]

for feat in top3_structural:
    feat_idx = list(X_test.columns).index(feat)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(feat_idx, shap_values, X_test.values,
                         feature_names=FEATURES, ax=ax, show=False,
                         dot_size=8, alpha=0.6)
    ax.set_title(f"SHAP Dependence Plot: {feat}\n({best_model_name})",
                 fontsize=12, pad=10)
    plt.tight_layout()
    save_fig(fig, f"fig_shap_dependence_{feat}")

# ── (6) Choropleth maps — requires geopandas ─────────────────
try:
    import geopandas as gpd
    gdf = gpd.read_file("data/shapefiles/cb_2019_us_county_500k.shp")
    gdf["fips"]   = gdf["GEOID"].str.zfill(5)
    gdf_cont = gdf[~gdf["STATEFP"].isin(["02","15"])].copy()

    # Merge SHAP dominant driver
    driver_df = shap_df[["fips","dominant_driver"]].copy()
    gdf_dom   = gdf_cont.merge(driver_df, on="fips", how="left")

    driver_categories = [d for d in gdf_dom["dominant_driver"].dropna().unique()
                         if d != "None"]
    palette = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c"]
    color_map = {d: c for d, c in zip(driver_categories, palette)}
    color_map["None"] = "#bdc3c7"

    gdf_dom["color"] = gdf_dom["dominant_driver"].map(color_map).fillna("#bdc3c7")

    fig, ax = plt.subplots(figsize=(16, 9))
    for driver, color in color_map.items():
        subset = gdf_dom[gdf_dom["dominant_driver"] == driver]
        if len(subset):
            subset.plot(ax=ax, color=color, edgecolor="none")
    legend_elements = [
        mpatches.Patch(facecolor=c, label=d)
        for d, c in color_map.items()
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              fontsize=10, title="Dominant structural driver",
              title_fontsize=10)
    ax.axis("off")
    ax.set_title("Dominant Structural Driver of County-Level T2DM Burden\n"
                 f"({best_model_name} + SHAP)", fontsize=14, pad=14)
    plt.tight_layout()
    save_fig(fig, "fig_dominant_driver_map")

    # County SHAP choropleth for top 3 structural features
    shap_geo = shap_df[["fips"] + top3_structural]
    gdf_shap = gdf_cont.merge(shap_geo, on="fips", how="left")

    for feat in top3_structural:
        fig, ax = plt.subplots(figsize=(16, 9))
        gdf_shap.plot(column=feat, ax=ax, legend=True,
                      cmap="RdBu_r",
                      missing_kwds={"color": "lightgrey"},
                      legend_kwds={"label": f"SHAP value ({feat})",
                                   "shrink": 0.6})
        ax.axis("off")
        ax.set_title(f"County-Level SHAP Values: {feat}\n({best_model_name})",
                     fontsize=13, pad=12)
        plt.tight_layout()
        save_fig(fig, f"fig_shap_map_{feat}")

    print("Choropleth maps saved.")

except Exception as e:
    print(f"  Choropleth maps skipped: {e}")
    print("  Install geopandas to generate county maps.")

print("\nSHAP analysis complete.")
print(f"\nAll figures saved in: {FIGURES}/")
print("Each figure has three formats: .pdf | .png (300 DPI) | .tif (600 DPI)")
