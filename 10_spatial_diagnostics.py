"""
10_spatial_diagnostics.py  (v2 — high-resolution figures)
==========================================================
Step 10: Moran's I, LISA maps, spatial residual checks.
         All figures saved as PDF + PNG (300 DPI) + TIF (600 DPI).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, os

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
    fig.savefig(f"{path}.tif", format="tiff", dpi=dpi_tif, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {name} [.pdf | .png {dpi_png}dpi | .tif {dpi_tif}dpi]")
    plt.close(fig)

# ── Load best model and data ──────────────────────────────────
with open(f"{PROCESSED}/best_tree_model.txt") as f:
    best_model_name = f.read().strip()

model_file = {
    "XGBoost":      "models/xgboost.pkl",
    "LightGBM":     "models/lightgbm.pkl",
    "Random Forest":"models/random_forest.pkl",
}[best_model_name]

model  = joblib.load(model_file)
train  = pd.read_csv(f"{PROCESSED}/train.csv", dtype={"fips": str})
val    = pd.read_csv(f"{PROCESSED}/val.csv",   dtype={"fips": str})
test   = pd.read_csv(f"{PROCESSED}/test.csv",  dtype={"fips": str})

with open(f"{PROCESSED}/feature_list_primary_22.txt") as f:
    FEATURES = [line.strip() for line in f if line.strip()]

OUTCOME  = "diabetes_prev"
all_df   = pd.concat([train, val, test], ignore_index=True)
all_df["y_pred"]  = model.predict(all_df[FEATURES].values)
all_df["residual"]= all_df[OUTCOME] - all_df["y_pred"]
print(f"Full dataset for spatial analysis: {len(all_df)} counties")

# ── Spatial analysis ──────────────────────────────────────────
try:
    import geopandas as gpd
    from libpysal.weights import Queen, Rook, KNN
    from esda.moran import Moran, Moran_Local

    gdf      = gpd.read_file("data/shapefiles/cb_2019_us_county_500k.shp")
    gdf["fips"] = gdf["GEOID"].str.zfill(5)
    gdf      = gdf.merge(
        all_df[["fips", OUTCOME, "y_pred", "residual"]],
        on="fips", how="inner"
    )
    gdf_cont = gdf[~gdf["STATEFP"].isin(["02","15"])].copy().reset_index(drop=True)
    print(f"Continental US counties: {len(gdf_cont)}")

    # Spatial weights
    w_queen  = Queen.from_dataframe(gdf_cont, silence_warnings=True)
    w_queen.transform = "r"

    # Moran's I — outcome
    mi_out   = Moran(gdf_cont[OUTCOME].values,  w_queen)
    mi_res   = Moran(gdf_cont["residual"].values, w_queen)

    # Sensitivity
    w_rook   = Rook.from_dataframe(gdf_cont, silence_warnings=True)
    w_rook.transform = "r"
    mi_rook  = Moran(gdf_cont[OUTCOME].values, w_rook)

    w_knn    = KNN.from_dataframe(gdf_cont, k=5, silence_warnings=True)
    w_knn.transform  = "r"
    mi_knn   = Moran(gdf_cont[OUTCOME].values, w_knn)

    morans_df = pd.DataFrame([
        {"measure":"T2DM prevalence (Queen)",  "I":round(mi_out.I,4), "p":round(mi_out.p_sim,4),  "weights":"Queen"},
        {"measure":"Model residuals (Queen)",   "I":round(mi_res.I,4), "p":round(mi_res.p_sim,4),  "weights":"Queen"},
        {"measure":"T2DM prevalence (Rook)",   "I":round(mi_rook.I,4),"p":round(mi_rook.p_sim,4), "weights":"Rook"},
        {"measure":"T2DM prevalence (KNN k=5)","I":round(mi_knn.I,4), "p":round(mi_knn.p_sim,4),  "weights":"KNN(k=5)"},
    ])
    morans_df.to_csv(f"{TABLES}/morans_i_results.csv", index=False)
    print(f"\nMoran's I results:\n{morans_df.to_string()}")

    # LISA
    from esda.moran import Moran_Local
    lisa = Moran_Local(gdf_cont[OUTCOME].values, w_queen)
    gdf_cont["lisa_q"]   = lisa.q
    gdf_cont["lisa_sig"] = lisa.p_sim < 0.05

    quad_colors = {1:"#e74c3c", 2:"#3498db", 3:"#2ecc71", 4:"#f39c12"}
    quad_labels = {1:"High-High", 2:"Low-High", 3:"Low-Low", 4:"High-Low"}
    gdf_cont["lisa_color"] = gdf_cont.apply(
        lambda r: quad_colors.get(r["lisa_q"],"#bdc3c7")
        if r["lisa_sig"] else "#bdc3c7", axis=1
    )

    # Figure: LISA cluster map
    fig, ax = plt.subplots(figsize=(16, 9))
    gdf_cont.plot(color=gdf_cont["lisa_color"], ax=ax, edgecolor="none")
    legend_elements = [
        mpatches.Patch(facecolor="#e74c3c", label="High-High cluster"),
        mpatches.Patch(facecolor="#2ecc71", label="Low-Low cluster"),
        mpatches.Patch(facecolor="#3498db", label="Low-High outlier"),
        mpatches.Patch(facecolor="#f39c12", label="High-Low outlier"),
        mpatches.Patch(facecolor="#bdc3c7", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10,
              title="LISA cluster (p < 0.05)", title_fontsize=10)
    ax.axis("off")
    ax.set_title(f"LISA Cluster Map — County-Level T2DM Prevalence\n"
                 f"Global Moran's I = {mi_out.I:.3f} (p = {mi_out.p_sim:.4f})",
                 fontsize=13, pad=12)
    plt.tight_layout()
    save_fig(fig, "fig_lisa_clusters")

    # Figure: Residual choropleth
    fig, ax = plt.subplots(figsize=(16, 9))
    vmax = gdf_cont["residual"].abs().quantile(0.95)
    gdf_cont.plot(column="residual", ax=ax, legend=True,
                  cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  missing_kwds={"color":"lightgrey"},
                  legend_kwds={"label":"Residual (observed − predicted)",
                               "shrink":0.6})
    ax.axis("off")
    ax.set_title(f"Model Residuals by County — {best_model_name}\n"
                 f"Residual Moran's I = {mi_res.I:.3f} (p = {mi_res.p_sim:.4f})",
                 fontsize=13, pad=12)
    plt.tight_layout()
    save_fig(fig, "fig_residual_choropleth")

    # Figure: Moran scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mi, title in zip(axes,
        [mi_out, mi_res],
        ["T2DM Prevalence", "Model Residuals"]
    ):
        y    = mi.y
        wy   = mi.wy
        ax.scatter(y, wy, alpha=0.4, s=10, color="#2c3e50")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axvline(0, color="grey", linewidth=0.8)
        m, b = np.polyfit(y, wy, 1)
        xline = np.linspace(y.min(), y.max(), 100)
        ax.plot(xline, m*xline+b, color="#e74c3c", linewidth=1.5)
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel(f"Spatial lag of {title}", fontsize=11)
        ax.set_title(f"Moran Scatter — {title}\n"
                     f"I = {mi.I:.3f}, p = {mi.p_sim:.4f}", fontsize=11)
    plt.suptitle("Moran Scatter Plots", fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig(fig, "fig_morans_scatter")

    print("Spatial analysis complete.")

except ImportError as e:
    print(f"Spatial analysis requires geopandas/libpysal/esda: {e}")
    pd.DataFrame([{"note":"Spatial libraries not installed"}]).to_csv(
        f"{TABLES}/morans_i_results.csv", index=False)
except Exception as e:
    print(f"Spatial analysis error: {e}")

print(f"\nAll figures: {FIGURES}/")
print("Each figure: .pdf | .png (300 DPI) | .tif (600 DPI)")
