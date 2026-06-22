"""
fix_esm10_final.py
==================
ESM_10 — LightGBM residual choropleth map
- Uses final LightGBM predictions
- Reports Moran's I for residuals
- Correct county count
"""

import pandas as pd, numpy as np, matplotlib, os, joblib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran

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

print("Loading data...")
with open(f"{PROCESSED}/feature_list_primary_22.txt") as f:
    FEATURES = [l.strip() for l in f if l.strip()]
OUTCOME = "diabetes_prev"
train = pd.read_csv(f"{PROCESSED}/train.csv", dtype={"fips":str})
val   = pd.read_csv(f"{PROCESSED}/val.csv",   dtype={"fips":str})
test  = pd.read_csv(f"{PROCESSED}/test.csv",  dtype={"fips":str})
all_df = pd.concat([train,val,test], ignore_index=True)

model = joblib.load("models/lightgbm.pkl")
all_df["y_pred"]   = model.predict(all_df[FEATURES].values)
all_df["residual"] = all_df[OUTCOME] - all_df["y_pred"]

print(f"Residual stats:")
print(f"  Mean:   {all_df['residual'].mean():.4f}")
print(f"  SD:     {all_df['residual'].std():.4f}")
print(f"  Min:    {all_df['residual'].min():.4f}")
print(f"  Max:    {all_df['residual'].max():.4f}")

gdf = gpd.read_file("data/shapefiles/cb_2019_us_county_500k.shp")
gdf["fips"] = gdf["GEOID"].str.zfill(5)
gdf_cont = gdf[~gdf["STATEFP"].isin(
    ["02","15","72","78","60","66","69"])].copy()

gdf_res = gdf_cont.merge(
    all_df[["fips","residual"]], on="fips", how="left")

n_mapped   = gdf_res["residual"].notna().sum()
n_unmapped = gdf_res["residual"].isna().sum()
print(f"Mapped counties: {n_mapped}")
print(f"Unmapped counties: {n_unmapped}")

# Compute residual Moran's I
gdf_res2 = gdf_res.dropna(subset=["residual"]).reset_index(drop=True)
w = Queen.from_dataframe(gdf_res2, silence_warnings=True)
w.transform = "r"
mi = Moran(gdf_res2["residual"].values, w, permutations=999)
print(f"Residual Moran's I = {mi.I:.3f}, p = {mi.p_sim:.4f}")

vmax = gdf_res["residual"].abs().quantile(0.95)
fig, ax = plt.subplots(figsize=(16,8))
gdf_res.plot(
    column="residual", ax=ax, legend=True,
    cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    missing_kwds={"color":"lightgrey"},
    legend_kwds={
        "label": "Residual (observed minus predicted, percentage points)",
        "shrink": 0.5, "orientation": "vertical"
    })
ax.set_xlim(MINX,MAXX); ax.set_ylim(MINY,MAXY)
ax.axis("off"); ax.set_title("")
ax.annotate(
    f"Red = underprediction  |  Blue = overprediction  |  "
    f"Residual Moran's I = {mi.I:.3f} (p = {mi.p_sim:.4f})",
    xy=(0.5,-0.01), xycoords="axes fraction",
    ha="center", va="top", fontsize=11, color="gray")
plt.tight_layout()

fig.savefig(f"{FIGURES}/ESM_10.tif", dpi=300, bbox_inches="tight",
            facecolor="white", pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(f"{FIGURES}/ESM_10.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(f"{FIGURES}/ESM_10.pdf", bbox_inches="tight", facecolor="white")
print("\nSaved ESM_10 [TIF | PNG | PDF]")
plt.close(fig)
print(f"Caption info:")
print(f"  Mapped counties: {n_mapped}")
print(f"  Unmapped: {n_unmapped}")
print(f"  Residual Moran's I = {mi.I:.3f}, p = {mi.p_sim:.4f}")
