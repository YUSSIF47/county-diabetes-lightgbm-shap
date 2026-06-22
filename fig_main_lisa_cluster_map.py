"""
fig_main_lisa_cluster_map.py
=================
Fig 7 — LISA cluster map
Changes:
1. Color-blind accessible scheme:
   High-High: #b2182b (dark red)
   Low-Low:   #2166ac (dark blue)
   Low-High:  #92c5de (light blue)
   High-Low:  #f4a582 (light orange)
   Not significant: #d9d9d9 (light gray)
2. Moran's I annotation box
3. PLOS-compliant export
"""

import pandas as pd, numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local

matplotlib.rcParams.update({
    "font.family":          "DejaVu Sans",
    "font.size":            11,
    "legend.fontsize":      11,
    "legend.title_fontsize":12,
})

PROCESSED = "data/processed"
FIGURES   = "outputs/figures/final"
os.makedirs(FIGURES, exist_ok=True)
MINX,MAXX,MINY,MAXY = -125.0,-66.5,24.0,49.5

# Color-blind accessible palette
# Based on ColorBrewer RdBu diverging scheme
CC = {
    1: "#b2182b",  # High-High — dark red
    2: "#92c5de",  # Low-High  — light blue
    3: "#2166ac",  # Low-Low   — dark blue
    4: "#f4a582",  # High-Low  — light salmon
    0: "#d9d9d9",  # Not significant — light gray
}

print("Loading data...")
OUTCOME = "diabetes_prev"
train = pd.read_csv(f"{PROCESSED}/train.csv", dtype={"fips":str})
val   = pd.read_csv(f"{PROCESSED}/val.csv",   dtype={"fips":str})
test  = pd.read_csv(f"{PROCESSED}/test.csv",  dtype={"fips":str})
all_df = pd.concat([train,val,test], ignore_index=True)

gdf = gpd.read_file("data/shapefiles/cb_2019_us_county_500k.shp")
gdf["fips"] = gdf["GEOID"].str.zfill(5)
gdf_cont = gdf[~gdf["STATEFP"].isin(
    ["02","15","72","78","60","66","69"])].copy()

gdf_cont2 = gdf_cont.merge(
    all_df[["fips", OUTCOME]], on="fips", how="inner").reset_index(drop=True)

print(f"Counties for LISA: {len(gdf_cont2)}")

# Spatial weights — queen contiguity
w = Queen.from_dataframe(gdf_cont2, silence_warnings=True)
w.transform = "r"

# Global Moran's I
mi = Moran(gdf_cont2[OUTCOME].values, w, permutations=999)
print(f"Global Moran's I = {mi.I:.3f}, p = {mi.p_sim:.4f} "
      f"(999 permutations)")

# Local Moran's I
lisa = Moran_Local(gdf_cont2[OUTCOME].values, w, permutations=999,
                   seed=42)
gdf_cont2["lisa_q"]   = lisa.q
gdf_cont2["lisa_sig"] = (lisa.p_sim < 0.05).astype(int)
gdf_cont2["lisa_cat"] = gdf_cont2.apply(
    lambda r: r["lisa_q"] if r["lisa_sig"] else 0, axis=1)

print("LISA cluster counts:")
print(gdf_cont2["lisa_cat"].value_counts().sort_index())

fig, ax = plt.subplots(figsize=(18, 10))
for cat, color in CC.items():
    gdf_cont2[gdf_cont2["lisa_cat"]==cat].plot(
        ax=ax, color=color, edgecolor="none")

ax.set_xlim(MINX,MAXX); ax.set_ylim(MINY,MAXY)
ax.axis("off"); ax.set_title("")

# Legend
n_hh = (gdf_cont2["lisa_cat"]==1).sum()
n_ll = (gdf_cont2["lisa_cat"]==3).sum()
n_lh = (gdf_cont2["lisa_cat"]==2).sum()
n_hl = (gdf_cont2["lisa_cat"]==4).sum()
n_ns = (gdf_cont2["lisa_cat"]==0).sum()

legend_els = [
    mpatches.Patch(facecolor=CC[1], edgecolor="none",
        label=f"High-High cluster (n={n_hh:,})"),
    mpatches.Patch(facecolor=CC[3], edgecolor="none",
        label=f"Low-Low cluster (n={n_ll:,})"),
    mpatches.Patch(facecolor=CC[2], edgecolor="none",
        label=f"Low-High outlier (n={n_lh:,})"),
    mpatches.Patch(facecolor=CC[4], edgecolor="none",
        label=f"High-Low outlier (n={n_hl:,})"),
    mpatches.Patch(facecolor=CC[0], edgecolor="none",
        label=f"Not significant (n={n_ns:,})"),
]
ax.legend(handles=legend_els, loc="lower left", fontsize=11,
    title="LISA cluster (p < 0.05, unadjusted)",
    title_fontsize=12, framealpha=0.92,
    borderpad=0.9, labelspacing=0.65, handlelength=1.6)

# Moran's I annotation
ax.annotate(
    f"Global Moran's I = {mi.I:.3f}  (p = {mi.p_sim:.4f}, "
    f"999 permutations)",
    xy=(0.98, 0.02), xycoords="axes fraction",
    ha="right", va="bottom", fontsize=11,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

plt.tight_layout()

fig.savefig(f"{FIGURES}/Fig7.tif", dpi=300, bbox_inches="tight",
            facecolor="white", pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(f"{FIGURES}/Fig7.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(f"{FIGURES}/Fig7.pdf", bbox_inches="tight", facecolor="white")
print("\nSaved Fig7 [TIF | PNG | PDF]")
plt.close(fig)
print("Fig 7 complete — color-blind accessible palette.")
