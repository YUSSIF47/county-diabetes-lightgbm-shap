"""
fig04_dominant_contributor_map.py
==============
Fig 4 final — corrected counts:
- Other structural contributors: n=597
- No positive structural contributor: n=1 (separate color)
- Not in analytic sample (light gray)
- Increased contrast between gray categories
"""

import pandas as pd, numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd

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

dom_df = pd.read_csv(f"{PROCESSED}/dominant_contributors_lgb.csv",
                     dtype={"fips":str})

NAMED = [
    "poverty_rate","food_insecurity_rate","snap_rate",
    "unemployment_rate","median_income","food_desert_index",
]
LABELS = {
    "poverty_rate":        "Poverty rate",
    "food_insecurity_rate":"Food insecurity rate",
    "snap_rate":           "SNAP participation rate",
    "unemployment_rate":   "Unemployment rate",
    "median_income":       "Median household income",
    "food_desert_index":   "Food desert index",
}
COLORS = {
    "poverty_rate":        "#e74c3c",
    "food_insecurity_rate":"#e67e22",
    "snap_rate":           "#3498db",
    "unemployment_rate":   "#9b59b6",
    "median_income":       "#2ecc71",
    "food_desert_index":   "#f39c12",
}
OTHER_COLOR   = "#707b7c"  # dark gray — other structural
NO_POS_COLOR  = "#fdfefe"  # white with outline — no positive SHAP
SAMPLE_COLOR  = "#d5d8dc"  # light gray — not in analytic sample

# Recode
def recode(d):
    if d in NAMED:       return d
    if d == "other":     return "no_positive"
    return "other_structural"

dom_df["cat"] = dom_df["dominant"].apply(recode)

# Verify counts
true_counts = dom_df["dominant"].value_counts()
named_counts = {d: int(true_counts.get(d,0)) for d in NAMED}
other_structural = int(true_counts.sum()) - sum(named_counts.values()) - int(true_counts.get("other",0))
no_positive = int(true_counts.get("other",0))

print("Verified counts:")
for d in NAMED:
    print(f"  {LABELS[d]}: {named_counts[d]}")
print(f"  Other structural contributors: {other_structural}")
print(f"  No positive structural contributor: {no_positive}")
print(f"  Sum: {sum(named_counts.values())+other_structural+no_positive}")

# Load shapefile
gdf = gpd.read_file("data/shapefiles/cb_2019_us_county_500k.shp")
gdf["fips"] = gdf["GEOID"].str.zfill(5)
gdf_cont = gdf[~gdf["STATEFP"].isin(
    ["02","15","72","78","60","66","69"])].copy()

gdf_dom = gdf_cont.merge(dom_df[["fips","cat"]], on="fips", how="left")
gdf_dom["cat"] = gdf_dom["cat"].fillna("not_in_sample")

print("\nMap counts after merge:")
print(gdf_dom["cat"].value_counts())

fig, ax = plt.subplots(figsize=(20,11))

# Plot in order: not_in_sample first (background)
gdf_dom[gdf_dom["cat"]=="not_in_sample"].plot(
    ax=ax, color=SAMPLE_COLOR, edgecolor="none")
# Other structural
gdf_dom[gdf_dom["cat"]=="other_structural"].plot(
    ax=ax, color=OTHER_COLOR, edgecolor="none")
# No positive
gdf_dom[gdf_dom["cat"]=="no_positive"].plot(
    ax=ax, color=NO_POS_COLOR, edgecolor="#2c3e50", linewidth=0.8)
# Named categories
for d in NAMED:
    gdf_dom[gdf_dom["cat"]==d].plot(
        ax=ax, color=COLORS[d], edgecolor="none")

ax.set_xlim(MINX,MAXX); ax.set_ylim(MINY,MAXY)
ax.axis("off"); ax.set_title("")

# Legend
legend_els = []
for d in NAMED:
    legend_els.append(mpatches.Patch(
        facecolor=COLORS[d], edgecolor="none",
        label=f"{LABELS[d]} (n={named_counts[d]:,})"))
legend_els.append(mpatches.Patch(
    facecolor=OTHER_COLOR, edgecolor="none",
    label=f"Other structural contributors (n={other_structural:,})"))
legend_els.append(mpatches.Patch(
    facecolor=NO_POS_COLOR, edgecolor="#2c3e50", linewidth=0.8,
    label=f"No positive structural contributor (n={no_positive:,})"))
legend_els.append(mpatches.Patch(
    facecolor=SAMPLE_COLOR, edgecolor="none",
    label="Not in analytic sample"))

ax.legend(handles=legend_els, loc="lower left", fontsize=11,
    title="Dominant positive structural SHAP contributor",
    title_fontsize=12, framealpha=0.92, ncol=2,
    borderpad=0.9, labelspacing=0.65, handlelength=1.6)

plt.tight_layout()

fig.savefig(f"{FIGURES}/Fig4.tif", dpi=300, bbox_inches="tight",
            facecolor="white", pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(f"{FIGURES}/Fig4.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(f"{FIGURES}/Fig4.pdf", bbox_inches="tight", facecolor="white")
print("\nSaved Fig4 [TIF | PNG | PDF]")
plt.close(fig)
print("Fig 4 v3 complete — corrected counts, three gray levels.")
