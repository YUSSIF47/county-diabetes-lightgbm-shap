"""
figS10_within_region_dominant_contributor_heatmap.py
==================
S10 Fig final version:
1. Add "No positive structural contributor" row
2. Denominator = all analytic counties per region
3. Fix "Food Env. Composite Index" -> "Food Environment Composite Index"
4. Reorder columns: Northeast, Midwest, South, West
5. PLOS-compliant export
"""

import pandas as pd, numpy as np, matplotlib, os
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

DL = {
    "poverty_rate":        "Poverty rate",
    "food_insecurity_rate":"Food insecurity rate",
    "snap_rate":           "SNAP participation rate",
    "unemployment_rate":   "Unemployment rate",
    "median_income":       "Median household income",
    "food_desert_index":   "Food desert index",
    "feci":                "Food Environment Composite Index",
    "fastfood_density":    "Fast-food density",
    "pct_agri_sector":     "Agriculture sector",
    "pct_mfg_sector":      "Manufacturing sector",
    "grocery_density":     "Grocery store density",
    "pct_foodsvc_sector":  "Food service sector",
    "rucc_code":           "Rurality (RUCC)",
    "other":               "No positive structural contributor",
}

REGION_ORDER = ["Northeast", "Midwest", "South", "West"]

dom_df = pd.read_csv(f"{PROCESSED}/dominant_contributors_lgb.csv",
                     dtype={"fips":str})

# Denominator = ALL analytic counties per region (including "other")
region_total = dom_df.groupby("census_region").size()
print("Regional totals (denominator):")
print(region_total)

# Find which region contains the "other" county
other_county = dom_df[dom_df["dominant"]=="other"]
print(f"\nCounty with no positive structural SHAP:")
print(other_county[["fips","dominant","census_region"]])

# All dominant counts by region including "other"
dom_region = dom_df.groupby(
    ["dominant","census_region"]).size().reset_index(name="n")
dom_region["pct"] = dom_region.apply(
    lambda r: r["n"] / region_total[r["census_region"]] * 100, axis=1)

pivot = dom_region.pivot(
    index="dominant", columns="census_region", values="pct").fillna(0)

# Label index
pivot.index = [DL.get(i,i) for i in pivot.index]

# Sort: named contributors by national frequency, "No positive" last
named_order = (dom_df[dom_df["dominant"]!="other"]["dominant"]
    .value_counts()
    .rename(index=DL)
    .loc[lambda s: s.index.isin(pivot.index)]
    .sort_values(ascending=False)
    .index.tolist())
final_order = named_order + ["No positive structural contributor"]
pivot = pivot.loc[final_order]

# Reorder columns
pivot = pivot[REGION_ORDER]

print("\nWithin-region percentages (denominator = all regional counties):")
print(pivot.round(2).to_string())
print(f"\nColumn totals:")
print(pivot.sum().round(2))

fig, ax = plt.subplots(figsize=(11, 10))
im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
               vmin=0, vmax=pivot.values.max())

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=12)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=11)
ax.set_title("")

# Cell annotations
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i,j]
        color = "white" if val > pivot.values.max()*0.6 else "black"
        txt = f"{val:.1f}%" if val >= 0.05 else "0.0%"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")

# Grid lines
ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)

# Separator line before last row
ax.axhline(len(pivot.index)-1.5, color="black",
           linewidth=1.5, linestyle="--", alpha=0.6)

cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("% of counties within region", fontsize=11)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()

fig.savefig(f"{FIGURES}/S10_within_region_dominant_contributor_heatmap.tif", dpi=300, bbox_inches="tight",
            facecolor="white", pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(f"{FIGURES}/S10_within_region_dominant_contributor_heatmap.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(f"{FIGURES}/S10_within_region_dominant_contributor_heatmap.pdf", bbox_inches="tight", facecolor="white")
print("\nSaved S10_within_region_dominant_contributor_heatmap [TIF | PNG | PDF]")
plt.close(fig)
print("S10_within_region_dominant_contributor_heatmap final complete.")
