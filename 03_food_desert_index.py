"""
03_food_desert_index.py  (v2 - correct sheet name)
===================================================
Step 3: Create the county-level food desert index (FDI) from FARA
        tract-level data using population-weighted aggregation.

        FDI_c = (sum of LILA pop / sum of total pop) * 100

        FARA 2019 sheet name: 'Food Access Research Atlas'
        Key columns: CensusTract, Pop2010, LILATracts_halfAnd10

Input:  data/raw/USDA_FARA_2019.xlsx
        data/processed/county_master_clean.csv
Output: data/processed/county_fdi.csv
        data/processed/county_master_with_fdi.csv
        outputs/tables/fdi_descriptives.csv
        outputs/figures/fig_fdi_distribution.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

RAW       = "data/raw"
PROCESSED = "data/processed"
FIGURES   = "outputs/figures"
TABLES    = "outputs/tables"
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(TABLES,  exist_ok=True)

# ── 3A. Load FARA Excel file ──────────────────────────────────
print("Loading USDA FARA 2019...")
print("  Reading sheet: 'Food Access Research Atlas'")
print("  This may take 1-2 minutes (72,000+ tracts)...")

fara = pd.read_excel(
    f"{RAW}/USDA_FARA_2019.xlsx",
    sheet_name="Food Access Research Atlas",
    dtype={"CensusTract": str}
)

print(f"  Loaded: {fara.shape}")
print(f"  Columns: {list(fara.columns[:15])}")

# ── 3B. Create county FIPS from tract FIPS ────────────────────
# CensusTract is 11 digits; first 5 digits = county FIPS
fara["fips"] = fara["CensusTract"].astype(str).str.zfill(11).str[:5]
print(f"  Tracts: {len(fara)}")
print(f"  Counties represented: {fara['fips'].nunique()}")

# ── 3C. Confirm required columns ─────────────────────────────
required = ["CensusTract", "Pop2010",
            "LILATracts_halfAnd10",
            "LILATracts_1And10",
            "LILATracts_1And20",
            "Urban"]
missing_cols = [c for c in required if c not in fara.columns]
if missing_cols:
    print(f"\nWARNING: Missing columns: {missing_cols}")
    print(f"Available columns: {list(fara.columns)}")
else:
    print("  All required columns confirmed.")

# ── 3D. Population-weighted aggregation to county ─────────────
def compute_fdi(df, lila_col, pop_col="Pop2010"):
    """
    FDI_c = (sum of LILA population) / (sum of total population) * 100
    """
    g = df.groupby("fips").apply(
        lambda x: pd.Series({
            "lila_pop":       (x[lila_col] * x[pop_col]).sum(),
            "total_pop":       x[pop_col].sum(),
            "n_tracts":        len(x),
            "n_lila_tracts":   x[lila_col].sum(),
        })
    ).reset_index()
    g["fdi"] = (g["lila_pop"] / g["total_pop"] * 100).round(3)
    g["fdi"] = g["fdi"].replace([np.inf, -np.inf], np.nan)
    return g

# Main FDI: 0.5 mile urban / 10 mile rural (USDA primary definition)
print("\nComputing food desert index...")
main = compute_fdi(fara, "LILATracts_halfAnd10")
main = main.rename(columns={
    "fdi":           "food_desert_index",
    "n_lila_tracts": "n_lila_tracts_main"
})

# Sensitivity S4a: 1 mile / 10 mile
if "LILATracts_1And10" in fara.columns:
    s4a = compute_fdi(fara, "LILATracts_1And10")[["fips","fdi"]].rename(
        columns={"fdi": "fdi_alt_1mi10mi"})
    main = main.merge(s4a, on="fips", how="left")

# Sensitivity S4b: 1 mile / 20 mile
if "LILATracts_1And20" in fara.columns:
    s4b = compute_fdi(fara, "LILATracts_1And20")[["fips","fdi"]].rename(
        columns={"fdi": "fdi_alt_1mi20mi"})
    main = main.merge(s4b, on="fips", how="left")

# Sensitivity S4c: binary high food-desert (FDI >= 20%)
main["fdi_binary_high"] = (main["food_desert_index"] >= 20).astype(int)

print(f"  FDI computed for {len(main)} counties")
print(f"  Mean FDI:   {main['food_desert_index'].mean():.2f}%")
print(f"  Median FDI: {main['food_desert_index'].median():.2f}%")
print(f"  Min FDI:    {main['food_desert_index'].min():.2f}%")
print(f"  Max FDI:    {main['food_desert_index'].max():.2f}%")
print(f"  High FDI counties (>=20%): {main['fdi_binary_high'].sum()}")

# ── 3E. Save FDI file ─────────────────────────────────────────
fdi_cols = ["fips", "food_desert_index", "fdi_binary_high",
            "lila_pop", "total_pop", "n_tracts", "n_lila_tracts_main"]
for col in ["fdi_alt_1mi10mi", "fdi_alt_1mi20mi"]:
    if col in main.columns:
        fdi_cols.append(col)

main[fdi_cols].to_csv(f"{PROCESSED}/county_fdi.csv", index=False)
print(f"\nFDI file saved: {PROCESSED}/county_fdi.csv")

# ── 3F. Descriptive statistics ────────────────────────────────
desc = main["food_desert_index"].describe().round(3)
desc.to_csv(f"{TABLES}/fdi_descriptives.csv", header=["value"])
print(f"\nFDI descriptives:\n{desc.to_string()}")

# ── 3G. Distribution plot ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(main["food_desert_index"].dropna(),
        bins=40, color="#c0392b", edgecolor="white", alpha=0.85)
ax.axvline(main["food_desert_index"].mean(),
           color="black", linestyle="--", linewidth=1.2, label="Mean")
ax.axvline(20, color="#e67e22", linestyle=":",
           linewidth=1.2, label="High FDI threshold (20%)")
ax.set_xlabel("Food Desert Index (% county population in LILA tracts)")
ax.set_ylabel("Number of counties")
ax.set_title("Distribution of County-Level Food Desert Index (USDA FARA 2019)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES}/fig_fdi_distribution.png", dpi=150)
plt.close()
print("FDI distribution plot saved.")

# ── 3H. Merge FDI into master dataset ────────────────────────
print("\nMerging FDI into master dataset...")
master = pd.read_csv(
    f"{PROCESSED}/county_master_clean.csv",
    dtype={"fips": str}
)
master["fips"] = master["fips"].str.zfill(5)
master = master.merge(main[fdi_cols], on="fips", how="left")

n_matched   = master["food_desert_index"].notna().sum()
n_unmatched = master["food_desert_index"].isna().sum()
print(f"  Matched:   {n_matched} counties")
print(f"  Unmatched: {n_unmatched} counties")

master.to_csv(f"{PROCESSED}/county_master_with_fdi.csv", index=False)
print(f"  Saved: {PROCESSED}/county_master_with_fdi.csv")
print(f"  Shape: {master.shape}")
print("\nScript 03 complete.")
