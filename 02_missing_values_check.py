"""
02_missing_values_check.py
==========================
Step 2: Check missing values and document excluded counties.

Input:  data/processed/county_master_pre_fdi.csv
Output: data/processed/missing_report.csv
        data/processed/excluded_counties.csv
        outputs/figures/fig_missing_heatmap.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

PROCESSED = "data/processed"
FIGURES = "outputs/figures"
TABLES = "outputs/tables"
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

master = pd.read_csv(f"{PROCESSED}/county_master_pre_fdi.csv", dtype={"fips": str})
print(f"Loaded master dataset: {master.shape}")
print(f"Total counties: {len(master)}")

# ── 2A. Outcome missingness ───────────────────────────────────
missing_outcome = master["diabetes_prev"].isna()
n_excluded = missing_outcome.sum()
print(f"\nCounties excluded (missing outcome): {n_excluded}")

excluded = master[missing_outcome][["fips"]].copy()
excluded.to_csv(f"{PROCESSED}/excluded_counties.csv", index=False)

# Drop counties with missing outcome
master_clean = master[~missing_outcome].copy()
print(f"Counties retained for analysis: {len(master_clean)}")

# ── 2B. Predictor missingness report ─────────────────────────
missing_counts = master_clean.isnull().sum()
missing_pct = (master_clean.isnull().mean() * 100).round(2)

missing_report = pd.DataFrame({
    "variable": missing_counts.index,
    "n_missing": missing_counts.values,
    "pct_missing": missing_pct.values
}).sort_values("pct_missing", ascending=False)

missing_report.to_csv(f"{TABLES}/missing_report.csv", index=False)
print(f"\nMissing value report:\n{missing_report[missing_report['n_missing'] > 0].to_string()}")

# ── 2C. Missingness heatmap ───────────────────────────────────
pred_cols = [c for c in master_clean.columns if c != "fips"]
fig, ax = plt.subplots(figsize=(14, 6))
miss_matrix = master_clean[pred_cols].isnull().astype(int)
ax.imshow(miss_matrix.T, aspect="auto", cmap="Reds", interpolation="none")
ax.set_yticks(range(len(pred_cols)))
ax.set_yticklabels(pred_cols, fontsize=7)
ax.set_xlabel("County index")
ax.set_title("Missing data pattern (red = missing)")
plt.tight_layout()
plt.savefig(f"{FIGURES}/fig_missing_heatmap.png", dpi=150)
plt.close()
print(f"\nMissing heatmap saved.")

# ── 2D. Summary table for paper ──────────────────────────────
summary = {
    "Total U.S. counties": len(master),
    "Excluded (missing outcome)": n_excluded,
    "Retained for analysis": len(master_clean),
}
print("\nSummary for paper:")
for k, v in summary.items():
    print(f"  {k}: {v}")

# Save clean master (still without FDI — added after step 03)
master_clean.to_csv(f"{PROCESSED}/county_master_clean.csv", index=False)
print(f"\nClean master saved: {PROCESSED}/county_master_clean.csv")
