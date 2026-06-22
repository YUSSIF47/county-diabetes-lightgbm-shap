"""
04_food_environment_composite_index.py
=======================================
Step 4: Create the Food Environment Composite Index (FECI) using PCA
        with Kaiser's criterion for component retention.

Input:  data/processed/county_master_with_fdi.csv
Output: data/processed/county_master_with_feci.csv
        outputs/tables/pca_variance_explained.csv
        outputs/figures/fig_pca_screeplot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

PROCESSED = "data/processed"
FIGURES = "outputs/figures"
TABLES = "outputs/tables"

master = pd.read_csv(f"{PROCESSED}/county_master_with_fdi.csv",
                     dtype={"fips": str})
print(f"Loaded: {master.shape}")

# ── 4A. Define PCA input variables ───────────────────────────
pca_vars = ["fastfood_density", "grocery_density", "snap_rate"]
available = [v for v in pca_vars if v in master.columns]
missing = [v for v in pca_vars if v not in master.columns]

if missing:
    print(f"Warning: PCA variables not found: {missing}")
    print("Entering available separately without composite index.")

print(f"PCA input variables: {available}")

# Rows with complete PCA inputs
pca_data = master[available].copy()
n_complete = pca_data.dropna().shape[0]
n_missing_pca = len(pca_data) - n_complete
print(f"Complete PCA rows: {n_complete} ({n_missing_pca} missing)")

# ── 4B. Standardise to z-scores ──────────────────────────────
scaler = StandardScaler()
# Fit only on non-missing rows
pca_matrix = pca_data.copy()
complete_idx = pca_matrix.dropna().index
pca_matrix_scaled = pd.DataFrame(
    scaler.fit_transform(pca_matrix.loc[complete_idx]),
    index=complete_idx,
    columns=available
)

# ── 4C. Run PCA ──────────────────────────────────────────────
pca = PCA(n_components=len(available))
pca.fit(pca_matrix_scaled)

eigenvalues = pca.explained_variance_
variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_ratio)

# ── 4D. Kaiser's criterion ───────────────────────────────────
# Retain components with eigenvalue > 1.0
n_retain = int((eigenvalues > 1.0).sum())
pc1_variance = variance_ratio[0] * 100

print(f"\nPCA results:")
for i, (ev, vr, cv) in enumerate(zip(eigenvalues, variance_ratio, cumulative_variance)):
    flag = "<-- retained (eigenvalue > 1.0)" if ev > 1.0 else ""
    print(f"  PC{i+1}: eigenvalue={ev:.3f}, variance={vr*100:.1f}%, "
          f"cumulative={cv*100:.1f}% {flag}")

print(f"\nComponents retained (Kaiser criterion): {n_retain}")
print(f"PC1 explains {pc1_variance:.1f}% of total variance")

# ── 4E. Apply Kaiser's criterion decision ─────────────────────
if n_retain >= 1:
    print("\nKaiser criterion met — using PC1 as Food Environment Composite Index.")
    scores = pca.transform(pca_matrix_scaled)
    master.loc[complete_idx, "feci"] = scores[:, 0]
    feci_used = True
else:
    print("\nKaiser criterion NOT met — indicators entered separately.")
    feci_used = False

# ── 4F. PCA loadings table ────────────────────────────────────
loadings = pd.DataFrame(
    pca.components_.T,
    index=available,
    columns=[f"PC{i+1}" for i in range(len(available))]
).round(3)

print(f"\nPCA loadings:\n{loadings.to_string()}")

variance_table = pd.DataFrame({
    "component": [f"PC{i+1}" for i in range(len(available))],
    "eigenvalue": eigenvalues.round(3),
    "variance_explained_pct": (variance_ratio * 100).round(2),
    "cumulative_pct": (cumulative_variance * 100).round(2),
    "retained_kaiser": eigenvalues > 1.0
})
variance_table.to_csv(f"{TABLES}/pca_variance_explained.csv", index=False)
loadings.to_csv(f"{TABLES}/pca_loadings.csv")

# ── 4G. Scree plot ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, "o-",
        color="#2c3e50", linewidth=2, markersize=8)
ax.axhline(1.0, color="#e74c3c", linestyle="--", linewidth=1.2,
           label="Kaiser criterion (eigenvalue = 1)")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Eigenvalue")
ax.set_title("PCA Scree Plot — Food Environment Indicators")
ax.legend()
ax.set_xticks(range(1, len(eigenvalues) + 1))
plt.tight_layout()
plt.savefig(f"{FIGURES}/fig_pca_screeplot.png", dpi=150)
plt.close()
print("Scree plot saved.")

# ── 4H. Save PCA scaler and model for later use in CV ─────────
import joblib
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/pca_scaler.pkl")
joblib.dump(pca, "models/pca_model.pkl")
print(f"PCA scaler and model saved to models/")
print(f"Kaiser criterion met: {feci_used}")

# ── 4I. Save updated master ───────────────────────────────────
master.to_csv(f"{PROCESSED}/county_master_with_feci.csv", index=False)
print(f"\nMaster with FECI saved: {PROCESSED}/county_master_with_feci.csv")
print(f"Shape: {master.shape}")
print(f"\nFECI summary:")
if "feci" in master.columns:
    print(master["feci"].describe().round(3).to_string())
