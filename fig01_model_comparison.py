"""
fig01_model_comparison.py
=================
Final clean Fig 1 — no arrow, no legend.
Black outline identifies LightGBM. Caption explains everything.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import joblib, os

matplotlib.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.labelsize":  12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize":  12,
})

PROCESSED = "data/processed"
FIGURES   = "outputs/figures/final"
os.makedirs(FIGURES, exist_ok=True)

with open(f"{PROCESSED}/feature_list_primary_22.txt") as f:
    FEATURES = [l.strip() for l in f if l.strip()]
OUTCOME = "diabetes_prev"

val  = pd.read_csv(f"{PROCESSED}/val.csv",  dtype={"fips":str})
test = pd.read_csv(f"{PROCESSED}/test.csv", dtype={"fips":str})
X_val  = val[FEATURES].values;  y_val  = val[OUTCOME].values
X_test = test[FEATURES].values; y_test = test[OUTCOME].values

n_val  = len(y_val)
n_test = len(y_test)
print(f"Validation n={n_val}, Test n={n_test}")

models = {
    "Elastic Net":    joblib.load("models/elastic_net.pkl"),
    "Random Forest":  joblib.load("models/random_forest.pkl"),
    "XGBoost":        joblib.load("models/xgboost.pkl"),
    "LightGBM":       joblib.load("models/lightgbm.pkl"),
}

MODEL_ORDER = ["LightGBM", "XGBoost", "Random Forest", "Elastic Net"]
COLORS = {
    "LightGBM":     "#e74c3c",
    "XGBoost":      "#e67e22",
    "Random Forest":"#3498db",
    "Elastic Net":  "#95a5a6",
}
EDGE = {m: "black" if m=="LightGBM" else "white" for m in MODEL_ORDER}
LW   = {m: 2.0    if m=="LightGBM" else 0.5     for m in MODEL_ORDER}

# Validation RMSE
val_rmse = {}
for name, m in models.items():
    yp = m.predict(X_val)
    val_rmse[name] = np.sqrt(mean_squared_error(y_val, yp))

# Test RMSE + bootstrap CI
test_rmse = {}
boot_lo   = {}
boot_hi   = {}
rng = np.random.default_rng(42)
for name, m in models.items():
    yp = m.predict(X_test)
    test_rmse[name] = np.sqrt(mean_squared_error(y_test, yp))
    boots = []
    for _ in range(1000):
        idx = rng.choice(len(y_test), size=len(y_test), replace=True)
        boots.append(np.sqrt(mean_squared_error(y_test[idx], yp[idx])))
    boot_lo[name] = np.percentile(boots, 2.5)
    boot_hi[name] = np.percentile(boots, 97.5)

print("\nValidation RMSE:")
for m in MODEL_ORDER:
    print(f"  {m}: {val_rmse[m]:.4f}")
print("\nTest RMSE:")
for m in MODEL_ORDER:
    print(f"  {m}: {test_rmse[m]:.4f} "
          f"(95% CI: {boot_lo[m]:.3f}--{boot_hi[m]:.3f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ── Panel A: Validation RMSE — no arrow, no legend ───────────
v_vals = [val_rmse[m] for m in MODEL_ORDER]
bars1  = ax1.bar(MODEL_ORDER, v_vals,
                 color=[COLORS[m] for m in MODEL_ORDER],
                 edgecolor=[EDGE[m] for m in MODEL_ORDER],
                 linewidth=[LW[m] for m in MODEL_ORDER],
                 width=0.55)
ax1.set_ylabel("RMSE (percentage points)", fontsize=12)
ax1.set_xlabel("Model", fontsize=12)
ax1.set_ylim(0, max(v_vals) * 1.25)
ax1.set_title(
    f"A.  Validation-set RMSE\n"
    f"(used for model selection; n = {n_val} counties)",
    fontsize=11, loc="left", pad=8)
for bar, val in zip(bars1, v_vals):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.006,
             f"{val:.3f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

# ── Panel B: Test RMSE with CI — no arrow, no legend ─────────
t_vals = [test_rmse[m] for m in MODEL_ORDER]
t_yerr = np.array([
    [test_rmse[m] - boot_lo[m] for m in MODEL_ORDER],
    [boot_hi[m]  - test_rmse[m] for m in MODEL_ORDER]
])
bars2 = ax2.bar(MODEL_ORDER, t_vals,
                color=[COLORS[m] for m in MODEL_ORDER],
                edgecolor=[EDGE[m] for m in MODEL_ORDER],
                linewidth=[LW[m] for m in MODEL_ORDER],
                width=0.55,
                yerr=t_yerr, capsize=5,
                error_kw={"elinewidth":1.5, "ecolor":"#2c3e50"})
ax2.set_ylabel("RMSE (percentage points)", fontsize=12)
ax2.set_xlabel("Model", fontsize=12)
ax2.set_ylim(0, max(t_vals) * 1.25)
ax2.set_title(
    f"B.  Held-out test RMSE\n"
    f"(95% bootstrap CIs; n = {n_test} counties)",
    fontsize=11, loc="left", pad=8)

# Labels above upper CI cap
for m, bar in zip(MODEL_ORDER, bars2):
    upper_cap = boot_hi[m]
    ax2.text(bar.get_x() + bar.get_width()/2,
             upper_cap + 0.012,
             f"{test_rmse[m]:.3f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

plt.tight_layout()

# PLOS-compliant save
fig.savefig(f"{FIGURES}/Fig1.tif", dpi=300, bbox_inches="tight",
            facecolor="white", pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(f"{FIGURES}/Fig1.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(f"{FIGURES}/Fig1.pdf", bbox_inches="tight", facecolor="white")
print("\nSaved Fig1 [TIF | PNG | PDF]")
plt.close(fig)
print("Clean version complete — no arrow, no legend.")
