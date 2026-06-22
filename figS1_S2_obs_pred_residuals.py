"""
fix_esm2_esm3_final.py
======================
ESM_2: Observed vs predicted — fixed model order, common axes
ESM_3: Residual plots — fixed model order, common y-axis
Both use final test-set performance numbers.
"""

import pandas as pd, numpy as np, matplotlib, os, joblib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

with open(f"{PROCESSED}/feature_list_primary_22.txt") as f:
    FEATURES = [l.strip() for l in f if l.strip()]
OUTCOME = "diabetes_prev"

test  = pd.read_csv(f"{PROCESSED}/test.csv", dtype={"fips":str})
X_test = test[FEATURES].values
y_test = test[OUTCOME].values

# Model order matches Fig 1
MODEL_ORDER = ["LightGBM", "XGBoost", "Random Forest", "Elastic Net"]
COLORS = {
    "LightGBM":     "#e74c3c",
    "XGBoost":      "#e67e22",
    "Random Forest":"#3498db",
    "Elastic Net":  "#95a5a6",
}
models = {
    "LightGBM":     joblib.load("models/lightgbm.pkl"),
    "XGBoost":      joblib.load("models/xgboost.pkl"),
    "Random Forest":joblib.load("models/random_forest.pkl"),
    "Elastic Net":  joblib.load("models/elastic_net.pkl"),
}

# Get predictions and metrics
preds   = {}
metrics = {}
for name in MODEL_ORDER:
    yp = models[name].predict(X_test)
    preds[name] = yp
    metrics[name] = {
        "rmse": np.sqrt(mean_squared_error(y_test, yp)),
        "r2":   r2_score(y_test, yp),
    }
    print(f"{name}: RMSE={metrics[name]['rmse']:.3f}, R2={metrics[name]['r2']:.3f}")

# Common axis limits for observed vs predicted
all_vals = np.concatenate([y_test] + list(preds.values()))
ax_min = max(4.0, all_vals.min() - 0.5)
ax_max = min(25.0, all_vals.max() + 0.5)
print(f"Common axis limits: {ax_min:.1f} to {ax_max:.1f}")

# Common y-axis for residuals
all_res = np.concatenate([y_test - preds[m] for m in MODEL_ORDER])
res_min = max(-3.0, np.percentile(all_res, 0.5) - 0.2)
res_max = min(5.0,  np.percentile(all_res, 99.5) + 0.2)
print(f"Common residual y-axis: {res_min:.1f} to {res_max:.1f}")

# ── ESM_2: Observed vs predicted ─────────────────────────────
print("\nGenerating ESM_2...")
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()
for ax, name in zip(axes, MODEL_ORDER):
    yp  = preds[name]
    r2  = metrics[name]["r2"]
    ax.scatter(y_test, yp, alpha=0.4, s=14,
               color=COLORS[name], edgecolors="none")
    ax.plot([ax_min, ax_max], [ax_min, ax_max],
            "k--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed diagnosed diabetes prevalence (%)",
                  fontsize=12)
    ax.set_ylabel("Predicted diagnosed diabetes prevalence (%)",
                  fontsize=12)
    ax.set_title("")
    # Primary model gets black outline annotation
    box_color = "black" if name == "LightGBM" else "white"
    box_lw    = 1.5    if name == "LightGBM" else 0.8
    ax.annotate(
        f"{name}\n$R^2 = {r2:.3f}$",
        xy=(0.05, 0.93), xycoords="axes fraction",
        fontsize=12, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=box_color, linewidth=box_lw, alpha=0.9))
    ax.legend(fontsize=10, loc="lower right")

plt.tight_layout()
fig.savefig(f"{FIGURES}/ESM_2.tif", dpi=300, bbox_inches="tight",
            facecolor="white", pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(f"{FIGURES}/ESM_2.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(f"{FIGURES}/ESM_2.pdf", bbox_inches="tight", facecolor="white")
print("  Saved ESM_2")
plt.close(fig)

# ── ESM_3: Residual plots ─────────────────────────────────────
print("Generating ESM_3...")
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()
for ax, name in zip(axes, MODEL_ORDER):
    yp  = preds[name]
    res = y_test - yp
    rmse = metrics[name]["rmse"]
    r2   = metrics[name]["r2"]
    ax.scatter(yp, res, alpha=0.4, s=14,
               color=COLORS[name], edgecolors="none")
    ax.axhline(0, color="#c0392b", linewidth=1.5, linestyle="--")
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(res_min, res_max)
    ax.set_xlabel("Predicted diagnosed diabetes prevalence (%)",
                  fontsize=12)
    ax.set_ylabel("Residual (observed minus predicted, pp)",
                  fontsize=12)
    ax.set_title("")
    box_color = "black" if name == "LightGBM" else "white"
    box_lw    = 1.5    if name == "LightGBM" else 0.8
    ax.annotate(
        f"{name}\nRMSE = {rmse:.3f},  $R^2 = {r2:.3f}$",
        xy=(0.05, 0.93), xycoords="axes fraction",
        fontsize=12, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=box_color, linewidth=box_lw, alpha=0.9))

plt.tight_layout()
fig.savefig(f"{FIGURES}/ESM_3.tif", dpi=300, bbox_inches="tight",
            facecolor="white", pil_kwargs={"compression":"tiff_lzw"})
fig.savefig(f"{FIGURES}/ESM_3.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(f"{FIGURES}/ESM_3.pdf", bbox_inches="tight", facecolor="white")
print("  Saved ESM_3")
plt.close(fig)

print("\nDone. ESM_2 and ESM_3 complete.")
print("Model order: LightGBM, XGBoost, Random Forest, Elastic Net")
print("Common axis limits applied to both figures.")
