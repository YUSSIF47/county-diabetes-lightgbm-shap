"""
07_model_comparison.py  (v2 — high-resolution figures)
=======================================================
Step 7: Compare all four models on the test set.
        Saves all figures as PDF + PNG (300 DPI) + TIF (600 DPI).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib, os

PROCESSED = "data/processed"
TABLES    = "outputs/tables"
FIGURES   = "outputs/figures"
os.makedirs(TABLES,  exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ── Figure saving utility ─────────────────────────────────────
def save_fig(fig, name, dpi_png=300, dpi_tif=600):
    path = f"{FIGURES}/{name}"
    fig.savefig(f"{path}.pdf", format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{path}.png", format="png", dpi=dpi_png, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{path}.tif", format="tiff", dpi=dpi_tif, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {name} [.pdf | .png {dpi_png}dpi | .tif {dpi_tif}dpi]")
    plt.close(fig)

# ── Load data ─────────────────────────────────────────────────
test = pd.read_csv(f"{PROCESSED}/test.csv", dtype={"fips": str})
with open(f"{PROCESSED}/feature_list.txt") as f:
    FEATURES = [line.strip() for line in f if line.strip()]

OUTCOME = "diabetes_prev"
X_test  = test[FEATURES].values
y_test  = test[OUTCOME].values

# ── Load all models ───────────────────────────────────────────
models = {
    "Elastic Net":   joblib.load("models/elastic_net.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost":       joblib.load("models/xgboost.pkl"),
    "LightGBM":      joblib.load("models/lightgbm.pkl"),
}

# ── Performance metrics ───────────────────────────────────────
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100

results     = {}
predictions = {}
print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'R2':>8}")
print("-" * 56)
for name, model in models.items():
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    mp   = mape(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    results[name] = {"RMSE": rmse, "MAE": mae, "MAPE": mp, "R2": r2}
    print(f"{name:<20} {rmse:>8.4f} {mae:>8.4f} {mp:>8.2f} {r2:>8.4f}")

# ── Bootstrap CIs ─────────────────────────────────────────────
print("\nBootstrap 95% CIs (1000 iterations)...")
rng         = np.random.default_rng(42)
boot_results= {name: {"RMSE": [], "MAE": []} for name in models}
for _ in range(1000):
    idx = rng.choice(len(y_test), size=len(y_test), replace=True)
    for name, y_pred in predictions.items():
        boot_results[name]["RMSE"].append(
            np.sqrt(mean_squared_error(y_test[idx], y_pred[idx])))
        boot_results[name]["MAE"].append(
            mean_absolute_error(y_test[idx], y_pred[idx]))

perf_rows = []
for name in models:
    rmse_ci = np.percentile(boot_results[name]["RMSE"], [2.5, 97.5])
    mae_ci  = np.percentile(boot_results[name]["MAE"],  [2.5, 97.5])
    perf_rows.append({
        "model": name,
        "rmse": round(results[name]["RMSE"], 4),
        "rmse_ci_lower": round(rmse_ci[0], 4),
        "rmse_ci_upper": round(rmse_ci[1], 4),
        "mae":  round(results[name]["MAE"], 4),
        "mae_ci_lower":  round(mae_ci[0], 4),
        "mae_ci_upper":  round(mae_ci[1], 4),
        "mape": round(results[name]["MAPE"], 2),
        "r2":   round(results[name]["R2"], 4),
    })
perf_df = pd.DataFrame(perf_rows).sort_values("rmse")
perf_df.to_csv(f"{TABLES}/model_performance.csv", index=False)
print(perf_df[["model","rmse","mae","mape","r2"]].to_string(index=False))

# ── Wilcoxon pairwise ─────────────────────────────────────────
print("\nWilcoxon signed-rank pairwise comparisons:")
model_names  = list(models.keys())
abs_errors   = {n: np.abs(y_test - predictions[n]) for n in model_names}
pairwise_rows= []
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        n1, n2   = model_names[i], model_names[j]
        stat, pval = stats.wilcoxon(abs_errors[n1], abs_errors[n2])
        better   = n1 if results[n1]["MAE"] < results[n2]["MAE"] else n2
        sig      = "**" if pval<0.01 else ("*" if pval<0.05 else "ns")
        pairwise_rows.append({
            "model_a": n1, "model_b": n2,
            "wilcoxon_stat": round(stat,2),
            "p_value": round(pval,4),
            "better_model": better
        })
        print(f"  {n1} vs {n2}: W={stat:.1f}, p={pval:.4f} {sig}")
pd.DataFrame(pairwise_rows).to_csv(f"{TABLES}/pairwise_comparison.csv", index=False)

# ── Best tree model ───────────────────────────────────────────
tree_models  = ["Random Forest","XGBoost","LightGBM"]
best_tree    = min(tree_models, key=lambda n: results[n]["RMSE"])
print(f"\nBest tree-based model: {best_tree}")
with open(f"{PROCESSED}/best_tree_model.txt","w") as f:
    f.write(best_tree)

# ── Figure 1: Performance comparison bar chart ────────────────
model_order = perf_df["model"].tolist()
colors      = ["#e74c3c","#e67e22","#2ecc71","#3498db"]
rmse_vals   = [results[m]["RMSE"] for m in model_order]
r2_vals     = [results[m]["R2"]   for m in model_order]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, vals, label, better in zip(
    axes,
    [rmse_vals, r2_vals],
    ["RMSE (lower = better)", "R² (higher = better)"],
    [True, False]
):
    bars = ax.barh(model_order[::-1], vals[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_xlabel(label, fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10)

axes[0].set_title("Model Comparison: RMSE\n(Test Set)", fontsize=12)
axes[1].set_title("Model Comparison: R²\n(Test Set)", fontsize=12)
plt.suptitle("Model Performance on Held-Out Test Set (n = 444 counties)",
             fontsize=13, y=1.02)
plt.tight_layout()
save_fig(fig, "fig_model_comparison")

# ── Figure 2: Residual plots ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()
for ax, (name, y_pred) in zip(axes, predictions.items()):
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.35, s=12, color="#2c3e50",
               edgecolors="none")
    ax.axhline(0, color="#e74c3c", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Predicted T2DM prevalence (%)", fontsize=11)
    ax.set_ylabel("Residual (observed − predicted)", fontsize=11)
    ax.set_title(f"{name}\nRMSE={results[name]['RMSE']:.3f}, "
                 f"R²={results[name]['R2']:.3f}", fontsize=11)
plt.suptitle("Residual Plots — All Models (Held-Out Test Set)",
             fontsize=13, y=1.01)
plt.tight_layout()
save_fig(fig, "fig_residuals")

# ── Figure 3: Observed vs predicted ──────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()
for ax, (name, y_pred) in zip(axes, predictions.items()):
    ax.scatter(y_test, y_pred, alpha=0.35, s=12,
               color="#2980b9", edgecolors="none")
    lims = [min(y_test.min(), y_pred.min())-0.5,
            max(y_test.max(), y_pred.max())+0.5]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Observed T2DM prevalence (%)", fontsize=11)
    ax.set_ylabel("Predicted T2DM prevalence (%)", fontsize=11)
    ax.set_title(f"{name}\nR²={results[name]['R2']:.3f}", fontsize=11)
    ax.legend(fontsize=9)
plt.suptitle("Observed vs Predicted — All Models (Held-Out Test Set)",
             fontsize=13, y=1.01)
plt.tight_layout()
save_fig(fig, "fig_observed_vs_predicted")

print(f"\nAll figures saved to {FIGURES}/")
print("Each figure: .pdf | .png (300 DPI) | .tif (600 DPI)")
print("Model comparison complete.")
