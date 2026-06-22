"""
07_model_comparison_fixed.py
============================
Step 7: Compare all four models.
- Model SELECTION uses validation set RMSE only
- Test set is touched exactly once for final performance reporting
- This corrects the data leakage in the original script
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

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100

# ── Load validation set for model selection ───────────────────
val = pd.read_csv(f"{PROCESSED}/val.csv", dtype={"fips": str})
with open(f"{PROCESSED}/feature_list_primary_22.txt") as f:
    FEATURES = [line.strip() for line in f if line.strip()]
OUTCOME = "diabetes_prev"
X_val = val[FEATURES].values
y_val = val[OUTCOME].values

# ── Load test set for final reporting only ────────────────────
test   = pd.read_csv(f"{PROCESSED}/test.csv", dtype={"fips": str})
X_test = test[FEATURES].values
y_test = test[OUTCOME].values

# ── Load all models ───────────────────────────────────────────
models = {
    "Elastic Net":   joblib.load("models/elastic_net.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost":       joblib.load("models/xgboost.pkl"),
    "LightGBM":      joblib.load("models/lightgbm.pkl"),
}

# ── STEP 1: Select best tree model using VALIDATION set only ──
print("="*60)
print("MODEL SELECTION — validation set RMSE (test set not used)")
print("="*60)
tree_models = ["Random Forest", "XGBoost", "LightGBM"]
val_rmse = {}
print(f"{'Model':<20} {'Val RMSE':>10}")
print("-"*32)
for name in tree_models:
    yp_val = models[name].predict(X_val)
    val_rmse[name] = np.sqrt(mean_squared_error(y_val, yp_val))
    print(f"{name:<20} {val_rmse[name]:>10.4f}")

best_tree = min(tree_models, key=lambda n: val_rmse[n])
print(f"\nBest tree-based model (validation): {best_tree}")
print("Test set has NOT been used for model selection.")
with open(f"{PROCESSED}/best_tree_model.txt", "w") as f:
    f.write(best_tree)

# ── STEP 2: Evaluate all models on test set (reporting only) ──
print("\n" + "="*60)
print("FINAL PERFORMANCE — test set (reported in manuscript)")
print("="*60)
results     = {}
predictions = {}
print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'R2':>8}")
print("-"*56)
for name, model in models.items():
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    mp   = mape(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    results[name] = {"RMSE": rmse, "MAE": mae, "MAPE": mp, "R2": r2}
    sel  = " <-- SELECTED (val)" if name == best_tree else ""
    print(f"{name:<20} {rmse:>8.4f} {mae:>8.4f} {mp:>8.2f} {r2:>8.4f}{sel}")

# ── Bootstrap CIs on test set ─────────────────────────────────
print("\nBootstrap 95% CIs (1000 iterations)...")
rng          = np.random.default_rng(42)
boot_results = {name: {"RMSE": [], "MAE": []} for name in models}
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
        "model":          name,
        "rmse":           round(results[name]["RMSE"], 4),
        "rmse_ci_lower":  round(rmse_ci[0], 4),
        "rmse_ci_upper":  round(rmse_ci[1], 4),
        "mae":            round(results[name]["MAE"], 4),
        "mae_ci_lower":   round(mae_ci[0], 4),
        "mae_ci_upper":   round(mae_ci[1], 4),
        "mape":           round(results[name]["MAPE"], 2),
        "r2":             round(results[name]["R2"], 4),
        "val_rmse":       round(val_rmse.get(name, float("nan")), 4),
        "selected_by_val": name == best_tree,
    })
perf_df = pd.DataFrame(perf_rows).sort_values("rmse")
perf_df.to_csv(f"{TABLES}/model_performance.csv", index=False)
print(perf_df[["model","val_rmse","rmse","mae","mape","r2"]].to_string(index=False))

# ── Wilcoxon pairwise ─────────────────────────────────────────
print("\nWilcoxon signed-rank pairwise comparisons:")
model_names   = list(models.keys())
abs_errors    = {n: np.abs(y_test - predictions[n]) for n in model_names}
pairwise_rows = []
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        n1, n2     = model_names[i], model_names[j]
        stat, pval = stats.wilcoxon(abs_errors[n1], abs_errors[n2])
        better     = n1 if results[n1]["MAE"] < results[n2]["MAE"] else n2
        sig        = "**" if pval<0.01 else ("*" if pval<0.05 else "ns")
        pairwise_rows.append({
            "model_a": n1, "model_b": n2,
            "wilcoxon_stat": round(stat, 2),
            "p_value":       round(pval, 4),
            "better_model":  better,
        })
        print(f"  {n1} vs {n2}: W={stat:.1f}, p={pval:.4f} {sig}")
pd.DataFrame(pairwise_rows).to_csv(
    f"{TABLES}/pairwise_comparison.csv", index=False)

print(f"\nModel selection: {best_tree} (chosen by validation RMSE = {val_rmse[best_tree]:.4f})")
print(f"Test-set RMSE for {best_tree}: {results[best_tree]['RMSE']:.4f}")
print("\nScript complete. Test set was used ONLY for final reporting.")
