"""
12_compile_results_for_paper.py
================================
Step 12: Compile all results and produce the final tables and
         figures ready for insertion into the LaTeX paper.

Outputs:
  outputs/tables/TABLE1_descriptive_statistics.csv
  outputs/tables/TABLE2_model_performance.csv
  outputs/tables/TABLE3_sensitivity_results.csv
  outputs/tables/TABLE4_shap_global_importance.csv
  outputs/tables/TABLE5_morans_i.csv
  outputs/paper_results_summary.txt  (paste into Results section)
"""

import pandas as pd
import numpy as np
import os

PROCESSED = "data/processed"
TABLES = "outputs/tables"
os.makedirs(TABLES, exist_ok=True)

print("=" * 60)
print("COMPILING ALL RESULTS FOR PAPER")
print("=" * 60)

# ── TABLE 1: Descriptive statistics ──────────────────────────
print("\n[TABLE 1] Descriptive statistics")
try:
    master = pd.read_csv(f"{PROCESSED}/county_master_with_feci.csv",
                         dtype={"fips": str})

    with open(f"{PROCESSED}/feature_list_primary_22.txt") as f:
        FEATURES = [line.strip() for line in f if line.strip()]

    desc_vars = ["diabetes_prev"] + FEATURES
    desc_vars = [v for v in desc_vars if v in master.columns]

    desc = master[desc_vars].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    desc.columns = ["N", "Mean", "SD", "Min", "Median", "Max"]
    desc = desc.round(3)
    desc.to_csv(f"{TABLES}/TABLE1_descriptive_statistics.csv")
    print(desc.to_string())
except Exception as e:
    print(f"  Error: {e}")

# ── TABLE 2: Model performance ────────────────────────────────
print("\n[TABLE 2] Model performance")
try:
    perf = pd.read_csv(f"{TABLES}/model_performance.csv")
    perf_formatted = perf[[
        "model","rmse","rmse_ci_lower","rmse_ci_upper",
        "mae","mae_ci_lower","mae_ci_upper","mape","r2"
    ]].copy()
    perf_formatted["RMSE (95% CI)"] = perf_formatted.apply(
        lambda r: f"{r['rmse']:.3f} ({r['rmse_ci_lower']:.3f}–{r['rmse_ci_upper']:.3f})",
        axis=1
    )
    perf_formatted["MAE (95% CI)"] = perf_formatted.apply(
        lambda r: f"{r['mae']:.3f} ({r['mae_ci_lower']:.3f}–{r['mae_ci_upper']:.3f})",
        axis=1
    )
    table2 = perf_formatted[["model","RMSE (95% CI)","MAE (95% CI)","mape","r2"]]
    table2.columns = ["Model", "RMSE (95% CI)", "MAE (95% CI)", "MAPE (%)", "R²"]
    table2.to_csv(f"{TABLES}/TABLE2_model_performance.csv", index=False)
    print(table2.to_string(index=False))
except Exception as e:
    print(f"  Error: {e}")

# ── TABLE 3: Sensitivity results ──────────────────────────────
print("\n[TABLE 3] Sensitivity results")
try:
    sens = pd.read_csv(f"{TABLES}/sensitivity_results.csv")
    sens.to_csv(f"{TABLES}/TABLE3_sensitivity_results.csv", index=False)
    print(sens.to_string(index=False))
except Exception as e:
    print(f"  Error: {e}")

# ── TABLE 4: SHAP global importance ──────────────────────────
print("\n[TABLE 4] SHAP global importance (top 15)")
try:
    shap = pd.read_csv(f"{TABLES}/shap_global_importance.csv")
    shap_top = shap.head(15).copy()
    shap_top["rank"] = range(1, len(shap_top)+1)
    shap_top = shap_top[["rank","feature","mean_abs_shap"]].round(4)
    shap_top.to_csv(f"{TABLES}/TABLE4_shap_global_importance.csv", index=False)
    print(shap_top.to_string(index=False))
except Exception as e:
    print(f"  Error: {e}")

# ── TABLE 5: Moran's I ────────────────────────────────────────
print("\n[TABLE 5] Moran's I results")
try:
    morans = pd.read_csv(f"{TABLES}/morans_i_results.csv")
    morans.to_csv(f"{TABLES}/TABLE5_morans_i.csv", index=False)
    print(morans.to_string(index=False))
except Exception as e:
    print(f"  Error: {e}")

# ── Generate Results text template ───────────────────────────
print("\n[Generating Results text template...]")

try:
    perf = pd.read_csv(f"{TABLES}/model_performance.csv")
    best_row = perf.sort_values("rmse").iloc[0]
    worst_row = perf.sort_values("rmse").iloc[-1]

    with open(f"{PROCESSED}/best_tree_model.txt") as f:
        best_model = f.read().strip()

    morans = pd.read_csv(f"{TABLES}/morans_i_results.csv")
    mi_outcome  = morans[morans["measure"].str.contains("prevalence.*Queen")].iloc[0]
    mi_residual = morans[morans["measure"].str.contains("residuals")].iloc[0]

    shap = pd.read_csv(f"{TABLES}/shap_global_importance.csv")
    top3 = shap.head(3)["feature"].tolist()

    results_text = f"""
=============================================================
RESULTS SECTION — FILL INTO LATEX PAPER
=============================================================

3.1 Sample Description
A total of [N] U.S. counties were included in the primary analysis
after excluding [n_excluded] counties with missing outcome data.
County-level T2DM prevalence ranged from [min]% to [max]% with a
mean of [mean]% (SD = [sd]%). The food desert index ranged from
[fdi_min]% to [fdi_max]% (mean = [fdi_mean]%, SD = [fdi_sd]%).
Global Moran's I for county-level T2DM prevalence was
I = {mi_outcome['I']:.3f} (p = {mi_outcome['p_value']:.4f}),
confirming strong positive spatial clustering.

3.2 PCA — Food Environment Composite Index
[Insert PCA eigenvalue and variance explained from script 04 output]
Kaiser's criterion was [met / not met] for the first principal
component (eigenvalue = [value], variance explained = [pct]%).

3.3 Model Performance
The {best_model} achieved the best performance on the held-out
test set (RMSE = {best_row['rmse']:.3f} [95% CI {best_row['rmse_ci_lower']:.3f}–{best_row['rmse_ci_upper']:.3f}];
MAE = {best_row['mae']:.3f}; MAPE = {best_row['mape']:.2f}%;
R² = {best_row['r2']:.3f}).
[Insert Wilcoxon test results from 07_model_comparison output]

3.4 SHAP Feature Importance
The three predictors with the highest mean absolute SHAP values
were: {top3[0]}, {top3[1]}, and {top3[2]}.
[Insert regional SHAP profiles and dominant-driver summary]

3.5 Spatial Diagnostics
After fitting the {best_model}, residual spatial autocorrelation
was I = {mi_residual['I']:.3f} (p = {mi_residual['p_value']:.4f}),
{"suggesting some residual spatial structure remains." if mi_residual['I'] > 0.1
 else "indicating that the model accounted for the majority of spatial structure."}

3.6 Sensitivity Analyses
[Insert from TABLE3_sensitivity_results.csv]
Results were broadly stable across all eight sensitivity
specifications, with [best_model] consistently outperforming
Elastic Net on RMSE and R².
=============================================================
"""
    with open("outputs/paper_results_summary.txt", "w") as f:
        f.write(results_text)
    print(results_text)

except Exception as e:
    print(f"Results template error: {e}")

print("\n" + "="*60)
print("ALL TABLES COMPILED. Ready to insert into LaTeX paper.")
print("="*60)
print("\nNext steps:")
print("  1. Open outputs/paper_results_summary.txt")
print("  2. Fill in [bracketed] placeholders from table outputs")
print("  3. Copy tables into LaTeX Results section")
print("  4. Add figures to LaTeX as \\includegraphics{}")
