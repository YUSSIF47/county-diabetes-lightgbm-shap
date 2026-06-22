"""
region_holdout_lgb.py
=====================
Region-held-out cross-validation using primary LightGBM model.
- Preprocessing (imputation + scaling) refitted within each fold
- LightGBM hyperparameters selected using validation-set RMSE:
  max_depth=3, learning_rate=0.1, n_estimators=500,
  subsample=0.7, colsample_bytree=1.0, num_leaves=31
- Uses full analytic dataset in held-out-region folds; preprocessing is refitted within each fold
"""

import pandas as pd, numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

PROCESSED = "data/processed"
with open(f"{PROCESSED}/feature_list_primary_22.txt") as f:
    FEATURES = [l.strip() for l in f if l.strip()]
OUTCOME = "diabetes_prev"

train = pd.read_csv(f"{PROCESSED}/train_unscaled.csv", dtype={"fips":str})
val   = pd.read_csv(f"{PROCESSED}/val_unscaled.csv",   dtype={"fips":str})
test  = pd.read_csv(f"{PROCESSED}/test_unscaled.csv",  dtype={"fips":str})
all_df = pd.concat([train, val, test], ignore_index=True)

print("Region-held-out cross-validation — LightGBM primary model")
print("Selected hyperparameters: max_depth=3, lr=0.1, n_estimators=500,")
print("  subsample=0.7, colsample_bytree=1.0, num_leaves=31")
print("Preprocessing refitted within each fold (no leakage)")
print("="*65)

regions = ["Northeast", "Midwest", "South", "West"]
results = []

for held_out in regions:
    tr = all_df[all_df["census_region"] != held_out].copy()
    te = all_df[all_df["census_region"] == held_out].copy()

    # Preprocessing fitted on training fold only
    imp = IterativeImputer(max_iter=10, random_state=42)
    imp.fit(tr[FEATURES])
    scaler = MinMaxScaler()
    scaler.fit(imp.transform(tr[FEATURES]))
    X_tr = scaler.transform(imp.transform(tr[FEATURES]))
    X_te = scaler.transform(imp.transform(te[FEATURES]))
    y_tr = tr[OUTCOME].values
    y_te = te[OUTCOME].values

    # LightGBM with hyperparameters selected using validation-set RMSE
    m = LGBMRegressor(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=500,
        subsample=0.7,
        colsample_bytree=1.0,
        num_leaves=31,
        random_state=42,
        n_jobs=1,
        verbose=-1
    )
    m.fit(X_tr, y_tr)
    y_pred = m.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    mape = np.mean(np.abs((y_te - y_pred) / y_te)) * 100
    r2   = r2_score(y_te, y_pred)

    print(f"{held_out}: RMSE={rmse:.3f}  MAE={mae:.3f}  "
          f"MAPE={mape:.2f}%  R2={r2:.3f}  (n={len(te)})")
    results.append({
        "Region":     held_out,
        "n_counties": len(te),
        "RMSE":       round(rmse, 3),
        "MAE":        round(mae,  3),
        "MAPE":       round(mape, 2),
        "R2":         round(r2,   3),
    })

pd.DataFrame(results).to_csv(
    "outputs/tables/region_holdout_lgb.csv", index=False)
print("="*65)
print("Saved: outputs/tables/region_holdout_lgb.csv")
