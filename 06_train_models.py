"""
06_train_models.py
==================
Step 6: Train Elastic Net, Random Forest, XGBoost, and LightGBM
        with hyperparameter tuning via cross-validation.

Input:  data/processed/train.csv
        data/processed/val.csv
        data/processed/feature_list_primary_22.txt
Output: models/*.pkl (trained models)
        outputs/tables/hyperparameter_results.csv
"""

import pandas as pd
import numpy as np
import json, os, joblib
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

PROCESSED = "data/processed"
TABLES = "outputs/tables"
os.makedirs("models", exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────
train = pd.read_csv(f"{PROCESSED}/train.csv", dtype={"fips": str})
val   = pd.read_csv(f"{PROCESSED}/val.csv",   dtype={"fips": str})

with open(f"{PROCESSED}/feature_list_primary_22.txt") as f:
    FEATURES = [line.strip() for line in f if line.strip()]

OUTCOME = "diabetes_prev"
X_train = train[FEATURES].values
y_train = train[OUTCOME].values
X_val   = val[FEATURES].values
y_val   = val[OUTCOME].values

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Features: {len(FEATURES)}")

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ════════════════════════════════════════════════════════════
# Model 1: Elastic Net
# ════════════════════════════════════════════════════════════
print("\n[1/4] Training Elastic Net...")
enet = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    alphas=np.logspace(-4, 1, 50),
    cv=cv,
    max_iter=10000,
    random_state=42,
    n_jobs=1
)
enet.fit(X_train, y_train)
joblib.dump(enet, "models/elastic_net.pkl")
print(f"  Best alpha: {enet.alpha_:.5f}, l1_ratio: {enet.l1_ratio_:.2f}")
print(f"  Val predictions range: {enet.predict(X_val).min():.2f} - {enet.predict(X_val).max():.2f}")

# ════════════════════════════════════════════════════════════
# Model 2: Random Forest
# ════════════════════════════════════════════════════════════
print("\n[2/4] Training Random Forest...")
rf_grid = {
    "n_estimators": [100, 300],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [2, 5],
    "max_features": [0.5, 0.7, "sqrt"],
}
rf_base = RandomForestRegressor(random_state=42, n_jobs=1)
rf_search = GridSearchCV(rf_base, rf_grid, cv=cv,
                         scoring="neg_root_mean_squared_error",
                         n_jobs=1, verbose=0)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
joblib.dump(rf_best, "models/random_forest.pkl")
print(f"  Best params: {rf_search.best_params_}")
print(f"  CV RMSE: {-rf_search.best_score_:.4f}")

# ════════════════════════════════════════════════════════════
# Model 3: XGBoost
# ════════════════════════════════════════════════════════════
print("\n[3/4] Training XGBoost...")
xgb_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300, 500],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}
xgb_base = XGBRegressor(
    random_state=42,
    n_jobs=1,
    tree_method="hist",
    verbosity=0
)
xgb_search = GridSearchCV(xgb_base, xgb_grid, cv=cv,
                           scoring="neg_root_mean_squared_error",
                           n_jobs=1, verbose=0)
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_
joblib.dump(xgb_best, "models/xgboost.pkl")
print(f"  Best params: {xgb_search.best_params_}")
print(f"  CV RMSE: {-xgb_search.best_score_:.4f}")

# ════════════════════════════════════════════════════════════
# Model 4: LightGBM
# ════════════════════════════════════════════════════════════
print("\n[4/4] Training LightGBM...")
lgb_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300, 500],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "num_leaves": [31, 63, 127],
}
lgb_base = LGBMRegressor(random_state=42, n_jobs=1, verbose=-1)
lgb_search = GridSearchCV(lgb_base, lgb_grid, cv=cv,
                           scoring="neg_root_mean_squared_error",
                           n_jobs=1, verbose=0)
lgb_search.fit(X_train, y_train)
lgb_best = lgb_search.best_estimator_
joblib.dump(lgb_best, "models/lightgbm.pkl")
print(f"  Best params: {lgb_search.best_params_}")
print(f"  CV RMSE: {-lgb_search.best_score_:.4f}")

# ── Save best hyperparameters table ──────────────────────────
hp_results = []
for name, search in [("XGBoost", xgb_search), ("LightGBM", lgb_search),
                     ("Random Forest", rf_search)]:
    row = {"model": name, "cv_rmse": round(-search.best_score_, 4)}
    row.update(search.best_params_)
    hp_results.append(row)

pd.DataFrame(hp_results).to_csv(f"{TABLES}/hyperparameter_results.csv", index=False)
print(f"\nHyperparameter results saved.")
print("\nAll models trained and saved to models/")
