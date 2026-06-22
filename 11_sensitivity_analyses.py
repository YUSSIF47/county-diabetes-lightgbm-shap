"""
11_sensitivity_analyses.py
==========================
Runs nine sensitivity analyses using the primary LightGBM architecture and
hyperparameters selected by validation-set RMSE.
"""
import os
import joblib
import shap
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

PROCESSED = "data/processed"
TABLES = "outputs/tables"
os.makedirs(TABLES, exist_ok=True)

with open(f"{PROCESSED}/feature_list_primary_22.txt") as f:
    FEATURES = [l.strip() for l in f if l.strip()]

OUTCOME = "diabetes_prev"
train = pd.read_csv(f"{PROCESSED}/train.csv", dtype={"fips": str})
val = pd.read_csv(f"{PROCESSED}/val.csv", dtype={"fips": str})
test = pd.read_csv(f"{PROCESSED}/test.csv", dtype={"fips": str})
train_us = pd.read_csv(f"{PROCESSED}/train_unscaled.csv", dtype={"fips": str})
test_us = pd.read_csv(f"{PROCESSED}/test_unscaled.csv", dtype={"fips": str})

y_train = train[OUTCOME].values
y_test = test[OUTCOME].values

LGB_PARAMS = dict(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=500,
    subsample=0.7,
    colsample_bytree=1.0,
    num_leaves=31,
    random_state=42,
    n_jobs=1,
    verbose=-1,
)

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate(y_true, y_pred, name):
    return {
        "Model": name,
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "MAPE": round(mape(y_true, y_pred), 2),
        "R2": round(r2_score(y_true, y_pred), 3),
    }

def fit_eval(X_train, X_test, name):
    model = LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train)
    return evaluate(y_test, model.predict(X_test), name)

health_behavior_clinical = [
    "physical_inactivity_prev", "obesity_prev", "smoking_prev", "hypertension_prev"
]
structural_contextual = [f for f in FEATURES if f not in health_behavior_clinical]

results = []
print("Running sensitivity analyses with LightGBM hyperparameters selected using validation-set RMSE")
print("=" * 72)

# Primary model for reference
results.append(fit_eval(train[FEATURES].values, test[FEATURES].values,
                        "Primary (LightGBM, 22 predictors)"))

# S1: structural/contextual predictors only
s1_feats = [f for f in structural_contextual if f in FEATURES]
results.append(fit_eval(train[s1_feats].values, test[s1_feats].values,
                        "S1: Structural/contextual predictors only"))

# S2: health-behavior and clinical covariates only
s2_feats = [f for f in health_behavior_clinical if f in FEATURES]
results.append(fit_eval(train[s2_feats].values, test[s2_feats].values,
                        "S2: Health-behavior/clinical covariates only"))

# S3: FECI excluded, preprocessing refitted
s3_feats = [f for f in FEATURES if f != "feci"]
imp_s3 = IterativeImputer(max_iter=10, random_state=42)
imp_s3.fit(train_us[s3_feats])
sc_s3 = MinMaxScaler()
X_tr_s3_imp = imp_s3.transform(train_us[s3_feats])
X_te_s3_imp = imp_s3.transform(test_us[s3_feats])
sc_s3.fit(X_tr_s3_imp)
X_tr_s3 = sc_s3.transform(X_tr_s3_imp)
X_te_s3 = sc_s3.transform(X_te_s3_imp)
results.append(fit_eval(X_tr_s3, X_te_s3, "S3: FECI excluded, preprocessing refitted"))

# S4: collinearity-reduced set from iterative VIF assessment
s4_remove = {"feci", "smoking_prev", "hypertension_prev"}
s4_feats = [f for f in FEATURES if f not in s4_remove]
results.append(fit_eval(train[s4_feats].values, test[s4_feats].values,
                        "S4: Collinearity-reduced predictor set"))

# S5: binary high-food-desert indicator replaces continuous food desert index
train_s5 = train.copy()
test_s5 = test.copy()
train_s5["fdi_binary_high"] = (train_us["food_desert_index"] >= 20).astype(int).values
test_s5["fdi_binary_high"] = (test_us["food_desert_index"] >= 20).astype(int).values
s5_feats = [f for f in FEATURES if f != "food_desert_index"] + ["fdi_binary_high"]
results.append(fit_eval(train_s5[s5_feats].values, test_s5[s5_feats].values,
                        "S5: Binary high-food-desert indicator"))

# S6: FECI excluded, primary preprocessing retained
s6_feats = [f for f in FEATURES if f != "feci"]
results.append(fit_eval(train[s6_feats].values, test[s6_feats].values,
                        "S6: FECI excluded, primary preprocessing retained"))

# S7: rurality excluded
s7_feats = [f for f in FEATURES if f != "rucc_code"]
results.append(fit_eval(train[s7_feats].values, test[s7_feats].values,
                        "S7: Rurality excluded"))

# S8: Census-region indicators added
train_s8 = train.copy()
test_s8 = test.copy()
for region in ["Midwest", "South", "West"]:
    train_s8[f"region_{region}"] = (train_s8["census_region"] == region).astype(int)
    test_s8[f"region_{region}"] = (test_s8["census_region"] == region).astype(int)
s8_feats = FEATURES + ["region_Midwest", "region_South", "region_West"]
results.append(fit_eval(train_s8[s8_feats].values, test_s8[s8_feats].values,
                        "S8: Census-region indicators added"))

# S9: top 10 training-partition SHAP predictors
primary_model = joblib.load("models/lightgbm.pkl")
explainer = shap.TreeExplainer(primary_model, feature_perturbation="tree_path_dependent")
train_shap = explainer.shap_values(train[FEATURES])
mean_abs_train = np.abs(train_shap).mean(axis=0)
top10_feats = [FEATURES[i] for i in np.argsort(mean_abs_train)[::-1][:10]]
print(f"S9 top 10 predictors: {top10_feats}")
results.append(fit_eval(train[top10_feats].values, test[top10_feats].values,
                        "S9: Top-10 training-ranked SHAP predictors"))

results_df = pd.DataFrame(results)
results_df.to_csv(f"{TABLES}/sensitivity_results.csv", index=False)
results_df.to_csv(f"{TABLES}/sensitivity_lgb.csv", index=False)
print("\nFinal sensitivity results:")
print(results_df.to_string(index=False))
print(f"\nSaved: {TABLES}/sensitivity_results.csv")
