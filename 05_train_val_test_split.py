"""
05_train_val_test_split.py
==========================
Creates a stratified 70/15/15 train/validation/test split by U.S. Census region.
The primary model uses the 22 predictors listed in feature_list_primary_22.txt.
Sensitivity-only variables are saved separately when available and are not used
for primary-model training.
"""
import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

PROCESSED = "data/processed"
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

master = pd.read_csv(f"{PROCESSED}/county_master_with_feci.csv", dtype={"fips": str})
print(f"Loaded: {master.shape}")

OUTCOME = "diabetes_prev"

northeast = {"09", "23", "25", "33", "44", "50", "34", "36", "42"}
midwest   = {"17", "18", "26", "39", "55", "19", "20", "27", "29", "31", "38", "46"}
south     = {"10", "11", "12", "13", "24", "37", "45", "51", "54", "01", "21", "28", "47", "05", "22", "40", "48"}
west      = {"04", "08", "16", "30", "32", "35", "49", "56", "02", "06", "15", "41", "53"}

def assign_region(fips):
    state = str(fips).zfill(5)[:2]
    if state in northeast:
        return "Northeast"
    if state in midwest:
        return "Midwest"
    if state in south:
        return "South"
    if state in west:
        return "West"
    return "West"  # assign any nonstandard/other code to West for stratification

master["census_region"] = master["fips"].apply(assign_region)
print(f"Region distribution:\n{master['census_region'].value_counts().to_string()}")

PRIMARY_FEATURES = ['physical_inactivity_prev', 'pct_white', 'poverty_rate', 'pct_black', 'food_insecurity_rate', 'smoking_prev', 'median_income', 'snap_rate', 'obesity_prev', 'pct_age65plus', 'pct_hispanic', 'hypertension_prev', 'pct_college_edu', 'food_desert_index', 'unemployment_rate', 'fastfood_density', 'feci', 'pct_agri_sector', 'pct_mfg_sector', 'pct_foodsvc_sector', 'grocery_density', 'rucc_code']
SENSITIVITY_ONLY = ['fdi_binary_high', 'rural_flag']

missing_primary = [f for f in PRIMARY_FEATURES if f not in master.columns]
if missing_primary:
    raise ValueError(f"Missing primary predictors: {missing_primary}")

available_sensitivity = [f for f in SENSITIVITY_ONLY if f in master.columns]
print(f"\nPrimary predictors ({len(PRIMARY_FEATURES)}): {PRIMARY_FEATURES}")
print(f"Sensitivity-only variables available: {available_sensitivity}")

with open(f"{PROCESSED}/feature_list_primary_22.txt", "w") as f:
    for feat in PRIMARY_FEATURES:
        f.write(feat + "\n")

with open(f"{PROCESSED}/feature_list_sensitivity_only.txt", "w") as f:
    for feat in available_sensitivity:
        f.write(feat + "\n")

health_behavior_clinical = [
    "obesity_prev", "physical_inactivity_prev", "hypertension_prev", "smoking_prev"
]
structural_contextual = [f for f in PRIMARY_FEATURES if f not in health_behavior_clinical]
feature_groups = {
    "primary_22": PRIMARY_FEATURES,
    "structural_contextual_only": structural_contextual,
    "health_behavior_clinical_only": [f for f in health_behavior_clinical if f in PRIMARY_FEATURES],
    "sensitivity_only": available_sensitivity,
}
with open(f"{PROCESSED}/feature_groups.json", "w") as f:
    json.dump(feature_groups, f, indent=2)

# Keep sensitivity-only variables in partition files if available, but fit imputation and
# scaling for the primary model using the 22 primary predictors only.
partition_columns = ["fips", "census_region", OUTCOME] + PRIMARY_FEATURES + available_sensitivity
model_df = master[partition_columns].copy().dropna(subset=[OUTCOME])
print(f"Modeling dataframe: {model_df.shape}")

train_val, test = train_test_split(
    model_df, test_size=0.15, stratify=model_df["census_region"], random_state=42
)
val_size = 0.15 / 0.85
train, val = train_test_split(
    train_val, test_size=val_size, stratify=train_val["census_region"], random_state=42
)
print(f"\nSplit: Train={len(train)}, Val={len(val)}, Test={len(test)}")

print("\nFitting iterative multivariate single imputer on the training partition...")
imputer = IterativeImputer(max_iter=10, random_state=42)
imputer.fit(train[PRIMARY_FEATURES])

train_imp = train.copy()
val_imp = val.copy()
test_imp = test.copy()
train_imp[PRIMARY_FEATURES] = imputer.transform(train[PRIMARY_FEATURES])
val_imp[PRIMARY_FEATURES] = imputer.transform(val[PRIMARY_FEATURES])
test_imp[PRIMARY_FEATURES] = imputer.transform(test[PRIMARY_FEATURES])
print(f"Missing after iterative imputation: {train_imp[PRIMARY_FEATURES].isnull().sum().sum()}")

scaler = MinMaxScaler()
scaler.fit(train_imp[PRIMARY_FEATURES])

train_scaled = train_imp.copy()
val_scaled = val_imp.copy()
test_scaled = test_imp.copy()
train_scaled[PRIMARY_FEATURES] = scaler.transform(train_imp[PRIMARY_FEATURES])
val_scaled[PRIMARY_FEATURES] = scaler.transform(val_imp[PRIMARY_FEATURES])
test_scaled[PRIMARY_FEATURES] = scaler.transform(test_imp[PRIMARY_FEATURES])

joblib.dump(imputer, "models/iterative_imputer_primary_22.pkl")
joblib.dump(scaler, "models/minmax_scaler_primary_22.pkl")

train_scaled.to_csv(f"{PROCESSED}/train.csv", index=False)
val_scaled.to_csv(f"{PROCESSED}/val.csv", index=False)
test_scaled.to_csv(f"{PROCESSED}/test.csv", index=False)
train_imp.to_csv(f"{PROCESSED}/train_unscaled.csv", index=False)
val_imp.to_csv(f"{PROCESSED}/val_unscaled.csv", index=False)
test_imp.to_csv(f"{PROCESSED}/test_unscaled.csv", index=False)

print(f"All partitions saved to {PROCESSED}/")
print("Script 05 complete.")
