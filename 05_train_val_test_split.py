"""
05_train_val_test_split.py  (FINAL VERSION — fixed region bug)
==============================================================
Stratified 70/15/15 split by Census region.
'Other' region assigned to West to avoid stratification error.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib, json, os

PROCESSED = "data/processed"
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

master = pd.read_csv(f"{PROCESSED}/county_master_with_feci.csv", dtype={"fips": str})
print(f"Loaded: {master.shape}")

OUTCOME = "diabetes_prev"

northeast = {"09","23","25","33","44","50","34","36","42"}
midwest   = {"17","18","26","39","55","19","20","27","29","31","38","46"}
south     = {"10","11","12","13","24","37","45","51","54","01","21","28","47","05","22","40","48"}
west      = {"04","08","16","30","32","35","49","56","02","06","15","41","53"}

def assign_region(fips):
    state = str(fips).zfill(5)[:2]
    if state in northeast: return "Northeast"
    if state in midwest:   return "Midwest"
    if state in south:     return "South"
    if state in west:      return "West"
    return "West"  # assign Other to West

master["census_region"] = master["fips"].apply(assign_region)
print(f"Region distribution:\n{master['census_region'].value_counts().to_string()}")

FOOD_ACCESS  = ["food_desert_index","fdi_binary_high"]
FOOD_ENV     = ["fastfood_density","grocery_density","snap_rate","food_insecurity_rate","feci"]
SOCIOECONOMIC= ["poverty_rate","median_income","unemployment_rate","pct_college_edu"]
DEMOGRAPHIC  = ["pct_white","pct_black","pct_hispanic","pct_age65plus"]
OCCUPATIONAL = ["pct_agri_sector","pct_mfg_sector","pct_foodsvc_sector"]
RURALITY     = ["rucc_code","rural_flag"]
HEALTH_BEHAV = ["obesity_prev","physical_inactivity_prev","hypertension_prev","smoking_prev"]

ALL_FEATURES = (FOOD_ACCESS + FOOD_ENV + SOCIOECONOMIC +
                DEMOGRAPHIC + OCCUPATIONAL + RURALITY + HEALTH_BEHAV)
ALL_FEATURES = [f for f in ALL_FEATURES if f in master.columns]
print(f"\nFeatures ({len(ALL_FEATURES)}): {ALL_FEATURES}")

with open(f"{PROCESSED}/feature_list.txt","w") as f:
    for feat in ALL_FEATURES: f.write(feat+"\n")

feature_groups = {
    "structural_only": [f for f in (FOOD_ACCESS+FOOD_ENV+SOCIOECONOMIC+
                        DEMOGRAPHIC+OCCUPATIONAL+RURALITY) if f in master.columns],
    "health_behav_only": [f for f in HEALTH_BEHAV if f in master.columns],
    "all_features": ALL_FEATURES,
}
with open(f"{PROCESSED}/feature_groups.json","w") as f:
    json.dump(feature_groups, f, indent=2)

model_df = master[["fips","census_region",OUTCOME]+ALL_FEATURES].copy()
model_df = model_df.dropna(subset=[OUTCOME])
print(f"Modelling dataframe: {model_df.shape}")

train_val, test = train_test_split(model_df, test_size=0.15,
    stratify=model_df["census_region"], random_state=42)
val_size = 0.15/0.85
train, val = train_test_split(train_val, test_size=val_size,
    stratify=train_val["census_region"], random_state=42)

print(f"\nSplit: Train={len(train)}, Val={len(val)}, Test={len(test)}")

print("\nFitting MICE imputer...")
imputer = IterativeImputer(max_iter=10, random_state=42)
imputer.fit(train[ALL_FEATURES])
train[ALL_FEATURES] = imputer.transform(train[ALL_FEATURES])
val[ALL_FEATURES]   = imputer.transform(val[ALL_FEATURES])
test[ALL_FEATURES]  = imputer.transform(test[ALL_FEATURES])
print(f"Missing after MICE: {train[ALL_FEATURES].isnull().sum().sum()}")

scaler = MinMaxScaler()
scaler.fit(train[ALL_FEATURES])
train_s = train.copy(); val_s = val.copy(); test_s = test.copy()
train_s[ALL_FEATURES] = scaler.transform(train[ALL_FEATURES])
val_s[ALL_FEATURES]   = scaler.transform(val[ALL_FEATURES])
test_s[ALL_FEATURES]  = scaler.transform(test[ALL_FEATURES])

joblib.dump(imputer, "models/mice_imputer.pkl")
joblib.dump(scaler,  "models/minmax_scaler.pkl")
train_s.to_csv(f"{PROCESSED}/train.csv", index=False)
val_s.to_csv(f"{PROCESSED}/val.csv",     index=False)
test_s.to_csv(f"{PROCESSED}/test.csv",   index=False)
train.to_csv(f"{PROCESSED}/train_unscaled.csv", index=False)
val.to_csv(f"{PROCESSED}/val_unscaled.csv",     index=False)
test.to_csv(f"{PROCESSED}/test_unscaled.csv",   index=False)

print(f"All partitions saved to {PROCESSED}/")
print("Script 05 complete.")
