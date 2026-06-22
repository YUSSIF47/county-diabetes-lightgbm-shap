"""
01_build_master_dataset.py  (FINAL VERSION for CRADLE)
=======================================================
Reads already-wide PLACES file + FEA clean CSV + ACS + RUCC.
Output: data/processed/county_master_pre_fdi.csv
"""
import pandas as pd
import numpy as np
import os

RAW       = "data/raw"
PROCESSED = "data/processed"
os.makedirs(PROCESSED, exist_ok=True)

def clean_fips(s):
    return s.astype(str).str.strip().str.zfill(5)

# PLACES (already wide)
print("Loading CDC PLACES 2025...")
places = pd.read_csv(f"{RAW}/CDC_PLACES_2025_county.csv", dtype={"fips": str})
places["fips"] = clean_fips(places["fips"])
places = places.drop_duplicates("fips")
print(f"  Counties: {len(places)}")

# Food Environment Atlas
print("Loading Food Environment Atlas...")
fea = pd.read_csv(f"{RAW}/USDA_FoodEnvAtlas_clean.csv", dtype={"fips": str})
fea["fips"] = clean_fips(fea["fips"])
fea = fea.drop_duplicates("fips")
print(f"  Counties: {len(fea)}")

# ACS
print("Loading ACS 2019-2023...")
acs = pd.read_csv(f"{RAW}/ACS_2019_2023_county.csv", dtype={"fips": str})
acs["fips"] = clean_fips(acs["fips"])
acs = acs.drop_duplicates("fips")
print(f"  Counties: {len(acs)}")

# RUCC
print("Loading RUCC 2013...")
rucc = pd.read_excel(f"{RAW}/ruralurbancodes2013.xls", dtype=str)
fips_col = next(c for c in rucc.columns if c.upper() in ["FIPS","FIPS_CODE","FIPSTXT"])
code_col = next(c for c in rucc.columns if "RUCC" in c.upper())
rucc["fips"]       = clean_fips(rucc[fips_col])
rucc["rucc_code"]  = pd.to_numeric(rucc[code_col], errors="coerce")
rucc["rural_flag"] = (rucc["rucc_code"] >= 4).astype(int)
rucc = rucc[["fips","rucc_code","rural_flag"]].drop_duplicates("fips")
print(f"  Counties: {len(rucc)}, rural: {rucc.rural_flag.sum()}")

# Merge
print("\nMerging...")
master = places.copy()
master = master.merge(fea,  on="fips", how="left")
master = master.merge(acs,  on="fips", how="left")
master = master.merge(rucc, on="fips", how="left")
print(f"Final shape: {master.shape}")
print(master.isnull().sum().to_string())

master.to_csv(f"{PROCESSED}/county_master_pre_fdi.csv", index=False)
print(f"Saved: {PROCESSED}/county_master_pre_fdi.csv")
print("Script 01 complete.")
