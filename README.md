# County-Level Diagnosed Diabetes Prevalence: XGBoost + SHAP

**Replication code and processed dataset for:**

> Yahaya Y, Khan S, Saha PR, Meia MA. *Predicting County-Level Diagnosed Diabetes Prevalence in the United States Using Explainable Gradient Boosting and SHAP-Based Geographic Interpretation.* Journal of Community Health (2026).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## Overview

This repository contains the Python analysis code and processed county-level dataset used in the above study. The study develops an explainable gradient boosting framework for predicting county-level diagnosed diabetes prevalence across 2,957 U.S. counties using food-environment, socioeconomic, occupational, demographic, and health-behavior indicators from five publicly available data sources.

---

## Repository structure

```
├── county_master_clean.csv        # Processed county-level dataset (2,957 counties)
├── requirements.txt               # Python package dependencies
├── 01_build_master_dataset.py     # Data assembly and FIPS linking
├── 02_missing_values_check.py     # Missing value diagnostics
├── 03_food_desert_index.py        # Population-weighted FDI aggregation
├── 04_food_environment_composite_index.py  # PCA-based FECI construction
├── 05_train_val_test_split.py     # Stratified 70/15/15 data split
├── 06_train_models.py             # Model training and hyperparameter tuning
├── 07_model_comparison.py         # Model comparison, Wilcoxon tests, bootstrap CIs
├── 08_09_shap_analysis.py         # SHAP analysis and geographic interpretation
├── 10_spatial_diagnostics.py      # Moran's I and LISA cluster maps
├── 11_sensitivity_analyses.py     # Nine sensitivity model specifications
├── 12_compile_results_for_paper.py # Results compilation
├── region_holdout.py              # Region-held-out geographic cross-validation
├── fix_notitles2.py               # Main manuscript figures (Fig1–Fig7)
└── regen_all_supp.py              # Supplementary figures (ESM_2–ESM_10)
```

---

## Data

### Processed dataset
`county_master_clean.csv` contains the final analytic dataset with 2,957 U.S. counties and all 22 primary predictors plus the outcome variable (county-level diagnosed diabetes prevalence). Variable definitions are provided in the manuscript Methods section.

### Raw data sources
Raw data are publicly available and must be downloaded separately:

| Source | URL |
|--------|-----|
| CDC PLACES 2025 | https://www.cdc.gov/places |
| USDA Food Access Research Atlas | https://www.ers.usda.gov/data-products/food-access-research-atlas/ |
| USDA Food Environment Atlas | https://www.ers.usda.gov/data-products/food-environment-atlas/ |
| U.S. Census ACS 5-year estimates | https://www.census.gov/programs-surveys/acs/ |
| USDA Rural-Urban Continuum Codes | https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/ |
| U.S. County Shapefile (Census TIGER) | https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html |

---

## Computational environment

- **Local preprocessing:** Python 3.9, Jupyter Notebook, Windows workstation
- **Model training and analysis:** Python 3.10, UTRGV CRADLE HPC cluster (kimq partition, 8 CPUs, 16 GB RAM)
- **Key package versions:** XGBoost 1.7.6, LightGBM 4.6.0, shap 0.49.1, geopandas 1.1.3, libpysal 4.13.0, esda 2.7.0

> **Important:** XGBoost must be pinned to version 1.7.6. Version 3.x causes a segfault on the CRADLE environment. All model training uses `n_jobs=1`.

---

## Reproducing the analysis

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download raw data sources (see table above) into `data/raw/`
4. Run scripts in order: `01_build_master_dataset.py` through `12_compile_results_for_paper.py`
5. Generate figures: `python fix_notitles2.py` and `python regen_all_supp.py`

All model training was performed on CRADLE HPC. Runtime for the full pipeline is approximately 2–3 hours on 8 CPUs.

---

## Citation

If you use this code or dataset, please cite:

```
Yahaya Y, Saha PR, Khan S, Meia MA (2026). Predicting County-Level Diagnosed
Diabetes Prevalence in the United States Using Explainable Gradient Boosting
and SHAP-Based Geographic Interpretation. Journal of Community Health.
https://doi.org/10.5281/zenodo.XXXXXXX
```

---

## License

This repository is licensed under the [MIT License](LICENSE).  
The processed dataset is licensed under [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

---

## Contact

Corresponding author: Yussif Yahaya — yussif.yahaya01@utrgv.edu  
School of Mathematical and Statistical Sciences, University of Texas Rio Grande Valley
