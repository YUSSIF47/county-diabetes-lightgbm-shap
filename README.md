# County-Level Diagnosed Diabetes Prevalence: LightGBM + SHAP

**Replication code and processed dataset for:**

> Yahaya Y, Khan S, Saha PR, Meia MA. *Predicting County-Level Diagnosed Diabetes Prevalence in the United States Using Explainable Gradient Boosting and Geographic Interpretation.* Manuscript prepared for submission to PLOS Digital Health (2026).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## Overview

This repository contains the Python analysis code and processed county-level dataset supporting the above manuscript. The study develops an explainable gradient boosting framework for predicting county-level diagnosed diabetes prevalence across 2,957 U.S. counties using food-environment, socioeconomic, occupational, demographic, and health-behavior indicators from five publicly available data sources.

The primary prediction model (LightGBM) was selected using validation-set RMSE before any evaluation on the held-out test set. SHAP TreeExplainer was applied to all 2,957 analytic counties to generate county-level geographic interpretability maps identifying the dominant positive structural contributor to predicted diabetes prevalence in each county.

---

## Repository structure

```
├── county_master_analytic_final.csv     # Final analytic dataset (2,957 counties, 25 columns)
├── feature_list_primary_22.txt          # 22 primary predictor names
├── dominant_contributors_lgb.csv        # Final dominant contributor assignments (LightGBM)
├── model_performance_table.csv          # Final validation and test performance results
│
├── 01_build_master_dataset.py           # Data assembly and FIPS linking
├── 02_missing_values_check.py           # Missing value diagnostics
├── 03_food_desert_index.py              # Population-weighted food desert index
├── 04_food_environment_composite_index.py  # PCA-based FECI construction
├── 05_train_val_test_split.py           # Stratified 70/15/15 split by Census region
├── 06_train_models.py                   # Model training and hyperparameter tuning
├── 07_model_comparison.py               # Model comparison (validation-based selection)
├── 08_09_shap_analysis.py               # SHAP analysis and geographic interpretation
├── 10_spatial_diagnostics.py            # Global and local Moran's I, LISA cluster maps
├── 11_sensitivity_analyses.py           # Nine sensitivity model specifications
├── 12_compile_results_for_paper.py      # Results compilation
│
├── region_holdout_lightgbm.py           # Region-held-out geographic cross-validation
├── shap_agreement_lightgbm_xgboost.py   # SHAP stability and model agreement analysis
│
├── fig01_model_comparison.py            # Fig 1: Validation and test performance
├── fig02_shap_importance.py             # Fig 2: Global mean absolute SHAP importance
├── fig03_shap_beeswarm.py               # Fig 3: SHAP summary plot
├── fig04_dominant_contributor_map.py    # Fig 4: Dominant positive structural contributor map
├── fig_main_lisa_cluster_map.py         # Fig 7: LISA cluster map
│
├── figS1_S2_obs_pred_residuals.py       # S1 Fig, S2 Fig: Observed vs predicted and residuals
├── figS3_S4_S5_shap_maps.py            # S3-S5 Fig: County-level SHAP maps
├── figS6_S7_S8_shap_dependence.py      # S6-S8 Fig: SHAP dependence plots (original units)
├── figS9_residual_map.py                # S9 Fig: LightGBM residual choropleth
├── figS10_within_region_dominant_contributor_heatmap.py  # S10 Fig: Within-region % heatmap
└── figS11_regional_signed_shap_heatmap.py               # S11 Fig: Regional mean signed SHAP
```

---

## Data

### Final analytic dataset
`county_master_analytic_final.csv` contains the final analytic dataset with 2,957 U.S. counties, the outcome variable (county-level diagnosed diabetes prevalence), FIPS code, Census region, and all 22 primary predictors. Values reflect the training-partition-fitted imputation applied to the full dataset for descriptive and SHAP analysis purposes.

Pennsylvania and Kentucky counties were absent from the CDC PLACES 2025 release and are not included in the analytic sample.

### Raw data sources
Raw data are publicly available and must be downloaded separately:

| Source | URL |
|---|---|
| CDC PLACES 2025 | https://www.cdc.gov/places |
| USDA Food Access Research Atlas | https://www.ers.usda.gov/data-products/food-access-research-atlas/ |
| USDA Food Environment Atlas | https://www.ers.usda.gov/data-products/food-environment-atlas/ |
| U.S. Census ACS 5-year estimates | https://www.census.gov/programs-surveys/acs/ |
| USDA Rural-Urban Continuum Codes | https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/ |
| U.S. County Shapefile (Census TIGER) | https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html |

---

## Model selection

The primary prediction model was selected using RMSE on the held-out validation set (n = 444 counties) before any evaluation on the test set. LightGBM achieved the lowest validation RMSE (0.4421), followed closely by XGBoost (0.4462). LightGBM was therefore selected as the primary model. The test set was evaluated exactly once after model selection was complete.

**Final performance (held-out test set, n = 444 counties):**

| Model | Val RMSE | Test RMSE | Test MAE | Test MAPE | Test R² |
|---|---|---|---|---|---|
| LightGBM (primary) | 0.4421 | 0.4234 | 0.3114 | 2.76% | 0.9636 |
| XGBoost (secondary) | 0.4462 | 0.3987 | 0.2902 | 2.56% | 0.9677 |
| Random Forest | 0.5616 | 0.5379 | 0.3959 | 3.53% | 0.9412 |
| Elastic Net | 0.5794 | 0.5852 | 0.4538 | 4.09% | 0.9304 |

---

## Computational environment

- **Local preprocessing:** Python 3.9, Jupyter Notebook, Windows workstation
- **Model training and analysis:** Python 3.10, UTRGV CRADLE HPC cluster (kimq partition, 8 CPUs, 16 GB RAM)
- **Key package versions:** XGBoost 1.7.6, LightGBM 4.6.0, shap 0.49.1, scikit-learn 1.7.0, geopandas 1.1.3, libpysal 4.13.0, esda 2.7.0

> **Note:** XGBoost must be pinned to version 1.7.6. Version 3.x causes a segmentation fault in the CRADLE environment. All model training uses n_jobs=1 for reproducibility.

---

## Reproducing the analysis

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download raw data sources into `data/raw/`
4. Run numbered scripts in order: `01_build_master_dataset.py` through `12_compile_results_for_paper.py`
5. Run additional analyses: `region_holdout_lightgbm.py`, `shap_agreement_lightgbm_xgboost.py`
6. Generate figures: run figure scripts in order

All model training was performed on CRADLE HPC. Runtime for the full pipeline is approximately 3--4 hours on 8 CPUs.

---

## Citation

If you use this code or dataset, please cite:

```
Yahaya Y, Khan S, Saha PR, Meia MA (2026). Predicting County-Level
Diagnosed Diabetes Prevalence in the United States Using Explainable
Gradient Boosting and Geographic Interpretation.
PLOS Digital Health. https://doi.org/10.5281/zenodo.XXXXXXX
```

---

## License

Code is licensed under the [MIT License](LICENSE).
The processed dataset is licensed under [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

---

## Contact

Corresponding author: Yussif Yahaya — yussif.yahaya01@utrgv.edu
School of Mathematical and Statistical Sciences, University of Texas Rio Grande Valley
