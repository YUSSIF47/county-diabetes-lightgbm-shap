import pandas as pd, numpy as np, joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor

PROCESSED="data/processed"
with open(f"{PROCESSED}/feature_list.txt") as f:
    FEATURES=[l.strip() for l in f if l.strip()]
OUTCOME="diabetes_prev"

train=pd.read_csv(f"{PROCESSED}/train_unscaled.csv",dtype={"fips":str})
val=pd.read_csv(f"{PROCESSED}/val_unscaled.csv",dtype={"fips":str})
test=pd.read_csv(f"{PROCESSED}/test_unscaled.csv",dtype={"fips":str})
all_df=pd.concat([train,val,test],ignore_index=True)

regions=["Northeast","Midwest","South","West"]
results=[]

for held_out in regions:
    tr=all_df[all_df["census_region"]!=held_out].copy()
    te=all_df[all_df["census_region"]==held_out].copy()
    imp=IterativeImputer(max_iter=10,random_state=42)
    imp.fit(tr[FEATURES])
    scaler=MinMaxScaler()
    scaler.fit(imp.transform(tr[FEATURES]))
    X_tr=scaler.transform(imp.transform(tr[FEATURES]))
    X_te=scaler.transform(imp.transform(te[FEATURES]))
    y_tr=tr[OUTCOME].values
    y_te=te[OUTCOME].values
    m=XGBRegressor(n_estimators=500,max_depth=3,learning_rate=0.1,
                   subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=1)
    m.fit(X_tr,y_tr)
    y_pred=m.predict(X_te)
    rmse=np.sqrt(mean_squared_error(y_te,y_pred))
    mae=mean_absolute_error(y_te,y_pred)
    mape=np.mean(np.abs((y_te-y_pred)/y_te))*100
    r2=r2_score(y_te,y_pred)
    print(f"{held_out}: RMSE={rmse:.3f} MAE={mae:.3f} MAPE={mape:.2f}% R2={r2:.3f} (n={len(te)})")
    results.append({"Region":held_out,"n_counties":len(te),"RMSE":round(rmse,3),
                    "MAE":round(mae,3),"MAPE":round(mape,2),"R2":round(r2,3)})

pd.DataFrame(results).to_csv("outputs/tables/region_holdout_results.csv",index=False)
print("Saved.")
