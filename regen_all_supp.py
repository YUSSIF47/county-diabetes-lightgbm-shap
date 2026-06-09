import pandas as pd, numpy as np, matplotlib, os, joblib, shap
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":14,"axes.titlesize":16,
    "axes.labelsize":14,"xtick.labelsize":13,"ytick.labelsize":13,
    "legend.fontsize":13,"legend.title_fontsize":14,
})

PROCESSED="data/processed"
FIGURES="outputs/figures/final"
os.makedirs(FIGURES,exist_ok=True)
MINX,MAXX,MINY,MAXY=-125.0,-66.5,24.0,49.5

def save_fig(fig,name):
    for ext,dpi in [("pdf",None),("png",300),("tif",600)]:
        kw={"dpi":dpi} if dpi else {}
        fig.savefig(f"{FIGURES}/{name}.{ext}",bbox_inches="tight",facecolor="white",**kw)
    print(f"  Saved: {name}")
    plt.close(fig)

LABELS={
    "poverty_rate":"Poverty rate (%)","snap_rate":"SNAP participation rate (%)",
    "median_income":"Median household income","food_insecurity_rate":"Food insecurity rate (%)",
    "physical_inactivity_prev":"Physical inactivity (%)","pct_white":"% White population",
    "pct_black":"% Black population","unemployment_rate":"Unemployment rate (%)",
    "food_desert_index":"Food desert index (%)","pct_foodsvc_sector":"% Food service sector",
}
def L(v): return LABELS.get(v,v.replace("_"," ").title())

print("Loading data...")
with open(f"{PROCESSED}/feature_list.txt") as f: FEATURES=[l.strip() for l in f if l.strip()]
OUTCOME="diabetes_prev"
models_dict={
    "Elastic Net":joblib.load("models/elastic_net.pkl"),
    "Random Forest":joblib.load("models/random_forest.pkl"),
    "XGBoost":joblib.load("models/xgboost.pkl"),
    "LightGBM":joblib.load("models/lightgbm.pkl"),
}
train=pd.read_csv(f"{PROCESSED}/train.csv",dtype={"fips":str})
val=pd.read_csv(f"{PROCESSED}/val.csv",dtype={"fips":str})
test=pd.read_csv(f"{PROCESSED}/test.csv",dtype={"fips":str})
all_df=pd.concat([train,val,test],ignore_index=True)
X_test=test[FEATURES].values; y_test=test[OUTCOME].values
model=joblib.load("models/xgboost.pkl")

print("Computing SHAP...")
explainer=shap.TreeExplainer(model)
shap_values=explainer.shap_values(all_df[FEATURES])
shap_df=pd.DataFrame(shap_values,columns=FEATURES)
shap_df["fips"]=all_df["fips"].values
mean_abs=pd.DataFrame({"feature":FEATURES,"v":np.abs(shap_values).mean(axis=0)}).sort_values("v",ascending=False).reset_index(drop=True)

structural=["food_desert_index","fastfood_density","grocery_density","snap_rate",
    "food_insecurity_rate","feci","poverty_rate","median_income","unemployment_rate",
    "pct_agri_sector","pct_mfg_sector","pct_foodsvc_sector","rucc_code"]

gdf=gpd.read_file("data/shapefiles/cb_2019_us_county_500k.shp")
gdf["fips"]=gdf["GEOID"].str.zfill(5)
gdf_cont=gdf[~gdf["STATEFP"].isin(["02","15","72","78","60","66","69"])].copy()

# ── ESM_2: Observed vs predicted — no title ───────────────────
print("ESM_2: Observed vs predicted...")
colors=["#2ecc71","#3498db","#e74c3c","#9b59b6"]
fig,axes=plt.subplots(2,2,figsize=(14,11))
axes=axes.flatten()
for ax,(name,m),col in zip(axes,models_dict.items(),colors):
    yp=m.predict(X_test)
    r2=r2_score(y_test,yp)
    ax.scatter(y_test,yp,alpha=0.35,s=14,color=col,edgecolors="none")
    lims=[min(y_test.min(),yp.min())-0.5,max(y_test.max(),yp.max())+0.5]
    ax.plot(lims,lims,"k--",linewidth=1.5,label="Perfect prediction")
    ax.set_xlabel("Observed diagnosed diabetes prevalence (%)",fontsize=13)
    ax.set_ylabel("Predicted diagnosed diabetes prevalence (%)",fontsize=13)
    ax.set_title("")
    ax.annotate(f"{name}\n$R^2 = {r2:.3f}$",xy=(0.05,0.93),xycoords="axes fraction",
        fontsize=13,va="top",
        bbox=dict(boxstyle="round,pad=0.3",facecolor="white",alpha=0.8))
    ax.legend(fontsize=11,loc="lower right")
plt.tight_layout()
save_fig(fig,"ESM_2")

# ── ESM_3: Residual plots — no title ─────────────────────────
print("ESM_3: Residual plots...")
fig,axes=plt.subplots(2,2,figsize=(14,11))
axes=axes.flatten()
for ax,(name,m),col in zip(axes,models_dict.items(),colors):
    yp=m.predict(X_test)
    res=y_test-yp
    rmse=np.sqrt(mean_squared_error(y_test,yp))
    r2=r2_score(y_test,yp)
    ax.scatter(yp,res,alpha=0.35,s=14,color=col,edgecolors="none")
    ax.axhline(0,color="#e74c3c",linewidth=1.5,linestyle="--")
    ax.set_xlabel("Predicted diagnosed diabetes prevalence (%)",fontsize=13)
    ax.set_ylabel("Residual (observed minus predicted)",fontsize=13)
    ax.set_title("")
    ax.annotate(f"{name}\nRMSE = {rmse:.3f},  $R^2 = {r2:.3f}$",
        xy=(0.05,0.93),xycoords="axes fraction",fontsize=13,va="top",
        bbox=dict(boxstyle="round,pad=0.3",facecolor="white",alpha=0.8))
plt.tight_layout()
save_fig(fig,"ESM_3")

# ── ESM_4: SHAP map poverty rate — no title ───────────────────
print("ESM_4: SHAP map poverty rate...")
feat="poverty_rate"
gdf_shap=gdf_cont.merge(shap_df[["fips",feat]],on="fips",how="left")
vmax=gdf_shap[feat].abs().quantile(0.95)
fig,ax=plt.subplots(figsize=(16,8))
gdf_shap.plot(column=feat,ax=ax,legend=True,cmap="RdBu_r",vmin=-vmax,vmax=vmax,
    missing_kwds={"color":"lightgrey"},
    legend_kwds={"label":f"SHAP value — {L(feat)}","shrink":0.5,"orientation":"vertical"})
ax.set_xlim(MINX,MAXX); ax.set_ylim(MINY,MAXY); ax.axis("off"); ax.set_title("")
ax.annotate("Red = increases predicted prevalence  |  Blue = decreases predicted prevalence",
    xy=(0.5,-0.01),xycoords="axes fraction",ha="center",va="top",fontsize=12,color="gray")
plt.tight_layout()
save_fig(fig,"ESM_4")

# ── ESM_5: SHAP map SNAP rate — no title ─────────────────────
print("ESM_5: SHAP map SNAP rate...")
feat="snap_rate"
gdf_shap=gdf_cont.merge(shap_df[["fips",feat]],on="fips",how="left")
vmax=gdf_shap[feat].abs().quantile(0.95)
fig,ax=plt.subplots(figsize=(16,8))
gdf_shap.plot(column=feat,ax=ax,legend=True,cmap="RdBu_r",vmin=-vmax,vmax=vmax,
    missing_kwds={"color":"lightgrey"},
    legend_kwds={"label":f"SHAP value — {L(feat)}","shrink":0.5,"orientation":"vertical"})
ax.set_xlim(MINX,MAXX); ax.set_ylim(MINY,MAXY); ax.axis("off"); ax.set_title("")
ax.annotate("Red = increases predicted prevalence  |  Blue = decreases predicted prevalence",
    xy=(0.5,-0.01),xycoords="axes fraction",ha="center",va="top",fontsize=12,color="gray")
plt.tight_layout()
save_fig(fig,"ESM_5")

# ── ESM_6: SHAP map median income — no title ─────────────────
print("ESM_6: SHAP map median income...")
feat="median_income"
gdf_shap=gdf_cont.merge(shap_df[["fips",feat]],on="fips",how="left")
vmax=gdf_shap[feat].abs().quantile(0.95)
fig,ax=plt.subplots(figsize=(16,8))
gdf_shap.plot(column=feat,ax=ax,legend=True,cmap="RdBu_r",vmin=-vmax,vmax=vmax,
    missing_kwds={"color":"lightgrey"},
    legend_kwds={"label":f"SHAP value — {L(feat)}","shrink":0.5,"orientation":"vertical"})
ax.set_xlim(MINX,MAXX); ax.set_ylim(MINY,MAXY); ax.axis("off"); ax.set_title("")
ax.annotate("Red = increases predicted prevalence  |  Blue = decreases predicted prevalence",
    xy=(0.5,-0.01),xycoords="axes fraction",ha="center",va="top",fontsize=12,color="gray")
plt.tight_layout()
save_fig(fig,"ESM_6")

# ── ESM_7: SHAP dependence poverty rate — no title ───────────
print("ESM_7: SHAP dependence poverty rate...")
feat="poverty_rate"; interact="median_income"
feat_idx=FEATURES.index(feat)
fig,ax=plt.subplots(figsize=(10,7))
sc=ax.scatter(all_df[feat].values,shap_values[:,feat_idx],
    c=all_df[interact].values,cmap="coolwarm",alpha=0.5,s=14,edgecolors="none")
plt.colorbar(sc,ax=ax,label=L(interact),shrink=0.8)
ax.axhline(0,color="grey",linewidth=0.8,linestyle="--")
ax.set_xlabel(L(feat),fontsize=14)
ax.set_ylabel(f"SHAP value for {L(feat)}",fontsize=14)
ax.set_title("")
plt.tight_layout()
save_fig(fig,"ESM_7")

# ── ESM_8: SHAP dependence SNAP rate — no title ──────────────
print("ESM_8: SHAP dependence SNAP rate...")
feat="snap_rate"; interact="food_insecurity_rate"
feat_idx=FEATURES.index(feat)
fig,ax=plt.subplots(figsize=(10,7))
sc=ax.scatter(all_df[feat].values,shap_values[:,feat_idx],
    c=all_df[interact].values,cmap="coolwarm",alpha=0.5,s=14,edgecolors="none")
plt.colorbar(sc,ax=ax,label=L(interact),shrink=0.8)
ax.axhline(0,color="grey",linewidth=0.8,linestyle="--")
ax.set_xlabel(L(feat),fontsize=14)
ax.set_ylabel(f"SHAP value for {L(feat)}",fontsize=14)
ax.set_title("")
plt.tight_layout()
save_fig(fig,"ESM_8")

# ── ESM_9: SHAP dependence median income — no title ──────────
print("ESM_9: SHAP dependence median income...")
feat="median_income"; interact="pct_white"
feat_idx=FEATURES.index(feat)
fig,ax=plt.subplots(figsize=(10,7))
sc=ax.scatter(all_df[feat].values,shap_values[:,feat_idx],
    c=all_df[interact].values,cmap="coolwarm",alpha=0.5,s=14,edgecolors="none")
plt.colorbar(sc,ax=ax,label=LABELS.get(interact,"% White population"),shrink=0.8)
ax.axhline(0,color="grey",linewidth=0.8,linestyle="--")
ax.set_xlabel(L(feat),fontsize=14)
ax.set_ylabel(f"SHAP value for {L(feat)}",fontsize=14)
ax.set_title("")
plt.tight_layout()
save_fig(fig,"ESM_9")

# ── ESM_10: Residual choropleth — no title ────────────────────
print("ESM_10: Residual choropleth...")
all_df["y_pred"]=model.predict(all_df[FEATURES].values)
all_df["residual"]=all_df[OUTCOME]-all_df["y_pred"]
gdf_res=gdf_cont.merge(all_df[["fips","residual"]],on="fips",how="left")
vmax=gdf_res["residual"].abs().quantile(0.95)
fig,ax=plt.subplots(figsize=(16,8))
gdf_res.plot(column="residual",ax=ax,legend=True,cmap="RdBu_r",vmin=-vmax,vmax=vmax,
    missing_kwds={"color":"lightgrey"},
    legend_kwds={"label":"Residual (observed minus predicted, percentage points)",
        "shrink":0.5,"orientation":"vertical"})
ax.set_xlim(MINX,MAXX); ax.set_ylim(MINY,MAXY); ax.axis("off"); ax.set_title("")
ax.annotate("Red = underprediction  |  Blue = overprediction",
    xy=(0.5,-0.01),xycoords="axes fraction",ha="center",va="top",fontsize=12,color="gray")
plt.tight_layout()
save_fig(fig,"ESM_10")

print("\n"+"="*55)
print("All supplementary figures saved:")
print("ESM_2 through ESM_10 in outputs/figures/final/")
print("No internal titles — captions carry all descriptions")
print("="*55)
