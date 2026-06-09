import pandas as pd, numpy as np, matplotlib, os, joblib, shap
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":14,"axes.titlesize":16,
    "axes.labelsize":14,"xtick.labelsize":13,"ytick.labelsize":13,
    "legend.fontsize":13,"legend.title_fontsize":14,
})

PROCESSED="data/processed"; FIGURES="outputs/figures/final"
MINX,MAXX,MINY,MAXY=-125.0,-66.5,24.0,49.5

def save_fig(fig,name):
    for ext,dpi in [("pdf",None),("png",300),("tif",600)]:
        kw={"dpi":dpi} if dpi else {}
        fig.savefig(f"{FIGURES}/{name}.{ext}",bbox_inches="tight",facecolor="white",**kw)
    print(f"  Saved: {name}")
    plt.close(fig)

LABELS={
    "physical_inactivity_prev":"Physical inactivity (%)","pct_white":"% White population",
    "poverty_rate":"Poverty rate (%)","pct_black":"% Black population",
    "snap_rate":"SNAP participation rate (%)","smoking_prev":"Smoking prevalence (%)",
    "median_income":"Median household income","food_insecurity_rate":"Food insecurity rate (%)",
    "pct_age65plus":"% Age 65+","obesity_prev":"Obesity prevalence (%)",
    "pct_hispanic":"% Hispanic population","hypertension_prev":"Hypertension prevalence (%)",
    "pct_college_edu":"% College educated","unemployment_rate":"Unemployment rate (%)",
    "food_desert_index":"Food desert index (%)","feci":"Food Env. Composite Index",
    "pct_agri_sector":"% Agriculture sector","fastfood_density":"Fast-food density (per 1k)",
    "grocery_density":"Grocery density (per 1k)","pct_mfg_sector":"% Manufacturing sector",
    "pct_foodsvc_sector":"% Food service sector","rucc_code":"RUCC rurality code",
}
def L(v): return LABELS.get(v,v.replace("_"," ").title())

structural_all=["food_desert_index","fdi_binary_high","fastfood_density","grocery_density",
    "snap_rate","food_insecurity_rate","feci","poverty_rate","median_income",
    "unemployment_rate","pct_agri_sector","pct_mfg_sector","pct_foodsvc_sector","rucc_code","rural_flag"]

structural_primary=["food_desert_index","fastfood_density","grocery_density","snap_rate",
    "food_insecurity_rate","feci","poverty_rate","median_income","unemployment_rate",
    "pct_agri_sector","pct_mfg_sector","pct_foodsvc_sector","rucc_code"]

DL={"poverty_rate":"Poverty rate","food_insecurity_rate":"Food insecurity rate",
    "snap_rate":"SNAP participation rate","median_income":"Median income",
    "unemployment_rate":"Unemployment rate","food_desert_index":"Food desert index",
    "feci":"Food Env. Composite Index","fastfood_density":"Fast-food density",
    "grocery_density":"Grocery density","pct_agri_sector":"Agriculture sector",
    "pct_mfg_sector":"Manufacturing sector","pct_foodsvc_sector":"Food service sector",
    "rucc_code":"Rurality (RUCC)","None":"Not determined"}

PAL={"poverty_rate":"#e74c3c","food_insecurity_rate":"#e67e22","snap_rate":"#3498db",
    "median_income":"#2ecc71","unemployment_rate":"#9b59b6","food_desert_index":"#f39c12",
    "feci":"#1abc9c","fastfood_density":"#e91e63","grocery_density":"#795548",
    "pct_agri_sector":"#8bc34a","pct_mfg_sector":"#ff5722",
    "pct_foodsvc_sector":"#00bcd4","rucc_code":"#673ab7","None":"#bdc3c7"}

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

print("Computing SHAP...")
model=joblib.load("models/xgboost.pkl")
explainer=shap.TreeExplainer(model)
shap_values=explainer.shap_values(all_df[FEATURES])
shap_df=pd.DataFrame(shap_values,columns=FEATURES)
shap_df["fips"]=all_df["fips"].values
shap_df["census_region"]=all_df["census_region"].values
mean_abs=pd.DataFrame({"feature":FEATURES,"v":np.abs(shap_values).mean(axis=0)}).sort_values("v",ascending=False).reset_index(drop=True)

# FIG 1 — no title
print("Fig 1...")
metrics={}
for name,m in models_dict.items():
    yp=m.predict(X_test)
    metrics[name]={"RMSE":np.sqrt(mean_squared_error(y_test,yp)),"R2":r2_score(y_test,yp)}
model_names=list(metrics.keys())
rmse_vals=[metrics[m]["RMSE"] for m in model_names]
r2_vals=[metrics[m]["R2"] for m in model_names]
colors=["#2ecc71","#3498db","#e74c3c","#9b59b6"]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
bars1=ax1.bar(model_names,rmse_vals,color=colors,edgecolor="white",width=0.55)
ax1.set_ylabel("RMSE (percentage points)",fontsize=14)
ax1.set_xlabel("Model",fontsize=14)
ax1.set_ylim(0,max(rmse_vals)*1.25)
ax1.set_title("") # explicitly blank
for bar,val in zip(bars1,rmse_vals):
    ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,f"{val:.3f}",ha="center",va="bottom",fontsize=13,fontweight="bold")
ax1.annotate("Lower is better",xy=(0.98,0.98),xycoords="axes fraction",ha="right",va="top",fontsize=11,color="gray")
bars2=ax2.bar(model_names,r2_vals,color=colors,edgecolor="white",width=0.55)
ax2.set_ylabel("$R^2$",fontsize=14)
ax2.set_xlabel("Model",fontsize=14)
ax2.set_ylim(0.88,1.0)
ax2.set_title("") # explicitly blank
for bar,val in zip(bars2,r2_vals):
    ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.001,f"{val:.3f}",ha="center",va="bottom",fontsize=13,fontweight="bold")
ax2.annotate("Higher is better",xy=(0.98,0.98),xycoords="axes fraction",ha="right",va="top",fontsize=11,color="gray")
fig.suptitle("")  # blank supertitle
plt.tight_layout()
save_fig(fig,"Fig1")

# FIG 2 — no title (remove title after plotting)
print("Fig 2...")
top20=mean_abs.head(20)
colors2=["#c0392b" if f in structural_all else "#2980b9" for f in top20["feature"]]
ylabels=[L(f) for f in top20["feature"]]
fig,ax=plt.subplots(figsize=(13,10))
ax.barh(ylabels[::-1],top20["v"][::-1],color=colors2[::-1],edgecolor="white",linewidth=0.5)
ax.set_xlabel("Mean |SHAP value| (percentage points)",fontsize=14)
ax.set_title("")  # no title
ax.legend(handles=[
    mpatches.Patch(facecolor="#c0392b",label="Structural determinants"),
    mpatches.Patch(facecolor="#2980b9",label="Health-behavior / demographic"),
],loc="lower right",fontsize=13,framealpha=0.9)
ax.grid(axis="x",alpha=0.3,linewidth=0.7)
plt.tight_layout()
save_fig(fig,"Fig2")

# FIG 3 — beeswarm, strip title after shap renders it
print("Fig 3...")
top15_feats=mean_abs.head(15)["feature"].tolist()
idx=[FEATURES.index(f) for f in top15_feats]
sv_top=shap_values[:,idx]
X_top=all_df[top15_feats].copy()
shap.summary_plot(sv_top,X_top,feature_names=[L(f) for f in top15_feats],
    show=False,plot_type="dot",max_display=15,plot_size=(13,10),
    color_bar_label="Feature value (red=high, blue=low)")
fig=plt.gcf()
# Remove ALL text objects that look like titles (at top of figure)
for ax in fig.get_axes():
    ax.set_title("")
fig.suptitle("")
# Remove any text at top of figure
for text in fig.texts:
    text.set_visible(False)
ax=fig.get_axes()[0]
ax.set_xlabel("SHAP value (impact on predicted diagnosed diabetes prevalence)",fontsize=14)
ax.tick_params(axis="y",labelsize=13)
ax.tick_params(axis="x",labelsize=13)
plt.tight_layout()
save_fig(fig,"Fig3")

# FIG 4 — dominant map, no title, no High food-desert indicator
print("Fig 4...")
gdf=gpd.read_file("data/shapefiles/cb_2019_us_county_500k.shp")
gdf["fips"]=gdf["GEOID"].str.zfill(5)
gdf_cont=gdf[~gdf["STATEFP"].isin(["02","15","72","78","60","66","69"])].copy()
struct_cols=[f for f in structural_primary if f in FEATURES]
pos=shap_df[struct_cols].clip(lower=0)
shap_df["dominant"]=pos.idxmax(axis=1)
shap_df.loc[pos.max(axis=1)==0,"dominant"]="None"
gdf_dom=gdf_cont.merge(shap_df[["fips","dominant"]],on="fips",how="left")
gdf_dom["dominant"]=gdf_dom["dominant"].fillna("None")
present=gdf_dom["dominant"].value_counts()
present=present[present>=3].index.tolist()
fig,ax=plt.subplots(figsize=(20,11))
for d in present:
    gdf_dom[gdf_dom["dominant"]==d].plot(ax=ax,color=PAL.get(d,"#bdc3c7"),edgecolor="none")
ax.set_xlim(MINX,MAXX); ax.set_ylim(MINY,MAXY); ax.axis("off")
ax.set_title("")  # no title
legend_els=[mpatches.Patch(facecolor=PAL.get(d,"#bdc3c7"),edgecolor="none",
    label=f"{DL.get(d,d)} (n={(gdf_dom['dominant']==d).sum():,})") for d in present]
legend_els.append(mpatches.Patch(facecolor="#bdc3c7",edgecolor="none",label="Other / not determined"))
ax.legend(handles=legend_els,loc="lower left",fontsize=12,
    title="Dominant positive structural SHAP contributor",title_fontsize=13,
    framealpha=0.92,ncol=2,borderpad=0.9,labelspacing=0.65,handlelength=1.6)
plt.tight_layout()
save_fig(fig,"Fig4")

# FIG 5 — by region bar, no title, no High food-desert indicator
print("Fig 5...")
dom_summary=shap_df.groupby(["dominant","census_region"]).size().reset_index(name="n")
dom_summary["label"]=dom_summary["dominant"].map(lambda x: DL.get(x,x.replace("_"," ").title()))
pivot=dom_summary.pivot(index="label",columns="census_region",values="n").fillna(0)
pivot=pivot.loc[pivot.sum(axis=1)>=3]
# Remove High food-desert indicator row if present
if "High food-desert indicator" in pivot.index:
    pivot=pivot.drop("High food-desert indicator")
fig,ax=plt.subplots(figsize=(16,8))
pivot.plot(kind="bar",ax=ax,colormap="Set2",edgecolor="white",linewidth=0.5)
ax.set_xlabel("Dominant Positive Structural SHAP Contributor",fontsize=14)
ax.set_ylabel("Number of counties",fontsize=14)
ax.set_title("")  # no title
ax.legend(title="Census Region",fontsize=13,title_fontsize=13,framealpha=0.9,loc="upper right")
ax.tick_params(axis="x",rotation=35,labelsize=12)
ax.grid(axis="y",alpha=0.3,linewidth=0.7)
plt.tight_layout()
save_fig(fig,"Fig5")

# FIG 6 — heatmap, no title
print("Fig 6...")
regional=shap_df.groupby("census_region")[FEATURES].mean().round(4)
top15=mean_abs.head(15)["feature"].tolist()
regional_top=regional[top15].T
regional_top.index=[L(f) for f in regional_top.index]
fig,ax=plt.subplots(figsize=(12,11))
vmax=regional_top.abs().max().max()
im=ax.imshow(regional_top.values,cmap="RdBu_r",aspect="auto",vmin=-vmax,vmax=vmax)
ax.set_xticks(range(len(regional_top.columns))); ax.set_xticklabels(regional_top.columns,fontsize=14)
ax.set_yticks(range(len(regional_top.index))); ax.set_yticklabels(regional_top.index,fontsize=13)
ax.set_title("")  # no title
cbar=plt.colorbar(im,ax=ax,shrink=0.65,pad=0.02)
cbar.ax.tick_params(labelsize=12)
cbar.set_label("Mean SHAP value (percentage points)",fontsize=13)
ax.set_xticks(np.arange(-0.5,len(regional_top.columns),1),minor=True)
ax.set_yticks(np.arange(-0.5,len(regional_top.index),1),minor=True)
ax.grid(which="minor",color="white",linewidth=1.5)
plt.tight_layout()
save_fig(fig,"Fig6")

# FIG 7 — LISA map, no title, Moran's I in annotation box only
print("Fig 7...")
gdf_cont2=gdf_cont.copy()
gdf_cont2=gdf_cont2.merge(all_df[["fips","diabetes_prev"]],on="fips",how="inner").reset_index(drop=True)
w=Queen.from_dataframe(gdf_cont2,silence_warnings=True); w.transform="r"
mi=Moran(gdf_cont2["diabetes_prev"].values,w)
lisa=Moran_Local(gdf_cont2["diabetes_prev"].values,w)
gdf_cont2["lisa_q"]=lisa.q; gdf_cont2["lisa_sig"]=lisa.p_sim<0.05
cc={1:"#e74c3c",2:"#3498db",3:"#2ecc71",4:"#f39c12"}
gdf_cont2["color"]=gdf_cont2.apply(
    lambda r: cc.get(r["lisa_q"],"#bdc3c7") if r["lisa_sig"] else "#bdc3c7",axis=1)
fig,ax=plt.subplots(figsize=(18,10))
gdf_cont2.plot(color=gdf_cont2["color"],ax=ax,edgecolor="none")
ax.set_xlim(MINX,MAXX); ax.set_ylim(MINY,MAXY); ax.axis("off")
ax.set_title("")  # no title
ax.legend(handles=[
    mpatches.Patch(facecolor="#e74c3c",label="High-High cluster"),
    mpatches.Patch(facecolor="#2ecc71",label="Low-Low cluster"),
    mpatches.Patch(facecolor="#3498db",label="Low-High outlier"),
    mpatches.Patch(facecolor="#f39c12",label="High-Low outlier"),
    mpatches.Patch(facecolor="#bdc3c7",label="Not significant"),
],loc="lower left",fontsize=13,title="LISA cluster (p < 0.05)",title_fontsize=13,
    framealpha=0.92,borderpad=0.9,labelspacing=0.65,handlelength=1.6)
ax.annotate(f"Global Moran's I = {mi.I:.3f}  (p = {mi.p_sim:.4f})",
    xy=(0.98,0.02),xycoords="axes fraction",ha="right",va="bottom",fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3",facecolor="white",alpha=0.85))
plt.tight_layout()
save_fig(fig,"Fig7")

print("\n"+"="*55)
print("All 7 figures saved — no internal titles")
print("="*55)
