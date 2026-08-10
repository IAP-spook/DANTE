#!/usr/bin/env python3
"""Figure 1 (schematic-led composite): model (hero, a; code-exact, horizontal flow) + data coverage (b,c,d).
Publication-grade, Python/matplotlib. No in-figure titles (caption holds them); black text; panel labels a-d."""
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, Normalize
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,
    "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":0.8,"legend.frameon":False})
BN="/home/dell/hdd8t/gravity_wave/_bench/"; GW="/home/dell/hdd8t/gravity_wave/data/"
FIGS="/home/dell/nanoclaw/groups/main/thermo_paper/figs/"
C=np.load(BN+"cov_fig1.npz")
alt=C["alt"]; lat=C["lat"]; lst=C["lst"]; rho=C["rho"]
f107_date=C["f107_date"].astype("datetime64[D]"); f107=C["f107"]; yr=C["yr"]
# v2clean: despike radio-burst (>300 sfu) + fill -1 missing sentinel in the displayed F10.7 (consistent with model drivers)
_fs=pd.Series(f107.astype(float),index=pd.to_datetime(f107_date)); _fs[(_fs<=0)|(_fs>300)]=np.nan
f107=_fs.interpolate("time",limit_direction="both").values
ap=pd.read_parquet(GW+"thermo_all/ap3h.parquet").sort_values("t")
apw=ap[(ap.t>="2015-03-15")&(ap.t<="2015-03-22")].ap.values
f107w=f107[(f107_date>=np.datetime64("2014-01-01"))&(f107_date<np.datetime64("2014-01-28"))]
# daily Ap (mean of 3-hourly ap) over the data span, for panel (b) bottom strip
_ap=ap[(ap.t>="2000-01-01")&(ap.t<"2025-01-01")].copy(); _ap["day"]=_ap.t.dt.floor("D")
apd=_ap.groupby("day").ap.mean(); apd_date=apd.index.values.astype("datetime64[D]"); apd_val=apd.values
abin=np.arange(230,541,15); ac=0.5*(abin[:-1]+abin[1:])
med=np.array([np.median(rho[(alt>=abin[i])&(alt<abin[i+1])]) for i in range(len(abin)-1)])
q1=np.array([np.percentile(rho[(alt>=abin[i])&(alt<abin[i+1])],25) for i in range(len(abin)-1)])
q3=np.array([np.percentile(rho[(alt>=abin[i])&(alt<abin[i+1])],75) for i in range(len(abin)-1)])

NEU="#4c6b8a"; SIG="#2171b5"; ACC="#d95f0e"; GREY="#8a8a8a"; GEO="#c0603a"
# bespoke premium sequential colormap (deep navy -> teal -> pale), cohesive with DANTE's blue identity
DCMAP=LinearSegmentedColormap.from_list("dante",["#0a1f3c","#173f6b","#2f6f8f","#57a6ab","#a4d4c4","#eef6ec"])
fig=plt.figure(figsize=(7.4,5.9))
gs=GridSpec(2,3,height_ratios=[1.0,0.92],hspace=0.5,wspace=0.66,figure=fig)

# ---------------- (a) hero schematic (code-exact; horizontal main flow) ----------------
axa=fig.add_subplot(gs[0,:]); axa.set_xlim(0,1); axa.set_ylim(0,1); axa.axis("off")
def node(x,y,w,h,txt,fc="#eef3f8",ec=NEU,fs=8,lw=1.1,tc="#111111"):
    axa.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.006,rounding_size=0.02",fc=fc,ec=ec,lw=lw,mutation_aspect=0.5,zorder=2))
    axa.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,zorder=3)
def harrow(x0,x1,y,c=NEU,lw=1.5):
    axa.add_patch(FancyArrowPatch((x0,y),(x1,y),arrowstyle="-|>",mutation_scale=14,lw=lw,color=c,shrinkA=2,shrinkB=2,zorder=1))
def spark(x,y,w,h,data,color,label):
    axi=axa.inset_axes([x,y,w,h]); axi.plot(np.linspace(0,1,len(data)),data,color=color,lw=1.2); axi.axis("off"); axi.margins(0.03)
    axa.text(x+w/2,y+h+0.015,label,ha="center",va="bottom",fontsize=6.7,color="#333333")
# row centres: ap 0.75 | static/linproj 0.505 | f107 0.195 ; head/anchor/output 0.47
spark(0.015,0.68,0.12,0.14,apw,ACC,"ap history · 5 d")
spark(0.015,0.125,0.12,0.14,f107w,SIG,"F10.7 history · 27 d")
node(0.015,0.42,0.12,0.17,"static covariates\nalt, lat, LT, lon,\nseason, F10.7, ap",fc="#f4f6f8",fs=6.3)
node(0.175,0.655,0.15,0.19,"embed +\nTransformer\nencoder",fc="#eaf0f7",fs=7.2)
node(0.175,0.10,0.15,0.19,"embed +\nTransformer\nencoder",fc="#eaf0f7",fs=7.2)
node(0.175,0.435,0.15,0.14,"linear\nprojection",fc="#f4f6f8",fs=7.2)
node(0.375,0.16,0.13,0.62,"cross-\nattention",fc="#e3edf7",ec=SIG)
node(0.565,0.33,0.15,0.28,"MLP head\n→ 5 quantiles",fc="#e3edf7",ec=SIG,fs=7.4)
node(0.735,0.39,0.10,0.16,"exp-atmos.\nanchor",fc="#fdeee2",ec=ACC,fs=6.0)
axo=axa.inset_axes([0.885,0.24,0.09,0.46]); axo.fill_betweenx(ac,q1,q3,color=SIG,alpha=0.22); axo.plot(med,ac,color=SIG,lw=1.6)
axo.set_xscale("log"); axo.set_xticks([]); axo.set_yticks([])
for sp in axo.spines.values(): sp.set_visible(False)
axa.text(0.93,0.72,"ρ(h) + UQ",ha="center",fontsize=6.9,color="#111111")
# horizontal main-flow arrows
harrow(0.135,0.175,0.75); harrow(0.135,0.175,0.195); harrow(0.135,0.175,0.505)
harrow(0.325,0.375,0.75); harrow(0.325,0.375,0.195); harrow(0.325,0.375,0.505)
harrow(0.505,0.565,0.47); harrow(0.715,0.735,0.47); harrow(0.835,0.882,0.47)
# static -> head skip: neat right-angle dashed path routed below cross-attention
axa.plot([0.075,0.075,0.36,0.36,0.565],[0.42,0.40,0.40,0.075,0.075],ls=(0,(4,3)),color=GREY,lw=1.2,zorder=1)
axa.add_patch(FancyArrowPatch((0.565,0.075),(0.565,0.325),arrowstyle="-|>",mutation_scale=13,lw=1.2,color=GREY,ls=(0,(4,3)),shrinkA=0,shrinkB=1,zorder=1))
axa.text(0.235,0.355,"static (skip)",fontsize=6,color=GREY,ha="center")
# two-mode badges (semantic fills; black text)
axa.add_patch(FancyBboxPatch((0.585,0.02),0.175,0.115,boxstyle="round,pad=0.004,rounding_size=0.02",fc="#eef7ee",ec="#3f8f3f",lw=0.9))
axa.text(0.672,0.098,"nowcast",ha="center",va="center",fontsize=8,color="#111111",fontweight="bold")
axa.text(0.672,0.05,"observed drivers",ha="center",va="center",fontsize=6.2,color="#111111")
axa.add_patch(FancyBboxPatch((0.775,0.02),0.215,0.115,boxstyle="round,pad=0.004,rounding_size=0.02",fc="#fdf0e8",ec=ACC,lw=0.9))
axa.text(0.882,0.098,"24-h forecast",ha="center",va="center",fontsize=8,color="#111111",fontweight="bold")
axa.text(0.882,0.05,"SWPC-predicted F10.7 + ap",ha="center",va="center",fontsize=6.0,color="#111111")
axa.annotate("a",xy=(0,1),xycoords="axes fraction",xytext=(-34,6),textcoords="offset points",fontsize=12,fontweight="bold",ha="left",va="bottom")

# ---------------- (b) the two model drivers over 2000-2024 (F10.7 top, Ap bottom) ----------------
gsb=GridSpecFromSubplotSpec(2,1,subplot_spec=gs[1,0],height_ratios=[1.0,0.72],hspace=0.18)
axb1=fig.add_subplot(gsb[0]); axb2=fig.add_subplot(gsb[1],sharex=axb1)
axb1.plot(f107_date,f107,color=SIG,lw=0.4,alpha=0.35)
sm=pd.Series(f107,index=pd.to_datetime(f107_date)).rolling(81,center=True,min_periods=20).mean()
axb1.plot(sm.index,sm.values,color=SIG,lw=1.6)
axb1.set_ylabel("F10.7 (sfu)",fontsize=7.5); axb1.set_ylim(0,None); axb1.tick_params(labelbottom=False,labelsize=7)
axb1.text(0.97,0.9,"solar (F10.7)",transform=axb1.transAxes,ha="right",va="top",fontsize=6.6,color=SIG)
axb2.fill_between(apd_date,apd_val,color=GEO,lw=0,alpha=0.30)
apsm=pd.Series(apd_val,index=pd.to_datetime(apd_date)).rolling(81,center=True,min_periods=20).mean()
axb2.plot(apsm.index,apsm.values,color=GEO,lw=1.4)
axb2.set_ylabel("Ap",fontsize=7.5); axb2.set_xlabel("year"); axb2.set_ylim(0,60); axb2.set_yticks([0,30,60]); axb2.tick_params(labelsize=7)
axb2.text(0.97,0.9,"geomagnetic (ap)",transform=axb2.transAxes,ha="right",va="top",fontsize=6.6,color=GEO)
axb2.set_xlim(np.datetime64("2000-01-01"),np.datetime64("2025-01-01"))
axb2.set_xticks(pd.to_datetime([f"{y}-01-01" for y in [2000,2008,2016,2024]])); axb2.set_xticklabels([2000,2008,2016,2024])
axb1.annotate("b",xy=(0,1),xycoords="axes fraction",xytext=(-34,6),textcoords="offset points",fontsize=12,fontweight="bold",ha="left",va="bottom")

# ---------------- (c) altitude distribution (bars coloured by altitude) ----------------
axc=fig.add_subplot(gs[1,1])
cb_edges=np.arange(225,545,8); cnt,_=np.histogram(alt,bins=cb_edges); cc=0.5*(cb_edges[:-1]+cb_edges[1:])
nrm=Normalize(cb_edges[0],cb_edges[-1])
axc.barh(cc,cnt,height=7.4,color=DCMAP(nrm(cc)),edgecolor="white",linewidth=0.3)
axc.set_ylabel("altitude (km)"); axc.set_xlabel("sample count (×10$^4$)"); axc.set_ylim(220,545)
axc.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v/1e4:.0f}"))
axc.annotate("c",xy=(0,1),xycoords="axes fraction",xytext=(-34,6),textcoords="offset points",fontsize=12,fontweight="bold",ha="left",va="bottom")

# ---------------- (d) latitude x local-time coverage ----------------
axd=fig.add_subplot(gs[1,2])
H,xe,ye=np.histogram2d(lst,lat,bins=[np.linspace(0,24,49),np.linspace(-90,90,49)])
im=axd.imshow(np.sqrt(H.T),origin="lower",extent=[0,24,-90,90],aspect="auto",cmap=DCMAP)
axd.set_xlabel("local time (h)"); axd.set_ylabel("latitude (°)"); axd.set_xlim(0,24); axd.set_ylim(-90,90)
axd.set_xticks([0,6,12,18,24]); axd.set_yticks([-90,-45,0,45,90])
cb=fig.colorbar(im,ax=axd,fraction=0.046,pad=0.03); cb.set_label("$\\sqrt{\\mathrm{count}}$",fontsize=7); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.6)
axd.annotate("d",xy=(0,1),xycoords="axes fraction",xytext=(-38,6),textcoords="offset points",fontsize=12,fontweight="bold",ha="left",va="bottom")

for ext in ["png","pdf","svg"]:
    fig.savefig(FIGS+"fig1."+ext,dpi=300 if ext=="png" else None,bbox_inches="tight")
print("saved fig1")
