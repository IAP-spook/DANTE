#!/usr/bin/env python3
"""Figure 2 — nowcast accuracy of DANTE (4.85M, 5-seed ensemble) on the full test set.
Three panels in one row (a,b,c aligned): (a) observed vs predicted density log-log hexbin
with y=x; (b) relative-error distribution; (c) stratified MAPE, DANTE vs NRLMSIS baseline
(dumbbell). All quantitative statistics (R2, MAPE, median/IQR) live in the caption, not on
the figure. KML (Acciarini et al., 2024) is compared in Table 1, not drawn here.
Publication-grade: no in-figure titles/stats, black text, standardized units."""
import glob, numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.unicode_minus":False,
    "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":0.8,"legend.frameon":False})
BN="/home/dell/hdd8t/gravity_wave/_bench/"; FIGS="/home/dell/nanoclaw/groups/main/thermo_paper/figs/"
NAME="DANTE"; OUR="#2171b5"; NRL="#98a2ac"; JBc="#c8663a"; HISTc="#3f8fa0"; DUMB="#dfe4e8"; TXT="black"
# bespoke density colormap: medium teal (sparse) -> deep navy (dense), cohesive with DANTE's identity
HBCMAP=LinearSegmentedColormap.from_list("dante_hb",["#a9d4c8","#6bb6ab","#3f8fa0","#2b6f8c","#184e77","#0a2340"])

A=np.load(BN+"arch_cache_v2clean.npz",allow_pickle=True); te=A["te"]
obs=A["rho"][te]; expo=A["expo"][te]; Dst=A["Dst"][te]; Ap=A["Ap"][te]
Qe=np.mean([np.load(f) for f in sorted(glob.glob(BN+"check5v2_transformer_s*_Q.npy"))],0)
med=expo*np.power(10.,Qe[:,2])
nrl=np.load(BN+"check5v2_nrlmsis2.npz")["nrl2"]  # NRLMSIS 2.0
jb=np.load(BN+"check5v2_jb2008.npz")["jb"]
n=len(te)

# ---- stats (reported in caption, NOT drawn) ----
lo=np.log10(obs); lp=np.log10(med)
r2=1-np.sum((lo-lp)**2)/np.sum((lo-lo.mean())**2); mape=(np.abs(med-obs)/obs).mean()*100
res=(med-obs)/obs*100; medi=np.median(res); q1,q3=np.percentile(res,[25,75])

strata=[("all",np.ones(n,bool)),("Dst\u2264\u221250",Dst<=-50),
        ("Dst\u2264\u2212100",Dst<=-100),("ap\u2265100",Ap>=100)]
ape=lambda p,m:(np.abs(p[m]-obs[m])/obs[m]).mean()*100
ours=[ape(med,m) for _,m in strata]; nrlm=[ape(nrl,m) for _,m in strata]; jbm=[ape(jb,m) for _,m in strata]

fig=plt.figure(figsize=(10.0,3.1))
gs=GridSpec(1,3,width_ratios=[1.18,1.0,1.05],wspace=0.55,figure=fig)
def plabel(ax,s,dx=-46):
    ax.annotate(s,xy=(0,1),xycoords="axes fraction",xytext=(dx,7),textcoords="offset points",
                fontsize=12,fontweight="bold",ha="left",va="bottom",color=TXT)

# ================= (a) observed vs predicted =================
axa=fig.add_subplot(gs[0,0])
hb=axa.hexbin(obs,med,xscale="log",yscale="log",gridsize=46,bins="log",
              cmap=HBCMAP,mincnt=1,linewidths=0)
lim=[min(obs.min(),med.min()),max(obs.max(),med.max())]
axa.plot(lim,lim,"--",color="0.15",lw=1.0,zorder=4)
axa.set_xlim(lim); axa.set_ylim(lim)
axa.set_xlabel("observed density (kg m$^{-3}$)")
axa.set_ylabel("predicted density (kg m$^{-3}$)")
axa.set_xticks([1e-14,1e-12,1e-10]); axa.set_yticks([1e-14,1e-12,1e-10])
cax=make_axes_locatable(axa).append_axes("right",size="4.5%",pad=0.10)
cb=fig.colorbar(hb,cax=cax); cb.set_label("point count (log$_{10}$)",fontsize=7)
cb.ax.tick_params(labelsize=6.5)
plabel(axa,"a")

# ================= (b) relative-error distribution =================
axb=fig.add_subplot(gs[0,1])
axb.hist(res,bins=70,range=(-60,60),color=HISTc,alpha=0.92,edgecolor="white",linewidth=0.2)
axb.axvline(0,color="0.35",lw=0.9,ls="--")
axb.set_xlabel("relative error (%)"); axb.set_ylabel("count")
axb.set_xlim(-60,60); axb.set_xticks([-60,-30,0,30,60])
plabel(axb,"b")

# ================= (c) stratified MAPE: DANTE vs NRLMSIS (dumbbell) =================
axc=fig.add_subplot(gs[0,2])
yb=np.arange(len(strata))[::-1]
axc.set_axisbelow(True); axc.grid(axis="x",color="#eeeeee",lw=0.7,zorder=0)
for i,y in enumerate(yb):
    axc.plot([ours[i],nrlm[i]],[y,y],color=DUMB,lw=2.2,zorder=1,solid_capstyle="round")
    axc.scatter(nrlm[i],y,s=40,color=NRL,ec="white",lw=0.8,zorder=3)
    axc.scatter(jbm[i],y,s=34,color=JBc,ec="white",lw=0.8,zorder=3,marker="D")
    axc.scatter(ours[i],y,s=40,color=OUR,ec="white",lw=0.8,zorder=4)
axc.set_yticks(yb); axc.set_yticklabels([s[0] for s in strata]); axc.set_ylim(-0.6,len(strata)-0.4)
axc.set_xlim(8,62); axc.set_xticks([10,30,50]); axc.set_xlabel("MAPE (%)")
leg=[Line2D([0],[0],marker="o",ls="none",color=OUR,label=NAME),
     Line2D([0],[0],marker="D",ls="none",color=JBc,label="JB2008"),
     Line2D([0],[0],marker="o",ls="none",color=NRL,label="NRLMSIS")]
axc.legend(handles=leg,loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=3,
           fontsize=6.6,handletextpad=0.2,columnspacing=0.9)
plabel(axc,"c",dx=-40)

for ext in ["png","pdf","svg"]:
    fig.savefig(FIGS+"fig2."+ext,dpi=300 if ext=="png" else None,bbox_inches="tight")
print("saved fig2  R2=%.3f MAPE=%.2f n=%d median=%.1f IQR=[%.0f,%.0f]"%(r2,mape,n,medi,q1,q3))
print("ours",[round(v,2) for v in ours],"nrl",[round(v,2) for v in nrlm])
