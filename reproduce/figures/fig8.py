#!/usr/bin/env python3
"""Figure 8 — DANTE tracks storm enhancement AND post-storm OVERCOOLING that the empirical model misses.
Nine well-sampled superstorms (2001-2024) spanning the solar cycle and the CHAMP/GRACE/GOCE/Swarm eras,
in a 3x3 grid. Daily neutral-density anomaly relative to the pre-storm baseline: observations (black),
DANTE deployed 5-seed ensemble (blue), NRLMSIS (grey), with the 3-hourly ap peak (bars, right axis).
During recovery the density falls below the baseline (overcooling, an NO/CO2 infrared process); DANTE
follows the depletion (recovery median bias ~1.0) while NRLMSIS stays biased high (up to ~1.7). DANTE is
driver-only, so the deployed model cannot memorise individual events. Loads fig8_overcool.npz."""
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.unicode_minus":False,
    "axes.spines.top":False,"axes.linewidth":0.8,"legend.frameon":False})
P="/home/dell/nanoclaw/groups/main/thermo_paper/"; FIGS=P+"figs/"
G=np.load(P+"fig8_overcool_nrl2.npz",allow_pickle=True); nev=int(G["nev"])  # NRLMSIS 2.0
DAN="#2171b5"; NRLc="#8a8a8a"; APc="#dbe3ec"
ncol=3; nrow=int(np.ceil(nev/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(3.5*ncol,2.9*nrow))
axes=np.atleast_2d(axes)
for ci in range(nev):
    r,c=divmod(ci,ncol); ax=axes[r,c]; k="ev%d"%ci
    dts=pd.to_datetime([str(x) for x in G[k+"_dates"]]); cen=pd.Timestamp(str(G[k+"_center"]))
    ob=G[k+"_obsnorm"]; da=G[k+"_danom"]; na=G[k+"_nanom"]; ap=G[k+"_ap"]; nm=str(G[k+"_name"])
    a2=ax.twinx(); a2.bar(dts,ap,width=0.9,color=APc,zorder=0); a2.set_ylim(0,max(np.nanmax(ap)*1.05,10))
    a2.spines["top"].set_visible(False)
    if c==ncol-1: a2.set_ylabel("ap index",color="0.55",fontsize=7.5)
    a2.tick_params(axis="y",labelcolor="0.55",labelsize=6.5)
    ax.set_zorder(a2.get_zorder()+1); ax.patch.set_visible(False)
    ax.axhline(1,color="0.5",ls=":",lw=1,zorder=2)
    ax.plot(dts,na,"^--",color=NRLc,lw=1.2,ms=3.5,label="NRLMSIS 2.0",zorder=3)
    ax.plot(dts,da,"-",color=DAN,lw=2.0,zorder=4,label="DANTE")
    ax.plot(dts,ob,"o",color="black",ms=3.8,zorder=5,label="observations")
    ax.axvline(cen,color="#7a3fb0",ls="--",lw=1,alpha=0.7,zorder=2)
    if c==0: ax.set_ylabel("density / pre-storm baseline",fontsize=7.8)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d")); ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=4))
    for lb in ax.get_xticklabels(): lb.set_rotation(28); lb.set_ha("right"); lb.set_fontsize(6.8)
    ax.set_title(nm,fontsize=8.4,fontweight="bold",pad=2)
    ax.annotate("abcdefghi"[ci],xy=(0,1),xycoords="axes fraction",xytext=(-30 if c==0 else -8,7),
                textcoords="offset points",fontsize=11,fontweight="bold",va="bottom")
for ci in range(nev,nrow*ncol):
    r,c=divmod(ci,ncol); axes[r,c].axis("off")
leg=[Line2D([0],[0],color="black",lw=0,marker="o",ms=5,label="Observations"),
     Line2D([0],[0],color=DAN,lw=2.2,label="DANTE (deployed 5-seed ensemble)"),
     Line2D([0],[0],color=NRLc,lw=1.3,ls="--",marker="^",ms=5,label="NRLMSIS 2.0"),
     Line2D([0],[0],color="0.5",ls=":",lw=1,label="Pre-storm baseline"),
     Patch(facecolor=APc,label="ap index (right axis)")]
fig.legend(handles=leg,loc="lower center",ncol=5,fontsize=8,handletextpad=0.5,columnspacing=1.5,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=[0,0.045,1,1])
for ext in ["png","pdf","svg"]:
    fig.savefig(FIGS+"fig8."+ext,dpi=300 if ext=="png" else None,bbox_inches="tight")
print("saved fig8 (3x3, %d storms, deployed ensemble)"%nev)
for ci in range(nev):
    k="ev%d"%ci; rec=np.array([pd.Timestamp(str(x))>=pd.Timestamp(str(G[k+"_center"]))+pd.Timedelta("3D") for x in G[k+"_dates"]])
    print("  %-24s recovery: obs-min=%.2f DANTE-bias=%.2f NRLMSIS-bias=%.2f"%(
        str(G[k+"_name"]),np.nanmin(G[k+"_obsnorm"][rec]),np.nanmedian(G[k+"_bo"][rec]),np.nanmedian(G[k+"_bn"][rec])))
