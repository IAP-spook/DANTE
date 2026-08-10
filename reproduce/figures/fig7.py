#!/usr/bin/env python3
"""Figure 7 — retrieved vertical density profiles with uncertainty, validated across six typical
space-weather / orbital scenarios (solar activity, geomagnetic activity, local time, latitude). For each
regime DANTE (headline 4.85M 5-seed ensemble) and the empirical NRLMSIS are evaluated at the
observations' ACTUAL conditions on the held-out test set and binned by altitude. DANTE's median (line)
with its calibrated 90% predictive interval (band, conformal per Fig 3) tracks the observations (black)
across 220-500 km in every regime, while NRLMSIS departs, most at low solar activity and in storms.
Per-panel per-point MAPE quantifies the gap. Loads arch_cache_v2clean.npz + check5 ensemble + check5v2_karman_nrl.npz."""
import glob, numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.unicode_minus":False,
    "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":0.8,"legend.frameon":False})
BN="/home/dell/hdd8t/gravity_wave/_bench/"; FIGS="/home/dell/nanoclaw/groups/main/thermo_paper/figs/"
D90=0.0221  # conformal widening (dex) -> calibrated 90% interval (Fig 3, 5-seed ensemble)
A=np.load(BN+"arch_cache_v2clean.npz",allow_pickle=True); te=A["te"]
obs=A["rho"][te]; expo=A["expo"][te]; Xs=A["Xs"][te]
alt=Xs[:,0]; f107=Xs[:,1]; lat=Xs[:,6]; Ap=A["Ap"][te]
lst=(np.degrees(np.arctan2(Xs[:,7],Xs[:,8]))/15.0)%24.0
Qe=np.mean([np.load(f) for f in sorted(glob.glob(BN+"check5v2_transformer_s*_Q.npy"))],0)
med=expo*np.power(10.,Qe[:,2]); lo=expo*np.power(10.,Qe[:,0]-D90); hi=expo*np.power(10.,Qe[:,4]+D90)
nrl=np.load(BN+"check5v2_nrlmsis2.npz")["nrl2"]  # NRLMSIS 2.0
jb=np.load(BN+"check5v2_jb2008.npz")["jb"]
BLU="#2171b5"; GRY="#7f7f7f"; ORG="#E69F00"

SCEN=[("low solar activity",(f107<100)&(Ap<15)),
      ("high solar activity",(f107>=150)&(Ap<15)),
      ("geomagnetic storm  (ap\u226530)",Ap>=30),
      ("dayside  (10\u201316 h LT)",(lst>=10)&(lst<16)),
      ("nightside  (22\u201304 h LT)",(lst>=22)|(lst<4)),
      ("high latitude  (|lat|\u226560\u00b0)",np.abs(lat)>=60)]
edges=np.arange(220,545,40); ac=0.5*(edges[:-1]+edges[1:])
fig,axes=plt.subplots(2,3,figsize=(10.4,6.9),sharey=True)
axes=axes.ravel()
for ci,(nm,sel) in enumerate(SCEN):
    ax=axes[ci]; OB=[];DM=[];DL=[];DH=[];NR=[];JB=[]
    for i in range(len(edges)-1):
        b=sel&(alt>=edges[i])&(alt<edges[i+1])
        if b.sum()>=30:
            OB.append(np.median(obs[b])); DM.append(np.median(med[b])); DL.append(np.median(lo[b])); DH.append(np.median(hi[b])); NR.append(np.median(nrl[b])); JB.append(np.median(jb[b]))
        else: OB.append(np.nan);DM.append(np.nan);DL.append(np.nan);DH.append(np.nan);NR.append(np.nan);JB.append(np.nan)
    OB=np.array(OB);DM=np.array(DM);DL=np.array(DL);DH=np.array(DH);NR=np.array(NR);JB=np.array(JB); k=np.isfinite(OB)
    ax.fill_betweenx(ac[k],DL[k],DH[k],color=BLU,alpha=0.22,lw=0,label="DANTE 90% interval")
    ax.plot(DM[k],ac[k],"-",color=BLU,lw=2.0,label="DANTE median",zorder=4)
    ax.plot(NR[k],ac[k],"^--",color=GRY,lw=1.4,ms=5,label="NRLMSIS",zorder=3)
    ax.plot(JB[k],ac[k],"s--",color=ORG,lw=1.4,ms=4,label="JB2008",zorder=3)
    ax.scatter(OB[k],ac[k],s=30,color="black",zorder=5,label="observations")
    ax.set_xscale("log")
    if ci>=3: ax.set_xlabel("neutral density (kg m$^{-3}$)")
    if ci%3==0: ax.set_ylabel("altitude (km)")
    ax.set_ylim(220,520)
    mo=np.abs(med[sel]-obs[sel])/obs[sel]*100; mn=np.abs(nrl[sel]-obs[sel])/obs[sel]*100; mj=np.abs(jb[sel]-obs[sel])/obs[sel]*100
    ax.annotate(nm,xy=(0.5,1.02),xycoords="axes fraction",ha="center",va="bottom",fontsize=8.4,fontweight="bold")
    ax.text(0.04,0.03,"n=%d\nMAPE  DANTE %.0f%% | JB2008 %.0f%% | NRLMSIS %.0f%%"%(sel.sum(),mo.mean(),mj.mean(),mn.mean()),
            transform=ax.transAxes,fontsize=6.3,color="0.3",va="bottom")
    ax.annotate("abcdef"[ci],xy=(0,1),xycoords="axes fraction",xytext=(-34 if ci%3==0 else -12,6),
                textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")
    print("%-26s n=%6d  MAPE DANTE=%.1f JB2008=%.1f NRLMSIS=%.1f"%(nm,sel.sum(),mo.mean(),mj.mean(),mn.mean()))
leg=[Line2D([0],[0],color=BLU,lw=2,label="DANTE median"),
     mpl.patches.Patch(color=BLU,alpha=0.22,label="DANTE 90% interval"),
     Line2D([0],[0],color=ORG,lw=1.4,ls="--",marker="s",label="JB2008"),
     Line2D([0],[0],color=GRY,lw=1.4,ls="--",marker="^",label="NRLMSIS"),
     Line2D([0],[0],color="black",lw=0,marker="o",label="observations")]
fig.legend(handles=leg,loc="lower center",ncol=5,fontsize=7.5,handletextpad=0.4,columnspacing=1.4,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=[0,0.03,1,1])
for ext in ["png","pdf","svg"]:
    fig.savefig(FIGS+"fig7."+ext,dpi=300 if ext=="png" else None,bbox_inches="tight")
print("saved fig7 (6-scenario profiles)")
