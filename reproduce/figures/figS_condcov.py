#!/usr/bin/env python3
"""Fig S6 — conditional coverage of the conformal 90% interval by geomagnetic severity.
Marginal coverage is near-nominal (89.7%), but conditional coverage declines in the strongest storms
(down to ~84% at Dst<=-100 nT), i.e. the split-conformal guarantee is marginal, not conditional, and the
intervals are mildly undercovered in the rare out-of-distribution storm tail. Same 5-seed ensemble,
same in-distribution calibration (20% hold-out) as Fig 4."""
import glob, numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.unicode_minus":False,
    "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":0.8,"legend.frameon":False})
BN="/home/dell/hdd8t/gravity_wave/_bench/"; FIGS="/home/dell/nanoclaw/groups/main/thermo_paper/figs/"
TEALS=LinearSegmentedColormap.from_list("dante_teal",["#a4d4c4","#57a6ab","#2f6f8f","#173f6b"])
A=np.load(BN+"arch_cache_v2clean.npz",allow_pickle=True); te=A["te"]
obs=A["rho"][te]; expo=A["expo"][te]; Dst=A["Dst"][te]; Ap=A["Ap"][te]
Qe=np.mean([np.load(f) for f in sorted(glob.glob(BN+"check5v2_transformer_s*_Q.npy"))],0)
le=np.log10(expo); rr=np.log10(obs); n=len(te)
rng=np.random.default_rng(0); idx=rng.permutation(n); ncal=int(0.20*n); cal=idx[:ncal]; ev=idx[ncal:]
lo_c=le[cal]+Qe[cal,0]; hi_c=le[cal]+Qe[cal,4]; E=np.maximum(lo_c-rr[cal],rr[cal]-hi_c)
k=min(int(np.ceil((len(E)+1)*0.90)),len(E)); d90=np.sort(E)[k-1]
inside=(rr>=le+Qe[:,0]-d90)&(rr<=le+Qe[:,4]+d90); evm=np.zeros(n,bool); evm[ev]=True
def cov(mask): m=mask&evm; return 100*inside[m].mean(), int(m.sum())
APB=[(0,15,"<15"),(15,30,"15–30"),(30,50,"30–50"),(50,100,"50–100"),(100,1e9,"≥100")]
DSB=[(-30,1e9,">−30"),(-50,-30,"−50…−30"),(-100,-50,"−100…−50"),(-1e9,-100,"≤−100")]
apc=[cov((Ap>=a)&(Ap<b)) for a,b,_ in APB]; dsc=[cov((Dst>a)&(Dst<=b)) for a,b,_ in DSB]
ov,_=cov(np.ones(n,bool))
print("overall 90%% cov=%.1f d90=%.4f"%(ov,d90))
print("ap :",[(l,round(c,1),nn) for (a,b,l),(c,nn) in zip(APB,apc)])
print("Dst:",[(l,round(c,1),nn) for (a,b,l),(c,nn) in zip(DSB,dsc)])

fig,(axa,axb)=plt.subplots(1,2,figsize=(8.4,3.2))
def panel(ax,data,labs,xlabel,sev):
    y=[c for c,_ in data]; nn=[m for _,m in data]; x=np.arange(len(y))
    nw=Normalize(0,len(y)-1); cols=[TEALS(nw(i)) for i in range(len(y))]
    ax.axhline(90,ls="--",lw=1.1,color="0.35",zorder=1)
    ax.bar(x,y,width=0.62,color=cols,edgecolor="white",linewidth=0.5,zorder=2)
    for xi,(yi,ni) in enumerate(zip(y,nn)):
        ax.text(xi,yi+0.3,"%.1f"%yi,ha="center",va="bottom",fontsize=7,color="0.15")
        ax.text(xi,79.2,"n=%s"%(f"{ni:,}"),ha="center",va="bottom",fontsize=5.8,color="0.5",rotation=0)
    ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=7.4)
    ax.set_ylim(79,93); ax.set_yticks([80,85,90]); ax.set_xlabel(xlabel)
    ax.text(0.02,0.93,"nominal 90%",transform=ax.transAxes,fontsize=6.8,color="0.35")
axa.set_ylabel("empirical 90% coverage (%)")
panel(axa,apc,[l for *_,l in APB],"geomagnetic ap",True)
panel(axb,dsc,[l for *_,l in DSB],"Dst (nT)",True)
axa.annotate("a",xy=(0,1),xycoords="axes fraction",xytext=(-34,6),textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")
axb.annotate("b",xy=(0,1),xycoords="axes fraction",xytext=(-30,6),textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")
fig.tight_layout()
for ext in ["png","pdf","svg"]:
    fig.savefig(FIGS+"figS_condcov."+ext,dpi=300 if ext=="png" else None,bbox_inches="tight")
print("saved figS_condcov")
