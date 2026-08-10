#!/usr/bin/env python3
"""Fig S8 — the physics-anchor residual is latitude-robust. The Vallado exponential atmosphere is a
latitude-independent climatology, so one might expect the residual y = log10(rho_obs) - log10(rho_exp)
that the network must learn to grow and broaden toward the poles (particle-precipitation region). It does
not: the residual's median and spread are nearly latitude-independent, so the network's learning task is
comparably hard at all latitudes, consistent with DANTE's uniform accuracy across latitude (Fig 5, Fig 6).
Test set (n=105,461); storm = ap>=30."""
import numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.unicode_minus":False,
    "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":0.8,"legend.frameon":False})
BN="/home/dell/hdd8t/gravity_wave/_bench/"; FIGS="/home/dell/nanoclaw/groups/main/thermo_paper/figs/"
QC="#2171b5"; SC="#d1495b"
A=np.load(BN+"arch_cache_v2clean.npz",allow_pickle=True); te=A["te"]
y=A["y"][te]; lat=A["Xs"][te][:,6]; Ap=A["Ap"][te]
edges=np.arange(-90,91,15); cen=0.5*(edges[:-1]+edges[1:])
def stats(mask):
    med=np.full(len(cen),np.nan); lo=np.full(len(cen),np.nan); hi=np.full(len(cen),np.nan); sp=np.full(len(cen),np.nan)
    for i in range(len(cen)):
        b=mask&(lat>=edges[i])&(lat<edges[i+1])
        if b.sum()>=40:
            q10,q25,q50,q75,q90=np.percentile(y[b],[10,25,50,75,90]); med[i]=q50; lo[i]=q25; hi[i]=q75; sp[i]=q90-q10
    return med,lo,hi,sp
mq,lq,hq,spq=stats(Ap<15); ms,ls,hs,sps=stats(Ap>=30)

fig,(axa,axb)=plt.subplots(1,2,figsize=(8.6,3.2))
# (a) median + IQR band vs latitude, quiet vs storm
k=np.isfinite(mq)
axa.fill_between(cen[k],lq[k],hq[k],color=QC,alpha=0.20,lw=0)
axa.plot(cen[k],mq[k],"-o",color=QC,lw=1.8,ms=4,label="quiet (ap<15)")
ks=np.isfinite(ms)
axa.plot(cen[ks],ms[ks],"-s",color=SC,lw=1.8,ms=4,label="storm (ap\u226530)")
axa.axhline(0,ls=":",lw=1,color="0.5")
axa.set_xlabel("geographic latitude (\u00b0)"); axa.set_ylabel(r"anchor residual  log$_{10}\rho_{\rm obs}-$log$_{10}\rho_{\rm exp}$")
axa.set_xticks([-60,-30,0,30,60]); axa.set_xlim(-90,90)
axa.legend(loc="upper right",fontsize=7,handletextpad=0.4)
axa.annotate("a",xy=(0,1),xycoords="axes fraction",xytext=(-44,6),textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")
# (b) residual spread (10-90%) vs latitude -> flat, does not grow at poles
axb.plot(cen[k],spq[k],"-o",color=QC,lw=1.8,ms=4,label="quiet")
axb.plot(cen[ks],sps[ks],"-s",color=SC,lw=1.8,ms=4,label="storm")
axb.set_ylim(0,1.7); axb.set_xlabel("geographic latitude (\u00b0)"); axb.set_ylabel("residual 10–90% spread (dex)")
axb.set_xticks([-60,-30,0,30,60]); axb.set_xlim(-90,90)
axb.legend(loc="lower center",fontsize=7,handletextpad=0.4,ncol=2)
axb.annotate("b",xy=(0,1),xycoords="axes fraction",xytext=(-40,6),textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")
fig.tight_layout()
for ext in ["png","pdf","svg"]:
    fig.savefig(FIGS+"figS_residlat."+ext,dpi=300 if ext=="png" else None,bbox_inches="tight")
print("saved figS_residlat")
print("quiet spread by lat:",[round(v,2) for v in spq])
print("storm spread by lat:",[round(v,2) for v in sps])
