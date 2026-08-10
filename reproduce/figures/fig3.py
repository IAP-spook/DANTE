#!/usr/bin/env python3
"""Figure 3 — calibrated uncertainty of DANTE (5-seed ensemble). Split-conformal (CQR) on an
in-distribution held-out fraction of the test set restores nominal coverage.
(a) reliability: nominal quantile level vs empirical coverage-below, raw vs conformal + ideal;
(b) central-interval coverage (50%,90%) raw vs conformal with nominal targets;
(c) sharpness/adaptivity: mean 90% prediction-interval relative width by geomagnetic stratum.
All numbers printed for the record; figure carries no in-panel stats."""
import glob, numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.unicode_minus":False,
    "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":0.8,"legend.frameon":False})
BN="/home/dell/hdd8t/gravity_wave/_bench/"; FIGS="/home/dell/nanoclaw/groups/main/thermo_paper/figs/"
NAME="DANTE"; RAWC="#c8663a"; CONC="#2171b5"; TXT="black"; QS=np.array([0.05,0.25,0.5,0.75,0.95])
# teal->navy sequential for the (c) interval-width bars (encodes storm severity), cohesive with Fig 1/3
TEALS=LinearSegmentedColormap.from_list("dante_teal",["#a4d4c4","#57a6ab","#2f6f8f","#173f6b"])

A=np.load(BN+"arch_cache_v2clean.npz",allow_pickle=True); te=A["te"]
obs=A["rho"][te]; expo=A["expo"][te]; Dst=A["Dst"][te]; Ap=A["Ap"][te]
Qe=np.mean([np.load(f) for f in sorted(glob.glob(BN+"check5v2_transformer_s*_Q.npy"))],0)  # (n,5) ensemble
le=np.log10(expo); rr=np.log10(obs); n=len(te)

# ---- split-conformal on in-distribution held-out fraction ----
rng=np.random.default_rng(0); idx=rng.permutation(n); ncal=int(0.20*n); cal=idx[:ncal]; ev=idx[ncal:]
def cqr(loQ,hiQ,alpha,m):  # widening delta from calibration set m
    lo=le[m]+Qe[m,loQ]; hi=le[m]+Qe[m,hiQ]; E=np.maximum(lo-rr[m],rr[m]-hi)
    k=min(int(np.ceil((len(E)+1)*(1-alpha))),len(E)); return np.sort(E)[k-1]
d90=cqr(0,4,0.10,cal); d50=cqr(1,3,0.50,cal)

# ---- (a) per-quantile reliability on eval ----
covbelow=lambda q: np.mean(rr[ev]<=le[ev]+q[ev])
raw_pq=[covbelow(Qe[:,i]) for i in range(5)]
shift=[-d90,-d50,0.0,d50,d90]
con_pq=[covbelow(Qe[:,i]+shift[i]) for i in range(5)]

# ---- (b) central-interval coverage on eval ----
def covint(loQ,hiQ,d):
    lo=le[ev]+Qe[ev,loQ]-d; hi=le[ev]+Qe[ev,hiQ]+d; return np.mean((rr[ev]>=lo)&(rr[ev]<=hi))
cov50_raw=covint(1,3,0); cov90_raw=covint(0,4,0); cov50_con=covint(1,3,d50); cov90_con=covint(0,4,d90)

# ---- (c) sharpness: mean 90% PI relative width (%) by stratum, conformal ----
relw=(np.power(10.,Qe[:,4]+d90)-np.power(10.,Qe[:,0]-d90))/np.power(10.,Qe[:,2])*100
evm=np.zeros(n,bool); evm[ev]=True
strata=[("all",np.ones(n,bool)),("Dst\u2264\u221250",Dst<=-50),
        ("Dst\u2264\u2212100",Dst<=-100),("ap\u2265100",Ap>=100)]
width=[relw[m&evm].mean() for _,m in strata]

print("d90=%.4f d50=%.4f dex  ncal=%d nev=%d"%(d90,d50,ncal,len(ev)))
print("raw per-quantile  ",[round(v,3) for v in raw_pq])
print("conf per-quantile ",[round(v,3) for v in con_pq])
print("cov50 raw=%.1f con=%.1f | cov90 raw=%.1f con=%.1f"%(cov50_raw*100,cov50_con*100,cov90_raw*100,cov90_con*100))
print("90%% PI rel width by stratum:",[round(v,1) for v in width])

# ================= FIGURE =================
fig=plt.figure(figsize=(10.0,3.1))
gs=GridSpec(1,3,width_ratios=[1.0,1.0,1.05],wspace=0.42,figure=fig)
def plabel(ax,s,dx=-46):
    ax.annotate(s,xy=(0,1),xycoords="axes fraction",xytext=(dx,7),textcoords="offset points",
                fontsize=12,fontweight="bold",ha="left",va="bottom",color=TXT)

# (a) reliability
axa=fig.add_subplot(gs[0,0])
axa.plot([0,1],[0,1],"--",color="0.35",lw=1.0)
axa.plot(QS,raw_pq,"o-",color=RAWC,lw=1.3,ms=5,label="raw quantiles")
axa.plot(QS,con_pq,"s-",color=CONC,lw=1.3,ms=5,label="conformalized")
axa.set_xlim(0,1); axa.set_ylim(0,1); axa.set_xticks([0,0.25,0.5,0.75,1.0]); axa.set_yticks([0,0.25,0.5,0.75,1.0])
axa.set_xlabel("nominal quantile level"); axa.set_ylabel("empirical fraction below")
axa.legend(loc="upper left",fontsize=7,handletextpad=0.4)
plabel(axa,"a")

# (b) interval coverage bars
axb=fig.add_subplot(gs[0,1])
xb=np.arange(2); w=0.36
axb.bar(xb-w/2,[cov50_raw*100,cov90_raw*100],w,color=RAWC,label="raw")
axb.bar(xb+w/2,[cov50_con*100,cov90_con*100],w,color=CONC,label="conformalized")
for xt,tv in zip(xb,[50,90]):
    axb.hlines(tv,xt-0.5,xt+0.5,color="0.2",lw=1.1,ls=":",zorder=5)
axb.set_xticks(xb); axb.set_xticklabels(["50% interval","90% interval"])
axb.set_ylabel("empirical coverage (%)"); axb.set_ylim(0,100)
axb.legend(loc="upper left",fontsize=7,handletextpad=0.4)
plabel(axb,"b",dx=-42)

# (c) sharpness by stratum
axc=fig.add_subplot(gs[0,2])
yb=np.arange(len(strata))[::-1]
axc.set_axisbelow(True); axc.grid(axis="x",color="#eeeeee",lw=0.7,zorder=0)
_wn=Normalize(min(width)*0.96,max(width)*1.02)
axc.barh(yb,width,color=[TEALS(_wn(w)) for w in width],height=0.58,edgecolor="white",linewidth=0.4,zorder=2)
axc.set_yticks(yb); axc.set_yticklabels([s[0] for s in strata]); axc.set_ylim(-0.6,len(strata)-0.4)
axc.set_xlabel("90% interval width (% of density)")
plabel(axc,"c",dx=-40)

for ext in ["png","pdf","svg"]:
    fig.savefig(FIGS+"fig3."+ext,dpi=300 if ext=="png" else None,bbox_inches="tight")
print("saved fig3")
