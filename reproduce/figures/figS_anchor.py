#!/usr/bin/env python3
"""Fig S7 — physics-anchor ablation. Removing the Vallado exponential anchor (predicting absolute
log10(rho) directly instead of the anomaly y = log10 rho - log10 rho_exp), with everything else
identical (4.85M data, same features, 30 ep, 5-seed protocol), makes the model learn the full six-decade
altitude dependence from scratch: it converges far more slowly and to a worse, more seed-variable optimum.
(a) per-epoch test MAPE (seed 0) for the anchored headline vs the no-anchor model.
(b) final 5-seed-ensemble MAPE by stratum, anchored vs no-anchor."""
import numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.unicode_minus":False,
    "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":0.8,"legend.frameon":False})
BN="/home/dell/hdd8t/gravity_wave/_bench/"; FIGS="/home/dell/nanoclaw/groups/main/thermo_paper/figs/"
ANC="#2171b5"; NOA="#d1495b"
# anchored per-epoch test MAPE (seed 0, headline pipeline)
L=np.load(BN+"loss_curve_v2.npz"); ep=L["ep"]; anc=L["test_mape"]
# no-anchor per-epoch test MAPE (seed 0), from the ablation run
noa=np.array([85.44,70.37,57.06,50.14,44.67,43.49,41.50,41.62,41.22,45.46,40.03,39.71,37.62,36.00,
              33.37,31.21,29.67,27.74,26.77,26.71,24.77,25.22,24.71,23.99,23.24,23.23,23.03,23.27,22.86,22.95])
# final 5-seed ensemble MAPE by stratum
STR=["All","Dst\u2264\u221250","Dst\u2264\u2212100","ap\u2265100"]
anc5=[13.91,17.60,22.45,21.66]; noa5=[19.84,22.37,28.24,27.05]

fig,(axa,axb)=plt.subplots(1,2,figsize=(8.6,3.2))
# (a) convergence
axa.plot(ep,anc,"-",color=ANC,lw=2.0,label="with physics anchor (headline)")
axa.plot(np.arange(len(noa)),noa,"-",color=NOA,lw=2.0,label="no anchor (predict log$_{10}\\rho$)")
axa.axhline(13.91,ls=":",lw=1.0,color=ANC,alpha=0.6)
axa.set_yscale("log"); axa.set_ylim(11,95); axa.set_yticks([15,20,30,50,80]); axa.set_yticklabels([15,20,30,50,80])
axa.set_xlabel("training epoch"); axa.set_ylabel("test MAPE (%)"); axa.set_xlim(0,29)
axa.legend(loc="upper right",fontsize=7,handletextpad=0.5)
axa.annotate("a",xy=(0,1),xycoords="axes fraction",xytext=(-38,6),textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")
# (b) final by stratum
x=np.arange(len(STR)); w=0.37
axb.bar(x-w/2,anc5,w,color=ANC,label="with anchor")
axb.bar(x+w/2,noa5,w,color=NOA,label="no anchor")
for xi,(a,b) in enumerate(zip(anc5,noa5)):
    axb.text(xi-w/2,a+0.4,"%.1f"%a,ha="center",va="bottom",fontsize=6.6,color=ANC)
    axb.text(xi+w/2,b+0.4,"%.1f"%b,ha="center",va="bottom",fontsize=6.6,color=NOA)
axb.set_xticks(x); axb.set_xticklabels(STR,fontsize=7.4); axb.set_ylabel("5-seed MAPE (%)"); axb.set_ylim(0,32)
axb.legend(loc="upper left",fontsize=7,handletextpad=0.5)
axb.annotate("b",xy=(0,1),xycoords="axes fraction",xytext=(-36,6),textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")
fig.tight_layout()
for ext in ["png","pdf","svg"]:
    fig.savefig(FIGS+"figS_anchor."+ext,dpi=300 if ext=="png" else None,bbox_inches="tight")
print("saved figS_anchor  anchored final ep MAPE=%.2f  no-anchor final=%.2f"%(anc[-1],noa[-1]))
