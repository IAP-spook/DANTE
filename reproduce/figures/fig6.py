#!/usr/bin/env python3
"""Figure 6 — architecture study in ONE panel: accuracy-efficiency frontier of a controlled backbone
bake-off (identical 1.2M data, features, quantile-UQ, 5-seed ensembles). x = training time (log);
y = nowcast MAPE; marker area = parameters; colour = 90% interval coverage (calibration; ideal 90).
A shaded band marks the narrow MAPE span of the competitive backbones (architecture = minor lever);
reference lines mark KML (SOTA, 14.11) and DANTE (Transformer) after data scaling to 4.85M (13.93),
with an arrow showing that the decisive gain comes from DATA scaling, not architecture. Numbers locked
(memory 5756)."""
import numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.unicode_minus":False,
    "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":0.8,"legend.frameon":False})
FIGS="/home/dell/nanoclaw/groups/main/thermo_paper/figs/"
# name, MAPE(%), cal90(%), params(K), train time(min)  [5-seed ensemble, 1.2M; memory 5756]
M=[("Transformer",14.54,87.2,186,16.8),("TFT-lite",14.42,81.4,188,8.3),("MLP",15.22,89.7,87,0.2),
   ("LSTM",15.14,88.5,181,5.7),("TCN",15.17,86.1,148,9.8),("GBDT",16.66,75.3,2.3,1.2),("S4D",17.92,89.4,57,9.0)]
name=[m[0] for m in M]; mape=np.array([m[1] for m in M]); cal=np.array([m[2] for m in M])
par=np.array([m[3] for m in M]); tmin=np.array([m[4] for m in M])
KAR=14.11; DANTE=13.91; OUR="#0072B2"

fig,ax=plt.subplots(figsize=(6.8,5.0))
# competitive-backbone MAPE band (the neural/MLP cluster)
comp=mape[mape<=15.5]; ax.axhspan(comp.min(),comp.max(),color="#eef3f7",zorder=0)
ax.text(0.12,(comp.min()+comp.max())/2,"competitive backbones span < 1% MAPE",fontsize=6.8,color="0.5",va="center",ha="left")
# reference lines
ax.axhline(DANTE,ls="-",color=OUR,lw=1.4,zorder=1)
# data-scaling arrow: Transformer @1.2M  ->  DANTE @4.85M
ax.annotate("",xy=(tmin[0],DANTE+0.03),xytext=(tmin[0],mape[0]-0.05),
            arrowprops=dict(arrowstyle="-|>",color=OUR,lw=1.8,shrinkA=6,shrinkB=2),zorder=5)
ax.text(tmin[0]*1.28,(mape[0]+DANTE)/2+0.02,"data scaling\n1.2 M \u2192 4.85 M",fontsize=7,color=OUR,va="center",ha="left")
ax.text(0.13,DANTE+0.04,"DANTE (4.85 M data)  13.91",fontsize=7.2,color=OUR,va="bottom",ha="left",fontweight="bold")
# bubbles
sizes=(par-par.min())/(par.max()-par.min())*300+50
sc=ax.scatter(tmin,mape,s=sizes,c=cal,cmap="GnBu",vmin=74,vmax=90,edgecolor="0.25",lw=0.8,zorder=3)
ax.scatter(tmin[0],mape[0],s=sizes[0]+170,facecolors="none",edgecolors=OUR,lw=2.0,zorder=4)  # chosen backbone
# labels (hand-placed, no occlusion)
lab={"Transformer":(0,13,"center"),"TFT-lite":(-6,-12,"right"),"MLP":(10,0,"left"),
     "LSTM":(0,11,"center"),"TCN":(4,-12,"left"),"GBDT":(9,4,"left"),"S4D":(10,2,"left")}
for n,x,y in zip(name,tmin,mape):
    dx,dy,h=lab[n]; ax.annotate(n,(x,y),xytext=(dx,dy),textcoords="offset points",fontsize=7.4,color="0.12",ha=h)
ax.set_xscale("log"); ax.set_xlim(0.1,55); ax.set_ylim(13.5,18.5)
ax.set_xlabel("training time (min, log scale)"); ax.set_ylabel("nowcast MAPE (%)")
cb=fig.colorbar(sc,ax=ax,pad=0.02,shrink=0.85); cb.set_label("90% interval coverage (%)   [ideal 90]",fontsize=7.5)
cb.ax.tick_params(labelsize=7)
# size legend
for pv,lb in [(50,"~50 K"),(186,"~190 K")]:
    ax.scatter([],[],s=(pv-par.min())/(par.max()-par.min())*300+50,c="0.75",edgecolor="0.25",lw=0.7,label="params "+lb)
ax.legend(loc="upper left",fontsize=6.8,labelspacing=1.3,borderpad=0.8,handletextpad=1.0)
fig.tight_layout()
for ext in ["png","pdf","svg"]:
    fig.savefig(FIGS+"fig6."+ext,dpi=300 if ext=="png" else None,bbox_inches="tight")
print("saved fig6 (single-panel efficiency frontier)")
