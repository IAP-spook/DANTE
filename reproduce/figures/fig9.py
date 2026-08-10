#!/usr/bin/env python3
"""Figure 9 — observation-resolved validation across the domain (full held-out test set, n=105,461).
DANTE nowcast error against real satellite densities, resolved by geographic latitude x altitude.
(a) DANTE MAPE map — uniformly low across the whole domain (robustness). (b) improvement factor
= NRLMSIS MAPE / DANTE MAPE — DANTE is 2-4x more accurate than the empirical baseline everywhere,
most at high latitudes and high altitudes. Headline 4.85M 5-seed ensemble. Loads arch_cache_v2clean.npz +
check5 ensemble quantiles + check5v2_karman_nrl.npz."""
import glob, numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.unicode_minus":False,"axes.linewidth":0.8})
BN="/home/dell/hdd8t/gravity_wave/_bench/"; FIGS="/home/dell/nanoclaw/groups/main/thermo_paper/figs/"
A=np.load(BN+"arch_cache_v2clean.npz",allow_pickle=True); te=A["te"]
obs=A["rho"][te]; expo=A["expo"][te]; Xs=A["Xs"][te]; alt=Xs[:,0]; lat=Xs[:,6]
Qe=np.mean([np.load(f) for f in sorted(glob.glob(BN+"check5v2_transformer_s*_Q.npy"))],0)
med=expo*np.power(10.,Qe[:,2])
nrl=np.load(BN+"check5v2_nrlmsis2.npz")["nrl2"]  # NRLMSIS 2.0
ape_d=np.abs(med-obs)/obs*100; ape_n=np.abs(nrl-obs)/obs*100
latE=np.arange(-90,91,15); altE=np.arange(220,541,40)
nx=len(latE)-1; ny=len(altE)-1
Dg=np.full((ny,nx),np.nan); Rg=np.full((ny,nx),np.nan)
for i in range(ny):
    for j in range(nx):
        b=(alt>=altE[i])&(alt<altE[i+1])&(lat>=latE[j])&(lat<latE[j+1])
        if b.sum()>=40:
            Dg[i,j]=ape_d[b].mean(); Rg[i,j]=ape_n[b].mean()/ape_d[b].mean()
# bespoke premium colormaps, cohesive with the paper's navy-teal identity (a) + warm accent (b)
CMAP_D=mcolors.LinearSegmentedColormap.from_list("dante_mape",["#eef6f2","#a4d4c4","#57a6ab","#2f6f8f","#173f6b","#0a2340"])
CMAP_R=mcolors.LinearSegmentedColormap.from_list("dante_imp",["#f6efe8","#e8c49a","#d38a55","#b5532e","#7a3218"])

fig,(axa,axb)=plt.subplots(1,2,figsize=(9.4,3.9),constrained_layout=True)
im0=axa.pcolormesh(latE,altE,Dg,cmap=CMAP_D,vmin=8,vmax=24,shading="flat")
cb0=fig.colorbar(im0,ax=axa,pad=0.02,shrink=0.9); cb0.set_label("DANTE MAPE (%)",fontsize=8)
axa.set_xlabel("geographic latitude (\u00b0)"); axa.set_ylabel("altitude (km)")
axa.set_xticks([-90,-45,0,45,90]); axa.annotate("a",xy=(0,1),xycoords="axes fraction",
    xytext=(-40,6),textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")
im1=axb.pcolormesh(latE,altE,Rg,cmap=CMAP_R,vmin=1,vmax=4,shading="flat")
cb1=fig.colorbar(im1,ax=axb,pad=0.02,shrink=0.9,extend="max"); cb1.set_label("improvement factor  (NRLMSIS MAPE / DANTE MAPE)",fontsize=7.5)
axb.set_xlabel("geographic latitude (\u00b0)"); axb.set_yticklabels([])
axb.set_xticks([-90,-45,0,45,90]); axb.annotate("b",xy=(0,1),xycoords="axes fraction",
    xytext=(-14,6),textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")
for ext in ["png","pdf","svg"]:
    fig.savefig(FIGS+"fig9."+ext,dpi=300 if ext=="png" else None,bbox_inches="tight")
print("saved fig9  DANTE MAPE range %.1f-%.1f  improvement %.1f-%.1f"%(np.nanmin(Dg),np.nanmax(Dg),np.nanmin(Rg),np.nanmax(Rg)))
print("overall DANTE=%.2f NRLMSIS=%.2f"%(ape_d.mean(),ape_n.mean()))
