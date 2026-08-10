#!/usr/bin/env python3
"""Figure 4 - observation vs model latitudinal density structure at three real satellite
altitudes, under a fixed solar-minimum regime (F10.7 < 90). Rebuilt from the SAME held-out
test set as Fig 9 (arch_cache_v2clean.npz 'te' index) with the headline 4.85M 5-seed DANTE ensemble
(check5 quantiles) and NRLMSIS (check5v2_karman_nrl.npz), all sample-matched point-for-point.
Three altitude layers separate three missions: GOCE ~270 km, CHAMP ~315 km, GRACE/Swarm ~480 km.
Top row: zonal-mean density vs geographic latitude (obs black + 1 sigma band, DANTE blue,
NRLMSIS grey). Bottom row: per-model bias median(model/obs) vs latitude. Solar-minimum climatology
(storm response is Fig 8); genuinely out-of-sample."""
import glob, numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.unicode_minus":False,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8,"legend.frameon":False})
BN="/home/dell/hdd8t/gravity_wave/_bench/"; FIGS="/home/dell/nanoclaw/groups/main/thermo_paper/figs/"
DAN="#2171b5"; NRLc="#7f7f7f"; OBSc="black"; JBc="#E69F00"

# ---- load held-out test set (same as Fig9) ----
A=np.load(BN+"arch_cache_v2clean.npz",allow_pickle=True); te=A["te"]
obs=A["rho"][te]; expo=A["expo"][te]; Xs=A["Xs"][te]
alt=Xs[:,0]; f107=Xs[:,1]; lat=Xs[:,6]
Qe=np.mean([np.load(f) for f in sorted(glob.glob(BN+"check5v2_transformer_s*_Q.npy"))],0)
dan=expo*np.power(10.,Qe[:,2])                       # DANTE ensemble median
nrl=np.load(BN+"check5v2_nrlmsis2.npz")["nrl2"]        # NRLMSIS 2.0
jb=np.load(BN+"check5v2_jb2008.npz")["jb"]             # JB2008
assert len(dan)==len(te)==len(nrl)==len(obs)==len(jb), (len(dan),len(te),len(nrl),len(obs),len(jb))

# ---- fixed physical regime: deep solar minimum ----
reg=(f107>0)&(f107<90)
BANDS=[("GOCE",250,295),("CHAMP",300,345),("GRACE / Swarm",450,510)]
EDGES=np.arange(-90,91,15); CEN=0.5*(EDGES[:-1]+EDGES[1:]); MINN=25

def binstat(v,la):
    idx=np.digitize(la,EDGES)-1
    mean=np.full(len(CEN),np.nan); q25=np.full(len(CEN),np.nan); q75=np.full(len(CEN),np.nan); cnt=np.zeros(len(CEN),int)
    for k in range(len(CEN)):
        b=idx==k; cnt[k]=b.sum()
        if b.sum()>=MINN:
            mean[k]=np.nanmean(v[b]); q25[k],q75[k]=np.nanpercentile(v[b],[25,75])
    return mean,q25,q75,cnt

fig,axes=plt.subplots(2,3,figsize=(10,5.7),sharex=True,
                      gridspec_kw=dict(height_ratios=[2.6,1.0],hspace=0.12,wspace=0.32))
for j,(nm,a0,a1) in enumerate(BANDS):
    m=reg&(alt>=a0)&(alt<a1)
    la=lat[m]; ob=obs[m]; da=dan[m]; nr=nrl[m]; jbv=jb[m]; amean=alt[m].mean(); n=m.sum()
    ob_m,ob_lo,ob_hi,cnt=binstat(ob,la); da_m,_,_,_=binstat(da,la); nr_m,_,_,_=binstat(nr,la); jb_m,_,_,_=binstat(jbv,la)
    ok=cnt>=MINN
    # per-panel density scale
    sc=10**np.floor(np.log10(np.nanmedian(ob_m))); kk=int(np.round(np.log10(sc)))
    ax=axes[0,j]
    ax.fill_between(CEN[ok],ob_lo[ok]/sc,ob_hi[ok]/sc,color="0.6",alpha=0.20,lw=0,zorder=1)
    ax.plot(CEN[ok],nr_m[ok]/sc,"--",color=NRLc,lw=1.5,marker="^",ms=4,zorder=3,label="NRLMSIS")
    ax.plot(CEN[ok],jb_m[ok]/sc,"--",color=JBc,lw=1.5,marker="s",ms=3.5,zorder=3,label="JB2008")
    ax.plot(CEN[ok],da_m[ok]/sc,"-",color=DAN,lw=2.2,zorder=4,label="DANTE")
    ax.plot(CEN[ok],ob_m[ok]/sc,"o",color=OBSc,ms=4.5,zorder=5,label="observations")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r"density ($10^{%d}$ kg m$^{-3}$)"%kk)
    ax.set_title("%s  \u00b7  ~%d km"%(nm,round(amean/5)*5),fontsize=9,fontweight="bold",pad=8)
    ax.annotate("abcdef"[j],xy=(0,1),xycoords="axes fraction",xytext=(-42 if j==0 else -14,8),
                textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")
    # ---- bias row ----
    axb=axes[1,j]; mape_d=np.mean(np.abs(da-ob)/ob)*100; mape_n=np.mean(np.abs(nr-ob)/ob)*100
    axb.axhspan(0.9,1.1,color="0.85",alpha=0.5,lw=0); axb.axhline(1,color="0.4",ls=":",lw=1)
    axb.plot(CEN[ok],(nr_m/ob_m)[ok],"--",color=NRLc,lw=1.5,marker="^",ms=3.5,zorder=3)
    axb.plot(CEN[ok],(jb_m/ob_m)[ok],"--",color=JBc,lw=1.5,marker="s",ms=3,zorder=3)
    axb.plot(CEN[ok],(da_m/ob_m)[ok],"-",color=DAN,lw=2.0,zorder=4)
    axb.set_ylim(0.55,2.05); axb.set_xlim(-90,90); axb.set_xticks([-60,-30,0,30,60])
    axb.set_xlabel("geographic latitude (\u00b0)")
    if j==0: axb.set_ylabel("model / obs")
    axb.annotate("abcdef"[3+j],xy=(0,1),xycoords="axes fraction",xytext=(-42 if j==0 else -14,4),
                 textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")

leg=[Line2D([0],[0],color=OBSc,lw=0,marker="o",ms=5,label="observations"),
     Line2D([0],[0],color=DAN,lw=2.2,label="DANTE (5-seed ensemble, held out)"),
     Line2D([0],[0],color=JBc,lw=1.5,ls="--",marker="s",ms=4,label="JB2008"),
     Line2D([0],[0],color=NRLc,lw=1.5,ls="--",marker="^",ms=4,label="NRLMSIS"),
     Patch(facecolor="0.6",alpha=0.20,label="observed IQR (25\u201375%)")]
fig.legend(handles=leg,loc="lower center",ncol=5,fontsize=7.8,handletextpad=0.5,
           columnspacing=1.8,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=[0,0.045,1,1])
for ext in ["png","pdf","svg"]:
    fig.savefig(FIGS+"fig4."+ext,dpi=300 if ext=="png" else None,bbox_inches="tight")
print("saved fig4")
for j,(nm,a0,a1) in enumerate(BANDS):
    m=reg&(alt>=a0)&(alt<a1); ob=obs[m]; da=dan[m]; nr=nrl[m]; jbv=jb[m]
    print("  %-14s ~%3dkm n=%6d  MAPE DANTE %.2f%%  JB2008 %.2f%%  NRLMSIS %.2f%%"%(
        nm,round(alt[m].mean()),m.sum(),np.mean(np.abs(da-ob)/ob)*100,np.mean(np.abs(jbv-ob)/ob)*100,np.mean(np.abs(nr-ob)/ob)*100))
