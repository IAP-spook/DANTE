#!/usr/bin/env python3
"""Figure S (sampling cadence) - how long a single satellite needs to build up a full global
cross-section of density. Latitude is filled every orbit and longitude within ~1 day; the pace
is set by LOCAL-TIME precession of the orbit plane. Top row: local solar time vs date for each
mission over a representative window (the drifting diagonal bands; GOCE is sun-synchronous / fixed
LT). Bottom: cumulative fraction of the global latitude x local-time plane (12 x 15deg  x  12 x 2h
= 144 cells) sampled as a function of elapsed days, averaged over multiple start epochs; the dashed
marks give the ~95% fill time. Data: TOLEOS v02 point database."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.unicode_minus":False,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8,"legend.frameon":False})
D="/home/dell/hdd8t/gravity_wave/data/thermo_all/"; FIGS="/home/dell/nanoclaw/groups/main/thermo_paper/figs/"
MC={"CHAMP":"#1b9e77","GRACE-A":"#d95f02","GRACE-FO":"#7570b3","GOCE":"#e7298a","Swarm-A":"#2171b5"}
SAT=[("CH","CHAMP"),("GA","GRACE-A"),("GC","GRACE-FO"),("GO","GOCE"),("SA","Swarm-A")]
df=pd.read_parquet(D+"dense5_points.parquet"); df["t"]=pd.to_datetime(df["t"])

NLAT,NLT=12,12; NCELL=NLAT*NLT; T=470.0; CP=np.arange(0,461,5.0)
def coverage(day,lat,lst):
    cell=(np.clip(((lat+90)/15).astype(int),0,NLAT-1)*NLT+np.clip((lst/2).astype(int),0,NLT-1))
    span=day.max(); curves=[]
    for s in np.arange(0,max(span-T,1),120.0):
        m=(day>=s)&(day<=s+T); rel=day[m]-s; ce=cell[m]
        if len(ce)<500: continue
        fo=pd.Series(rel).groupby(ce).min().values
        curves.append((fo[:,None]<=CP[None,:]).sum(0)/NCELL*100)
    return np.mean(curves,0) if curves else np.full_like(CP,np.nan)

fig=plt.figure(figsize=(11,6.2))
gs=fig.add_gridspec(2,5,height_ratios=[1.0,1.35],hspace=0.55,wspace=0.32)
fill={}
for j,(c,nm) in enumerate(SAT):
    g=df[df.sat==c].sort_values("t"); day=(g.t-g.t.min()).dt.total_seconds().values/86400.0
    # ---- top: LST vs date over a representative 470-day window ----
    s0=np.percentile(day,15); mw=(day>=s0)&(day<=s0+470)
    ax=fig.add_subplot(gs[0,j]); cmap=LinearSegmentedColormap.from_list("c",["white",MC[nm]])
    dts=mdates.date2num(g.t.values[mw])
    ax.hist2d(dts,g.lst.values[mw],bins=[45,24],range=[[dts.min(),dts.max()],[0,24]],cmap=cmap)
    ax.set_ylim(0,24); ax.set_yticks([0,6,12,18,24])
    ax.set_title(nm,fontsize=8.5,fontweight="bold",color=MC[nm],pad=3)
    if j==0: ax.set_ylabel("local solar time (h)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    for lb in ax.get_xticklabels(): lb.set_fontsize(6)
    ax.annotate("abcde"[j],xy=(0,1),xycoords="axes fraction",xytext=(-26,4),textcoords="offset points",
                fontsize=11,fontweight="bold",va="bottom")
    # ---- coverage curve ----
    cov=coverage(day,g.lat.values,g.lst.values)
    if np.nanmax(cov)>=95: fill[nm]=CP[np.argmax(cov>=95)]
    else: fill[nm]=None
    fig._cov=fig.__dict__.get("_cov",{}); fig._cov[nm]=cov

axb=fig.add_subplot(gs[1,:])
for c,nm in SAT:
    cov=fig._cov[nm]; axb.plot(CP,cov,"-",color=MC[nm],lw=2.2,label=nm)
    f=fill[nm]
    if f is not None:
        axb.plot([f,f],[0,95],":",color=MC[nm],lw=1); axb.plot(f,95,"o",color=MC[nm],ms=4)
axb.axhline(95,color="0.6",ls="--",lw=0.8); axb.text(462,95,"95%",fontsize=7,color="0.4",va="center")
axb.set_xlim(0,460); axb.set_ylim(0,101); axb.set_xlabel("elapsed time (days)")
axb.set_ylabel("global lat \u00d7 local-time\nplane sampled (%)")
axb.annotate("f",xy=(0,1),xycoords="axes fraction",xytext=(-34,4),textcoords="offset points",
             fontsize=11,fontweight="bold",va="bottom")
txt=[]
for c,nm in SAT:
    f=fill[nm]; txt.append("%s: %s"%(nm,("~%d d"%f) if f is not None else "fixed LT (sun-sync)"))
axb.legend(loc="lower right",fontsize=7.5,ncol=1)
axb.annotate("time to a full global cross-section (95%):\n"+"   ".join(txt[:3])+"\n"+"   ".join(txt[3:]),
             xy=(0.015,0.96),xycoords="axes fraction",ha="left",va="top",fontsize=7,color="0.2",
             linespacing=1.5,bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="0.8",lw=0.5,alpha=0.9))
fig.tight_layout()
for ext in ["png","pdf","svg"]:
    fig.savefig(FIGS+"figS_cadence."+ext,dpi=300 if ext=="png" else None,bbox_inches="tight")
print("saved figS_cadence"); print("fill(95%) days:",{k:(int(v) if v is not None else None) for k,v in fill.items()})
