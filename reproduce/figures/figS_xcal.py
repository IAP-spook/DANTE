#!/usr/bin/env python3
"""Figure 2 — Data coverage + inter-mission cross-calibration (integrated).
(a) Sample-count coverage of the five missions in the time x altitude plane (monthly x 10-km bins;
    grey = no observation), with each mission's median-altitude track overplotted, and a thin strip
    of the number of concurrently operating missions per month (shows the overlap windows used below).
(b-e) In those overlap windows, each mission's point density is normalised by NRLMSIS (removing the
    altitude/local-time/latitude dependence) and the two missions' daily-median ratio rho/NRLMSIS is
    compared. The residual inter-mission offset (median same-day ratio-of-ratios, day-level bootstrap
    95% CI) is a few percent; all quantitative values are in the caption. One shared mission-colour
    legend serves the whole figure."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.dates as mdates, matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from nrlmsise00 import msise_flat  # NRLMSISE-00 used purely as a climatological normalizer for cross-cal
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.unicode_minus":False,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8,"legend.frameon":False})
BN="/home/dell/hdd8t/gravity_wave/_bench/"; D="/home/dell/hdd8t/gravity_wave/data/thermo_all/"
MLW="/home/dell/hdd8t/waccmx_ml/"; FIGS="/home/dell/nanoclaw/groups/main/thermo_paper/figs/"
# unified mission colours (whole figure)
SATNAME={"CH":"CHAMP","GA":"GRACE-A","GC":"GRACE-FO","GO":"GOCE","SA":"Swarm-A"}
MCOL={"CHAMP":"#d1495b","GOCE":"#8a4fbf","GRACE-A":"#e6820e","GRACE-FO":"#2171b5","Swarm-A":"#2a2a2a"}

# ---------------- drivers (daily F10.7 / f107a / Ap) ----------------
gi=[]
for ln in open(MLW+"indices/Kp_ap_Ap_SN_F107_since_1932.txt"):
    if ln.startswith("#") or not ln.strip(): continue
    c=ln.split()
    if len(c)<27: continue
    try: gi.append(("%04d-%02d-%02d"%(int(c[0]),int(c[1]),int(c[2])),float(c[26]),float(c[23])))
    except: continue
gd=pd.DataFrame(gi,columns=["day","f107","Ap"]).drop_duplicates("day").set_index("day").sort_index()
gd.index=pd.to_datetime(gd.index); gd=gd.reindex(pd.date_range(gd.index.min(),gd.index.max(),freq="D")).ffill()
gd["f107a"]=gd.f107.rolling(81,min_periods=40).mean(); gd["day"]=gd.index.strftime("%Y-%m-%d")
drv=gd.set_index("day")[["f107","f107a","Ap"]]

# ================= figure layout =================
fig=plt.figure(figsize=(9.6,8.8))
gs=gridspec.GridSpec(4,2,height_ratios=[2.9,0.55,2.0,2.0],hspace=0.42,wspace=0.22,figure=fig)

# ---------------- (a) coverage heatmap + tracks ----------------
Z=np.load(BN+"holdout_fullcache.npz",allow_pickle=True)
yr=Z["yr"].astype(float); mo=Z["mo"].astype(float); sat=Z["sat"]; alt=Z["Xs"][:,0]
tdec=yr+(mo-1)/12.0
xe=np.arange(2000,2025.001,1/12.); ye=np.arange(225,541,10.)
H,_,_=np.histogram2d(tdec,alt,bins=[xe,ye]); xc=0.5*(xe[:-1]+xe[1:]); yc=0.5*(ye[:-1]+ye[1:])
Hm=np.ma.masked_where(H.T<1,H.T)
axA=fig.add_subplot(gs[0,:])
pc=axA.pcolormesh(xc,yc,Hm,cmap="cividis",norm=LogNorm(1,max(2,Hm.max())),shading="auto"); axA.set_facecolor("0.93")
for u in ["CH","GO","GA","GC","SA"]:
    xs=[];ys=[]
    for y in range(2000,2025):
        for m in range(1,13):
            mm=(sat==u)&(yr==y)&(mo==m)
            if mm.sum()>30: xs.append(y+(m-1)/12.); ys.append(np.median(alt[mm]))
    if xs: axA.plot(xs,ys,"-",color=MCOL[SATNAME[u]],lw=1.5,label=SATNAME[u],
                    path_effects=[pe.Stroke(linewidth=2.8,foreground="white"),pe.Normal()])
axA.plot([2000,2025],[400,400],color="0.75",ls="--",lw=0.6); axA.plot([2000,2025],[500,500],color="0.75",ls="--",lw=0.6)
axA.set_ylabel("altitude (km)"); axA.set_xlim(2000,2025); axA.set_ylim(225,540); axA.set_xticklabels([])
axA.legend(frameon=False,fontsize=7.4,ncol=5,loc="lower center",bbox_to_anchor=(0.5,1.02),handlelength=1.7,columnspacing=1.6)
cb=fig.colorbar(pc,ax=axA,pad=0.01,fraction=0.03,aspect=26); cb.set_label("samples per month × 10 km",fontsize=7.4); cb.ax.tick_params(labelsize=6.5)
axA.annotate("a",xy=(0,1),xycoords="axes fraction",xytext=(-40,6),textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")

# thin strip: number of concurrent missions per month (shows the overlap windows)
axN=fig.add_subplot(gs[1,:],sharex=axA)
nc=[]
for x in xc:
    y=int(np.floor(x)); m=int(round((x-y)*12))+1
    mm=(yr==y)&(mo==m); nc.append(len(np.unique(sat[mm])) if mm.sum()>0 else 0)
axN.fill_between(xc,nc,step="mid",color="#4f6d9c",alpha=0.65)
axN.set_ylabel("# missions",fontsize=7.5); axN.set_ylim(0,3.3); axN.set_yticks([0,1,2,3]); axN.tick_params(labelsize=7)
axN.set_xlim(2000,2025); axN.set_xticks(range(2000,2026,5)); axN.set_xticklabels([str(y) for y in range(2000,2026,5)])

# ---------------- (b-e) pairwise cross-calibration ----------------
df=pd.read_parquet(D+"dense5_points.parquet"); df["t"]=pd.to_datetime(df["t"]); df["day"]=df.t.dt.strftime("%Y-%m-%d")
RNG=np.random.default_rng(0); SUB=60000
def ratio_series(code,t0,t1):
    g=df[(df.sat==code)&(df.t>=t0)&(df.t<=t1)].copy()
    if len(g)>SUB: g=g.iloc[RNG.choice(len(g),SUB,replace=False)]
    j=g.join(drv,on="day").dropna(subset=["f107","f107a","Ap"])
    dts=[x.to_pydatetime() for x in j.t]
    ms=msise_flat(dts,j.alt.values,j.lat.values,j.lon.values,j.f107a.values,j.f107.values,j.Ap.values)[:,5]*1e3
    j=j.assign(ratio=j.rho.values/ms); j=j[(j.ratio>0.2)&(j.ratio<5)]
    daily=j.groupby("day")["ratio"].median(); daily.index=pd.to_datetime(daily.index); return daily.sort_index()
def offset_ci(rA,rB,nboot=2000):
    d=pd.concat([rA.rename("A"),rB.rename("B")],axis=1).dropna(); o=(d.A/d.B).values; days=len(o)
    med=np.median(o); r=np.corrcoef(d.A,d.B)[0,1]; bs=[np.median(RNG.choice(o,days,replace=True)) for _ in range(nboot)]
    lo,hi=np.percentile(bs,[2.5,97.5]); return (med-1)*100,(lo-1)*100,(hi-1)*100,r,days
PAIRS=[("CH","GO","2009-11-01","2010-09-30"),("CH","GA","2002-08-01","2010-09-30"),
       ("GA","SA","2014-01-01","2016-12-31"),("GC","SA","2018-05-01","2022-11-30")]
cells=[gs[2,0],gs[2,1],gs[3,0],gs[3,1]]; stroke=[pe.Stroke(linewidth=3.0,foreground="white"),pe.Normal()]
print("pair              offset%   95%CI            r     Ndays")
for i,(a,b,t0,t1) in enumerate(PAIRS):
    nA,nB=SATNAME[a],SATNAME[b]; rA=ratio_series(a,t0,t1); rB=ratio_series(b,t0,t1)
    off,lo,hi,r,nd=offset_ci(rA,rB); ax=fig.add_subplot(cells[i])
    ax.plot(rA.index,rA.values,"o",ms=1.5,color=MCOL[nA],alpha=0.22,zorder=2)
    ax.plot(rB.index,rB.values,"o",ms=1.5,color=MCOL[nB],alpha=0.22,zorder=2)
    for rr,nn in [(rA,nA),(rB,nB)]:
        s=rr.rolling(30,min_periods=8,center=True).median()
        ax.plot(s.index,s.values,"-",color=MCOL[nn],lw=2.0,zorder=6,path_effects=stroke)
    ax.axhline(1,color="0.5",ls=":",lw=1,zorder=1)
    ax.set_ylim(0.6,1.7)
    if i%2==0: ax.set_ylabel(r"$\rho\,/\,$NRLMSIS")
    if i>=2: ax.set_xlabel("year")
    ax.annotate("bcde"[i],xy=(0,1),xycoords="axes fraction",xytext=(-34,6),textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")
    # professional per-panel mission labels, colour-matched to the trend lines (no abbreviations)
    ax.text(0.035,0.955,nA,transform=ax.transAxes,color=MCOL[nA],fontsize=7.6,fontweight="bold",va="top",ha="left")
    ax.text(0.035,0.845,nB,transform=ax.transAxes,color=MCOL[nB],fontsize=7.6,fontweight="bold",va="top",ha="left")
    span=(pd.Timestamp(t1)-pd.Timestamp(t0)).days/365.25
    if span<2:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(max(1,int(round(span/5))))); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for lb in ax.get_xticklabels(): lb.set_fontsize(7)
    print("%-16s %+6.1f   [%+5.1f,%+5.1f]   %.2f   %d"%(nA+"-"+nB,off,lo,hi,r,nd))
for ext in ["png","pdf","svg"]:
    fig.savefig(FIGS+"figS_xcal."+ext,dpi=300 if ext=="png" else None,bbox_inches="tight")
print("saved figS_xcal (integrated coverage + cross-calibration)")
