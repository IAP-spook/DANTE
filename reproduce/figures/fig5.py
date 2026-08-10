#!/usr/bin/env python3
"""Figure 5 — Taylor diagram of the 24-hour operational forecast (DANTE vs baselines), the field-standard
model-evaluation diagram (Taylor, 2001). Statistics are computed in density-ANOMALY space
y=log10(rho)-log10(rho_exp) (what the model predicts) against observations, on the RSGA-covered test
subset (n=77,810). (a) all conditions; (b) storm (ap>=30). Each point encodes three skills at once:
correlation (azimuth), normalized standard deviation (radius), centered RMS error (distance to REF).
Shows DANTE forecasts sit closest to observations; in storms the SWPC-driver forecast clearly beats the
persistence-driver forecast and KML. Loads forecast5m.npz + fig5_anom.npz."""
import numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.unicode_minus":False,"axes.linewidth":0.8})
BN="/home/dell/hdd8t/gravity_wave/_bench/"; FIGS="/home/dell/nanoclaw/groups/main/thermo_paper/figs/"
F=dict(np.load(BN+"forecast5m_v2.npz")); F["nrl_fc"]=np.load(BN+"nrlmsis2_forecast.npz")["nrl2_fc"]  # NRLMSIS 2.0 forecast
expo=np.load(BN+"fig5_anom.npz")["expo"]
obs=F["obs"]; Ap=F["Ap"]; yo=np.log10(obs)-np.log10(expo)
MODELS=[("DANTE nowcast","med_now","#0072B2","o"),
        ("DANTE forecast (persistence drivers)","med_pe","#56B4E9","o"),
        ("DANTE forecast (SWPC drivers)","med_fc","#009E73","s"),
        ("NRLMSIS 2.0 forecast","nrl_fc","#999999","^")]
def stats(pred,m):
    r=yo[m]; f=np.log10(pred[m])-np.log10(expo[m])
    return f.std()/r.std(), np.corrcoef(f,r)[0,1]
CMIN=0.9; SMAX=1.25; YMAX=SMAX*np.sin(np.arccos(CMIN))
CTICKS=[0.90,0.95,0.99]

def taylor_axes(ax,title=True):
    amax=np.arccos(CMIN); th=np.linspace(0,amax,120)
    for s in [0.25,0.5,0.75,1.0]:                         # std arcs
        ax.plot(s*np.cos(th),s*np.sin(th),color="0.88",lw=0.7,zorder=0)
    ax.plot(np.cos(th),np.sin(th),color="0.55",lw=1.0,ls="--",zorder=1)   # ref std=1
    ax.plot(SMAX*np.cos(th),SMAX*np.sin(th),color="0.7",lw=1.0,zorder=1)  # outer boundary arc
    for c in CTICKS:                                       # correlation ticks outside the arc
        a=np.arccos(c)
        ax.plot([SMAX*np.cos(a),(SMAX+0.03)*np.cos(a)],[SMAX*np.sin(a),(SMAX+0.03)*np.sin(a)],color="0.5",lw=0.9,clip_on=False)
        ax.text((SMAX+0.09)*np.cos(a),(SMAX+0.09)*np.sin(a),"%.2f"%c,fontsize=7,color="0.3",
                ha="center",va="center",rotation=np.degrees(a),rotation_mode="anchor",clip_on=False)
    if title:
        am=np.arccos(0.955)                                # "correlation" title only on right panel
        ax.text((SMAX+0.22)*np.cos(am),(SMAX+0.22)*np.sin(am),"correlation",fontsize=8.5,color="0.3",
                ha="center",va="center",rotation=np.degrees(am)-90,rotation_mode="anchor",clip_on=False)
    for e in [0.1,0.2,0.3,0.4]:                            # centered-RMSE arcs about REF=(1,0)
        ph=np.linspace(0,np.pi,220); xx=1+e*np.cos(ph); yy=e*np.sin(ph)
        keep=(xx>=0)&(np.hypot(xx,yy)<=SMAX)&(yy<=YMAX)
        ax.plot(xx[keep],yy[keep],color="#b39ddb",lw=0.7,ls=":",zorder=0)
    ax.plot(1,0,marker="*",ms=12,color="black",zorder=5)
    ax.set_xlim(0.0,SMAX+0.10); ax.set_ylim(0,YMAX+0.12); ax.set_aspect("equal")
    ax.set_xlabel("normalized standard deviation",fontsize=8); ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)

fig,axes=plt.subplots(1,2,figsize=(9.6,3.0))
for ax,(lab,m) in zip(axes,[("all conditions",np.ones(len(obs),bool)),("storm (ap\u226530)",Ap>=30)]):
    taylor_axes(ax,title=(ax is axes[1]))
    for nm,k,col,mk in MODELS:
        p=F[k]; mm=m&np.isfinite(p)&(p>0); sd,cc=stats(p,mm)
        ax.plot(sd*cc,sd*np.sqrt(1-cc**2),marker=mk,ms=7.5,color=col,mec="white",mew=0.7,ls="none",zorder=6)
    if ax is axes[0]: ax.set_ylabel("normalized standard deviation")
    ax.annotate(lab,xy=(0.5,1.0),xycoords="axes fraction",ha="center",va="bottom",fontsize=9,color="black")
axes[0].annotate("a",xy=(0,1),xycoords="axes fraction",xytext=(-32,4),textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")
axes[1].annotate("b",xy=(0,1),xycoords="axes fraction",xytext=(-26,4),textcoords="offset points",fontsize=12,fontweight="bold",va="bottom")
leg=[Line2D([0],[0],marker="*",ls="none",color="black",label="observations (reference)")]+\
    [Line2D([0],[0],marker=mk,ls="none",color=col,label=nm) for nm,k,col,mk in MODELS]
fig.legend(handles=leg,loc="lower center",ncol=3,fontsize=7,handletextpad=0.3,columnspacing=1.1,bbox_to_anchor=(0.5,0.0))
fig.subplots_adjust(left=0.06,right=0.99,bottom=0.30,top=0.93,wspace=0.14)
for ext in ["png","pdf","svg"]:
    fig.savefig(FIGS+"fig5."+ext,dpi=300 if ext=="png" else None,bbox_inches="tight")
print("saved fig5 (Taylor)")
for lab,m in [("ALL",np.ones(len(obs),bool)),("STORM",Ap>=30)]:
    print(lab,[(nm.split(' (')[0],round(stats(F[k],m&np.isfinite(F[k])&(F[k]>0))[0],3),round(stats(F[k],m&np.isfinite(F[k])&(F[k]>0))[1],4)) for nm,k,_,_ in MODELS])
