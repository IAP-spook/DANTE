#!/usr/bin/env python3
"""Clean check: does 4.85M (5-min) training data improve accuracy vs 1.2M? Train transformer on
dense5 TRAIN months (current protocol: periodic feats + quantile head + 30ep), evaluate on the SAME
1.2M dense TEST set (from arch_cache.npz), compare to the 1.2M-trained transformer (14.86%). seed=arg."""
import sys, time, numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'/home/dell/hdd8t/gravity_wave/_bench/karman'); sys.path.insert(0,'/home/dell/nanoclaw/groups/main/arch_bench')
import karman; from models import build_nn
from f107_clean import clean_f107   # v2clean: despike radio-burst + fill -1 sentinel in F10.7
GW="/home/dell/hdd8t/gravity_wave/data/"; MLW="/home/dell/hdd8t/waccmx_ml/"; BN="/home/dell/hdd8t/gravity_wave/_bench/"
dev=torch.device("cuda"); QS=np.array([0.05,0.25,0.5,0.75,0.95],np.float32); seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
FEAT=["alt","f107","f107a","Ap","sinA","cosA","lat","sinLT","cosLT","sinLON","cosLON"]; La=40; Lf=27
C5=BN+"arch_cache5_train_v2clean.npz"   # v2clean isolated train cache
def gg_build():
    gi=[]
    for ln in open(MLW+"indices/Kp_ap_Ap_SN_F107_since_1932.txt"):
        if ln.startswith("#") or not ln.strip(): continue
        c=ln.split()
        if len(c)<27: continue
        try: gi.append(("%04d-%02d-%02d"%(int(c[0]),int(c[1]),int(c[2])),float(c[26]),float(c[23])))
        except: continue
    gd=pd.DataFrame(gi,columns=["day","f107","Ap"]).drop_duplicates("day").set_index("day").sort_index()
    gd.index=pd.to_datetime(gd.index); gd=gd.reindex(pd.date_range(gd.index.min(),gd.index.max(),freq="D")).ffill()
    gd=clean_f107(gd); print("v2clean: F10.7 cleaned days=%d"%gd.attrs.get("f107_n_cleaned",-1),flush=True)
    gd["f107a"]=gd.f107.rolling(81,min_periods=40).mean()
    gg=gd.reset_index().rename(columns={"index":"dt"}); gg["day"]=gg.dt.dt.strftime("%Y-%m-%d"); return gd,gg
import os
if not os.path.exists(C5):
    print("building dense5 TRAIN cache ...",flush=True); t=time.time()
    d=pd.read_parquet(GW+"thermo_all/dense5_points.parquet")
    d["day"]=d.t.dt.strftime("%Y-%m-%d"); d["yr"]=d.t.dt.year; d["mo"]=d.t.dt.month; d["doy"]=d.t.dt.dayofyear
    d["sinA"]=np.sin(2*np.pi*d.doy/365); d["cosA"]=np.cos(2*np.pi*d.doy/365)
    d["sinLT"]=np.sin(2*np.pi*d.lst/24); d["cosLT"]=np.cos(2*np.pi*d.lst/24)
    d["sinLON"]=np.sin(np.pi*d.lon/180); d["cosLON"]=np.cos(np.pi*d.lon/180)
    d["expo"]=karman.util.exponential_atmosphere(torch.tensor(d.alt.values,dtype=torch.float32)).numpy(); d["y"]=np.log10(d.rho)-np.log10(d.expo)
    gd,gg=gg_build(); d=d.merge(gg[["day","f107","Ap","f107a"]],on="day",how="left").dropna(subset=["f107","Ap","f107a"]).reset_index(drop=True)
    ap=pd.read_parquet(GW+"thermo_all/ap3h.parquet").sort_values("t"); apt=ap.t.values.astype("datetime64[ns]"); apv=np.log1p(ap.ap.values.astype(np.float32))
    ia=np.searchsorted(apt,d.t.values.astype("datetime64[ns]"),side="right")-1; SA=np.zeros((len(d),La),np.float32)
    for i,e in enumerate(ia):
        s=max(0,e-La+1); seg=apv[s:e+1]; SA[i,La-len(seg):]=seg
    ft=gd.index.values.astype("datetime64[ns]"); fv=(gd.f107.values.astype(np.float32)-120)/80
    idf=np.searchsorted(ft,d.t.values.astype("datetime64[D]").astype("datetime64[ns]"),side="right")-1; SF=np.zeros((len(d),Lf),np.float32)
    for i,e in enumerate(idf):
        s=max(0,e-Lf+1); seg=fv[s:e+1]; SF[i,Lf-len(seg):]=seg
    ok=(ia>=La-1)&(idf>=Lf-1); d=d[ok].reset_index(drop=True); SA=SA[ok]; SF=SF[ok]
    months=np.array(range(1,13)); custom={2001:3,2003:10,2005:5,2012:9,2013:5,2015:3,2022:1,2024:4}
    tm=lambda yy: custom[yy] if yy in custom else int(np.roll(months,yy-2000)[2])
    istest=(d.yr.map(tm)==d.mo).values; tr=np.where(~istest)[0]
    Xs=d[FEAT].values.astype(np.float32); y=d.y.values.astype(np.float32)
    np.savez(C5,Xs=Xs[tr],SA=SA[tr],SF=SF[tr],y=y[tr]); print("dense5 train n=%d built in %.1f min"%(len(tr),(time.time()-t)/60),flush=True)
Z=np.load(C5); Xtr=Z["Xs"]; SAtr=Z["SA"]; SFtr=Z["SF"]; ytr=Z["y"]
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr_t=(Xtr-mu)/sd
# dense (1.2M) TEST set from arch_cache
A=np.load(BN+"arch_cache_v2clean.npz",allow_pickle=True); te=A["te"]
Xte=(A["Xs"][te]-mu)/sd; SAte=A["SA"][te]; SFte=A["SF"][te]; obs=A["rho"][te]; expo=A["expo"][te]; Dst=A["Dst"][te]; Ap=A["Ap"][te]
le=np.log10(expo); rr=np.log10(obs)
torch.manual_seed(seed); np.random.seed(seed)
net=build_nn("transformer",len(FEAT),5,La,Lf).to(dev); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30)
qt=torch.tensor(QS).to(dev)
def pinball(p,t): e=t.unsqueeze(1)-p; return torch.maximum(qt*e,(qt-1)*e).mean()
At=torch.tensor(SAtr).to(dev); Ft=torch.tensor(SFtr).to(dev); Xt=torch.tensor(Xtr_t).to(dev); yt=torch.tensor(ytr).to(dev); n=len(ytr); bs=16384
print("train n=%d  seed=%d"%(n,seed),flush=True); t0=time.time()
for ep in range(30):
    net.train(); pm=torch.randperm(n)
    for i in range(0,n,bs):
        j=pm[i:i+bs]; opt.zero_grad(); pinball(net(At[j],Ft[j],Xt[j]),yt[j]).backward(); opt.step()
    sch.step()
    if ep%5==0: print("  ep%d %.1fmin"%(ep,(time.time()-t0)/60),flush=True)
net.eval(); Q=np.zeros((len(te),5),np.float32)
with torch.no_grad():
    for i in range(0,len(te),bs):
        Q[i:i+bs]=np.sort(net(torch.tensor(SAte[i:i+bs]).to(dev),torch.tensor(SFte[i:i+bs]).to(dev),torch.tensor(Xte[i:i+bs]).to(dev)).cpu().numpy(),1)
med=expo*np.power(10.,Q[:,2]); ape=lambda m:(np.abs(med[m]-obs[m])/obs[m]).mean()*100
st={"ALL":np.ones(len(te),bool),"Dst<=-50":Dst<=-50,"Dst<=-100":Dst<=-100,"ap>=100":Ap>=100}
np.save(BN+"check5v2_transformer_s%d_Q.npy"%seed,Q)
print("\n=== 4.85M-trained transformer (seed %d) on SAME 1.2M test set ==="%seed)
for k,m in st.items():
    if m.sum()>20: print("  %-12s MAPE=%.2f%%"%(k,ape(m)))
print("  cal90=%.1f cal50=%.1f  (1.2M-transformer ref: ALL 14.86, Dst<=-100 25.82, cal90 86.8)"%(100*np.mean((rr>=le+Q[:,0])&(rr<=le+Q[:,4])),100*np.mean((rr>=le+Q[:,1])&(rr<=le+Q[:,3]))),flush=True)
