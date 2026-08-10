#!/usr/bin/env python3
"""M1 generalization experiment: leave-one-satellite-out and altitude-band hold-out.
Builds a full-feature cache once (cleaned F10.7, identical pipeline to the headline model, with
sat/alt/time kept), then for a given MODE trains ONE transformer seed on the in-domain points and
evaluates DANTE vs NRLMSIS on the held-out (unseen) satellite or altitude band. Isolated: writes
holdout_* only; leaves v1/v2 caches and models untouched.  usage: _holdout_exp.py <mode>
modes: loso_GO | loso_SA | loso_CH | altband_350_420"""
import sys, time, os, numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'/home/dell/hdd8t/gravity_wave/_bench/karman'); sys.path.insert(0,'/home/dell/nanoclaw/groups/main/arch_bench')
import karman; from models import build_nn; from f107_clean import clean_f107; from nrlmsise00 import msise_flat
GW="/home/dell/hdd8t/gravity_wave/data/"; MLW="/home/dell/hdd8t/waccmx_ml/"; BN="/home/dell/hdd8t/gravity_wave/_bench/"
dev=torch.device("cuda"); QS=np.array([0.05,0.25,0.5,0.75,0.95],np.float32)
FEAT=["alt","f107","f107a","Ap","sinA","cosA","lat","sinLT","cosLT","sinLON","cosLON"]; La=40; Lf=27
FULL=BN+"holdout_fullcache.npz"; mode=sys.argv[1] if len(sys.argv)>1 else "loso_GO"

# ---------- build full-feature cache once ----------
if not os.path.exists(FULL):
    print("building full-feature cache ...",flush=True); t=time.time()
    d=pd.read_parquet(GW+"thermo_all/dense5_points.parquet"); d["t"]=pd.to_datetime(d["t"])
    d["day"]=d.t.dt.strftime("%Y-%m-%d"); d["yr"]=d.t.dt.year; d["mo"]=d.t.dt.month; d["doy"]=d.t.dt.dayofyear
    d["sinA"]=np.sin(2*np.pi*d.doy/365); d["cosA"]=np.cos(2*np.pi*d.doy/365)
    d["sinLT"]=np.sin(2*np.pi*d.lst/24); d["cosLT"]=np.cos(2*np.pi*d.lst/24)
    d["sinLON"]=np.sin(np.pi*d.lon/180); d["cosLON"]=np.cos(np.pi*d.lon/180)
    d["expo"]=karman.util.exponential_atmosphere(torch.tensor(d.alt.values,dtype=torch.float32)).numpy()
    d["y"]=np.log10(d.rho)-np.log10(d.expo)
    gi=[]
    for ln in open(MLW+"indices/Kp_ap_Ap_SN_F107_since_1932.txt"):
        if ln.startswith("#") or not ln.strip(): continue
        c=ln.split()
        if len(c)<27: continue
        try: gi.append(("%04d-%02d-%02d"%(int(c[0]),int(c[1]),int(c[2])),float(c[26]),float(c[23])))
        except: continue
    gd=pd.DataFrame(gi,columns=["day","f107","Ap"]).drop_duplicates("day").set_index("day").sort_index()
    gd.index=pd.to_datetime(gd.index); gd=gd.reindex(pd.date_range(gd.index.min(),gd.index.max(),freq="D")).ffill()
    gd=clean_f107(gd); gd["f107a"]=gd.f107.rolling(81,min_periods=40).mean()
    gg=gd.reset_index().rename(columns={"index":"dt"}); gg["day"]=gg.dt.dt.strftime("%Y-%m-%d")
    d=d.merge(gg[["day","f107","Ap","f107a"]],on="day",how="left").dropna(subset=["f107","Ap","f107a"]).reset_index(drop=True)
    ap=pd.read_parquet(GW+"thermo_all/ap3h.parquet").sort_values("t"); apt=ap.t.values.astype("datetime64[ns]"); apv=np.log1p(ap.ap.values.astype(np.float32))
    ia=np.searchsorted(apt,d.t.values.astype("datetime64[ns]"),side="right")-1; SA=np.zeros((len(d),La),np.float32)
    for i,e in enumerate(ia):
        s=max(0,e-La+1); seg=apv[s:e+1]; SA[i,La-len(seg):]=seg
    ft=gd.index.values.astype("datetime64[ns]"); fv=(gd.f107.values.astype(np.float32)-120)/80
    idf=np.searchsorted(ft,d.t.values.astype("datetime64[D]").astype("datetime64[ns]"),side="right")-1; SF=np.zeros((len(d),Lf),np.float32)
    for i,e in enumerate(idf):
        s=max(0,e-Lf+1); seg=fv[s:e+1]; SF[i,Lf-len(seg):]=seg
    ok=(ia>=La-1)&(idf>=Lf-1); d=d[ok].reset_index(drop=True); SA=SA[ok]; SF=SF[ok]
    Xs=d[FEAT].values.astype(np.float32); y=d.y.values.astype(np.float32)
    np.savez(FULL,Xs=Xs,SA=SA,SF=SF,y=y,expo=d.expo.values.astype(np.float32),rho=d.rho.values.astype(np.float32),
             sat=d.sat.values.astype("U2"),yr=d.yr.values.astype(np.int16),mo=d.mo.values.astype(np.int8),
             t=d.t.values.astype("datetime64[ns]").astype(np.int64),lat=d.lat.values.astype(np.float32),lon=d.lon.values.astype(np.float32))
    print("full cache n=%d built %.1fmin"%(len(d),(time.time()-t)/60),flush=True)

Z=np.load(FULL,allow_pickle=True)
Xs=Z["Xs"]; SA=Z["SA"]; SF=Z["SF"]; y=Z["y"]; expo=Z["expo"]; rho=Z["rho"]; sat=Z["sat"]; alt=Xs[:,0]
# ---------- define hold-out masks ----------
if mode.startswith("loso_"):
    held=mode.split("_")[1]; test=(sat==held); train=~test; desc="leave-out satellite %s"%held
elif mode.startswith("altband_"):
    lo,hi=[float(x) for x in mode.split("_")[1:3]]; test=(alt>=lo)&(alt<hi); train=~test; desc="hold-out altitude %.0f-%.0f km"%(lo,hi)
else: raise SystemExit("bad mode")
tri=np.where(train)[0]; tei=np.where(test)[0]
print("MODE=%s (%s)  train=%d test=%d"%(mode,desc,len(tri),len(tei)),flush=True)
mu=Xs[tri].mean(0); sd=Xs[tri].std(0)+1e-6
def norm(ix): return ((Xs[ix]-mu)/sd).astype(np.float32)
seed=0; torch.manual_seed(seed); np.random.seed(seed)
net=build_nn("transformer",len(FEAT),5,La,Lf).to(dev)
opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30)
qt=torch.tensor(QS).to(dev)
def pinball(p,t): e=t.unsqueeze(1)-p; return torch.maximum(qt*e,(qt-1)*e).mean()
At=torch.tensor(SA[tri]).to(dev); Ft=torch.tensor(SF[tri]).to(dev); Xt=torch.tensor(norm(tri)).to(dev); yt=torch.tensor(y[tri]).to(dev)
n=len(tri); bs=16384; t0=time.time()
for ep in range(30):
    net.train(); pm=torch.randperm(n)
    for i in range(0,n,bs):
        j=pm[i:i+bs]; opt.zero_grad(); pinball(net(At[j],Ft[j],Xt[j]),yt[j]).backward(); opt.step()
    sch.step()
    if ep%5==0: print("  ep%d %.1fmin"%(ep,(time.time()-t0)/60),flush=True)
# ---------- evaluate on held-out ----------
net.eval(); Xte=norm(tei); SAte=SA[tei]; SFte=SF[tei]; Q=np.zeros((len(tei),5),np.float32)
with torch.no_grad():
    for i in range(0,len(tei),bs):
        Q[i:i+bs]=np.sort(net(torch.tensor(SAte[i:i+bs]).to(dev),torch.tensor(SFte[i:i+bs]).to(dev),torch.tensor(Xte[i:i+bs]).to(dev)).cpu().numpy(),1)
obs=rho[tei]; med=expo[tei]*np.power(10.,Q[:,2]); mape_d=(np.abs(med-obs)/obs).mean()*100
# NRLMSIS on the same held-out points
dts=pd.to_datetime(Z["t"][tei]).to_pydatetime(); nrl=msise_flat(list(dts),alt[tei],Z["lat"][tei],Z["lon"][tei],Xs[tei,2],Xs[tei,1],Xs[tei,3])[:,5]*1e3
mape_n=(np.abs(nrl-obs)/obs).mean()*100
le=np.log10(expo[tei]); rr=np.log10(obs); cov90=100*np.mean((rr>=le+Q[:,0])&(rr<=le+Q[:,4]))
np.savez(BN+"holdout_%s.npz"%mode,mode=mode,mape_dante=mape_d,mape_nrl=mape_n,cov90=cov90,n_test=len(tei),
         alt_lo=alt[tei].min(),alt_hi=alt[tei].max(),Q=Q,obs=obs,med=med,nrl=nrl,alt_te=alt[tei],lat_te=Z["lat"][tei])
print("\n=== HOLD-OUT %s ===  DANTE MAPE=%.2f%%  NRLMSIS MAPE=%.2f%%  improvement=%.2fx  cov90=%.1f  (n=%d, alt %.0f-%.0f)"%(
    mode,mape_d,mape_n,mape_n/mape_d,cov90,len(tei),alt[tei].min(),alt[tei].max()),flush=True)
