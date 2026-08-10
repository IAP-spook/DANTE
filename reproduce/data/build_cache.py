#!/usr/bin/env python3
"""Shared data pipeline for the architecture bake-off. Builds ONCE and caches all arrays so every
model uses IDENTICAL data/splits/features. Forecastable razor feature set. Anti-leakage: scaler fit
on TRAIN only (done in run_model, not here); histories strictly causal (<= t); test months held out.
Cache -> /home/dell/hdd8t/gravity_wave/_bench/arch_cache.npz"""
import sys; sys.path.insert(0,'/home/dell/hdd8t/gravity_wave/_bench/karman'); sys.path.insert(0,'/home/dell/nanoclaw/groups/main/arch_bench')
import numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings("ignore")
import karman
from f107_clean import clean_f107   # v2clean: despike radio-burst + fill -1 sentinel in F10.7
GW="/home/dell/hdd8t/gravity_wave/data/"; MLW="/home/dell/hdd8t/waccmx_ml/"; BN="/home/dell/hdd8t/gravity_wave/_bench/"
CACHE=BN+"arch_cache_v2clean.npz"   # v2clean: isolated cache, originals untouched
def build():
    d=pd.read_parquet(GW+"thermo_all/dense_points.parquet")
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
    gd=clean_f107(gd); print("v2clean: F10.7 cleaned days=%d"%gd.attrs.get("f107_n_cleaned",-1),flush=True)
    gd["f107a"]=gd.f107.rolling(81,min_periods=40).mean()
    gg=gd.reset_index().rename(columns={"index":"dt"}); gg["day"]=gg.dt.dt.strftime("%Y-%m-%d")
    d=d.merge(gg[["day","f107","Ap","f107a"]],on="day",how="left").dropna(subset=["f107","Ap","f107a"]).reset_index(drop=True)
    ap=pd.read_parquet(GW+"thermo_all/ap3h.parquet").sort_values("t"); apt=ap.t.values.astype("datetime64[ns]"); apv=np.log1p(ap.ap.values.astype(np.float32))
    La=40; ia=np.searchsorted(apt,d.t.values.astype("datetime64[ns]"),side="right")-1
    SA=np.zeros((len(d),La),np.float32)
    for i,e in enumerate(ia):
        s=max(0,e-La+1); seg=apv[s:e+1]; SA[i,La-len(seg):]=seg
    ft=gd.index.values.astype("datetime64[ns]"); fv=(gd.f107.values.astype(np.float32)-120)/80
    Lf=27; idf=np.searchsorted(ft,d.t.values.astype("datetime64[D]").astype("datetime64[ns]"),side="right")-1
    SF=np.zeros((len(d),Lf),np.float32)
    for i,e in enumerate(idf):
        s=max(0,e-Lf+1); seg=fv[s:e+1]; SF[i,Lf-len(seg):]=seg
    ok=(ia>=La-1)&(idf>=Lf-1); d=d[ok].reset_index(drop=True); SA=SA[ok]; SF=SF[ok]
    # storm strata drivers (evaluation only, NOT a model feature)
    om=pd.read_parquet(MLW+"drivers/omni_dst_ae_hourly.parquet").sort_values("t").reset_index(drop=True)
    ot=om.t.values.astype("datetime64[ns]"); Dser=om.Dst.values.astype(np.float32)
    tt=d.t.values.astype("datetime64[ns]"); jj=np.clip(np.searchsorted(ot,tt,side="right")-1,0,len(ot)-1)
    Dst=Dser[jj]; Dst[np.isnan(Dst)]=0.0
    FEAT=["alt","f107","f107a","Ap","sinA","cosA","lat","sinLT","cosLT","sinLON","cosLON"]
    Xs=d[FEAT].values.astype(np.float32); y=d.y.values.astype(np.float32)
    # splits: Karman test months; a distinct val month per year (HPO/early-stop); rest train
    months=np.array(range(1,13)); custom={2001:3,2003:10,2005:5,2012:9,2013:5,2015:3,2022:1,2024:4}
    tm=lambda yy: custom[yy] if yy in custom else int(np.roll(months,yy-2000)[2])
    vm=lambda yy: ((tm(yy)+5)%12)+1
    testm=d.yr.map(tm).values; valm=d.yr.map(vm).values
    istest=(testm==d.mo.values); isval=(valm==d.mo.values)&(~istest)
    te=np.where(istest)[0]; va=np.where(isval)[0]; tr=np.where(~istest&~isval)[0]
    # tree-summary features (for GBDT/MLP controls): static + history summaries
    def summ(S): return np.stack([S[:,-1],S.mean(1),S.max(1),S.min(1),S[:,-1]-S[:,-6],S.sum(1)],1)
    Xtree=np.concatenate([Xs, summ(SA), summ(SF)],1).astype(np.float32)
    np.savez(CACHE, Xs=Xs, SA=SA, SF=SF, y=y, expo=d.expo.values.astype(np.float32), rho=d.rho.values.astype(np.float32),
             Dst=Dst, Ap=d.Ap.values.astype(np.float32), tr=tr, va=va, te=te, Xtree=Xtree, FEAT=np.array(FEAT), La=La, Lf=Lf)
    print("cached n=%d  train=%d val=%d test=%d  static=%d Xtree=%d"%(len(d),len(tr),len(va),len(te),Xs.shape[1],Xtree.shape[1]))
if __name__=="__main__":
    build()
