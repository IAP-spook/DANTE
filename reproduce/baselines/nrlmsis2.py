#!/usr/bin/env python3
"""M4 (reviewer): compute NRLMSIS 2.0 total mass density on the EXACT same 1.2M test points as the
NRLMSISE-00 baseline, to show the empirical-model version choice does not flatter DANTE. Reconstructs
the test set identically to _karman_nrl_cache_v2clean.py (same cleaned F10.7, same daily Ap), then runs
pymsis version 2.0. Isolated output -> check5v2_nrlmsis2.npz. CPU-only (does not touch the GPU retrain)."""
import sys; sys.path.insert(0,'/home/dell/hdd8t/gravity_wave/_bench/karman'); sys.path.insert(0,'/home/dell/nanoclaw/groups/main/arch_bench')
from f107_clean import clean_f107
import numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings("ignore")
import karman; from pymsis import msis
GW="/home/dell/hdd8t/gravity_wave/data/"; MLW="/home/dell/hdd8t/waccmx_ml/"; BN="/home/dell/hdd8t/gravity_wave/_bench/"
d=pd.read_parquet(GW+"thermo_all/dense_points.parquet")
d["day"]=d.t.dt.strftime("%Y-%m-%d"); d["yr"]=d.t.dt.year; d["mo"]=d.t.dt.month
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
ap=pd.read_parquet(GW+"thermo_all/ap3h.parquet").sort_values("t"); apt=ap.t.values.astype("datetime64[ns]")
La=40; ia=np.searchsorted(apt,d.t.values.astype("datetime64[ns]"),side="right")-1
ft=gd.index.values.astype("datetime64[ns]"); Lf=27; idf=np.searchsorted(ft,d.t.values.astype("datetime64[D]").astype("datetime64[ns]"),side="right")-1
ok=(ia>=La-1)&(idf>=Lf-1); d=d[ok].reset_index(drop=True)
months=np.array(range(1,13)); custom={2001:3,2003:10,2005:5,2012:9,2013:5,2015:3,2022:1,2024:4}
tm=lambda yy: custom[yy] if yy in custom else int(np.roll(months,yy-2000)[2])
istest=(d.yr.map(tm)==d.mo).values; te=np.where(istest)[0]
A=np.load(BN+"arch_cache_v2clean.npz",allow_pickle=True)
assert np.array_equal(te,A["te"]), "te mismatch"; assert np.allclose(d.rho.values[te],A["rho"][A["te"]],rtol=1e-4), "rho mismatch"
dte=d.iloc[te].reset_index(drop=True)
dates=dte.t.values.astype("datetime64[s]"); lat=dte.lat.values.astype(float); lon=dte.lon.values.astype(float)
alt=dte.alt.values.astype(float); f107=dte.f107.values.astype(float); f107a=dte.f107a.values.astype(float); Ap=dte.Ap.values.astype(float)
aps=np.repeat(Ap[:,None],7,axis=1)   # daily Ap broadcast to the 7 slots (matches the daily-Ap NRLMSISE-00 baseline)
nrl2=np.full(len(te),np.nan); bs=20000
for i in range(0,len(te),bs):
    j=slice(i,min(i+bs,len(te)))
    out=msis.run(dates[j],lon[j],lat[j],alt[j],f107[j],f107a[j],aps[j],version=2.0)
    nrl2[i:i+bs]=np.asarray(out)[:,0]   # col 0 = total mass density (kg/m3)
np.savez(BN+"check5v2_nrlmsis2.npz",nrl2=nrl2)
obs=d.rho.values[te]; nrl0=np.load(BN+"check5v2_karman_nrl.npz")["nrl"]; Dst=A["Dst"][A["te"]]; Apte=A["Ap"][A["te"]]
ape=lambda p,m:(np.abs(p[m]-obs[m])/obs[m]).mean()*100
st={"ALL":np.ones(len(te),bool),"Dst<=-50":Dst<=-50,"Dst<=-100":Dst<=-100,"ap>=100":Apte>=100}
print("=== NRLMSIS 2.0 vs NRLMSISE-00 on identical test set (n=%d) ==="%len(te))
print("  %-12s %8s %8s"%("stratum","MSISE00","MSIS2.0"))
for k,m in st.items():
    if m.sum()>20: print("  %-12s %7.2f%% %7.2f%%"%(k,ape(nrl0,m),ape(nrl2,m)))
print("  (DANTE headline ALL=13.91%%, JB2008 ALL=24.78%%)")
