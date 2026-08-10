#!/usr/bin/env python3
"""Compute Karman + NRLMSIS on the EXACT arch_cache TEST set (t/lat/lon reconstructed by rebuilding
dense d identically). Cache -> check5_karman_nrl.npz aligned to arch_cache te order. Verifies rho match."""
import sys; sys.path.insert(0,'/home/dell/hdd8t/gravity_wave/_bench/karman'); sys.path.insert(0,'/home/dell/nanoclaw/groups/main/arch_bench')
from f107_clean import clean_f107   # v2clean: NRLMSIS baseline uses the SAME cleaned F10.7
import numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings("ignore")
import karman; from nrlmsise00 import msise_flat
from pyatmos import download_sw_jb2008, read_sw_jb2008, jb2008
SWJB=read_sw_jb2008(download_sw_jb2008())   # SET SOLFSMY + DTCFILE (F10/S10/M10/Y10 + Dst)
GW="/home/dell/hdd8t/gravity_wave/data/"; MLW="/home/dell/hdd8t/waccmx_ml/"; BN="/home/dell/hdd8t/gravity_wave/_bench/"; dev=torch.device("cuda")
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
gd=clean_f107(gd); print("v2clean: F10.7 cleaned days=%d"%gd.attrs.get("f107_n_cleaned",-1),flush=True)
gd["f107a"]=gd.f107.rolling(81,min_periods=40).mean()
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
import time as _t
obs=d.rho.values[te]
dts=pd.to_datetime(dte.t.values).astype(str).tolist(); al=dte.alt.values.astype(float); la=dte.lat.values.astype(float); lo=dte.lon.values.astype(float)
jb=np.full(len(te),np.nan); t0=_t.time()
for i in range(len(te)):
    try: jb[i]=jb2008(dts[i],(la[i],lo[i],al[i]),SWJB).rho
    except Exception as ex:
        if i<3: print("jb err",ex,flush=True)
    if i%20000==0: print("  jb %d/%d %.1fmin"%(i,len(te),(_t.time()-t0)/60),flush=True)
np.savez(BN+"check5v2_jb2008.npz",jb=jb)
m=np.isfinite(jb)&(jb>0)
print("cached JB2008. JB2008(all valid)=%.2f%%  n=%d valid=%d"%((np.abs(jb[m]-obs[m])/obs[m]).mean()*100,len(te),int(m.sum())),flush=True)
