#!/usr/bin/env python3
"""D4: NRLMSIS 2.0 in 24h-FORECAST mode (RSGA forecast drivers) on the EXACT same RSGA subset as
forecast5m_v2.npz, to replace the NRLMSISE-00 nrl_fc. Reconstructs dsub + forecast drivers identically
to _forecast24_5m_v2clean.py (does NOT retrain DANTE; its forecasts are already cached). Isolated
output -> nrlmsis2_forecast.npz. CPU-only."""
import sys; sys.path.insert(0,'/home/dell/hdd8t/gravity_wave/_bench/karman')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from pymsis import msis
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
_bad=(gd.f107<=0)|(gd.f107>300); gd.loc[_bad,"f107"]=np.nan; gd["f107"]=gd.f107.interpolate("time",limit_direction="both")
gd["f107a"]=gd.f107.rolling(81,min_periods=40).mean()
gg=gd.reset_index().rename(columns={"index":"dt"}); gg["day"]=gg.dt.dt.strftime("%Y-%m-%d")
d=d.merge(gg[["day","f107","Ap","f107a"]],on="day",how="left").dropna(subset=["f107","Ap","f107a"]).reset_index(drop=True)
ap=pd.read_parquet(GW+"thermo_all/ap3h.parquet").sort_values("t"); apt=ap.t.values.astype("datetime64[ns]")
La=40; ia=np.searchsorted(apt,d.t.values.astype("datetime64[ns]"),side="right")-1
ft=gd.index.values.astype("datetime64[ns]"); Lf=27; idf=np.searchsorted(ft,d.t.values.astype("datetime64[D]").astype("datetime64[ns]"),side="right")-1
ok=(ia>=La-1)&(idf>=Lf-1); d=d[ok].reset_index(drop=True)
months=np.array(range(1,13)); custom={2001:3,2003:10,2005:5,2012:9,2013:5,2015:3,2022:1,2024:4}
tm=lambda yy: custom[yy] if yy in custom else int(np.roll(months,yy-2000)[2])
te=np.where((d.yr.map(tm)==d.mo).values)[0]
rs=pd.read_csv(GW+"rsga/rsga_forecast.csv"); rs["day"]=rs["target"].astype(str); rs=rs.drop_duplicates("day").set_index("day")
dte=d.iloc[te]; has=dte.day.isin(rs.index).values; sub=te[has]; dsub=d.iloc[sub].reset_index(drop=True)
f107_fc=rs.loc[dsub.day,"f107_fc_real"].values.astype(float); ap_fc=rs.loc[dsub.day,"ap_fc_real"].values.astype(float)
dates=dsub.t.values.astype("datetime64[s]"); lat=dsub.lat.values.astype(float); lon=dsub.lon.values.astype(float); alt=dsub.alt.values.astype(float)
f107a=dsub.f107a.values.astype(float); aps=np.repeat(ap_fc[:,None],7,axis=1)
nrl2_fc=np.full(len(sub),np.nan); bs=20000
for i in range(0,len(sub),bs):
    j=slice(i,min(i+bs,len(sub)))
    nrl2_fc[i:i+bs]=np.asarray(msis.run(dates[j],lon[j],lat[j],alt[j],f107_fc[j],f107a[j],aps[j],version=2.0))[:,0]
np.savez(BN+"nrlmsis2_forecast.npz",nrl2_fc=nrl2_fc)
F=np.load(BN+"forecast5m_v2.npz"); obs=F["obs"]; Apo=F["Ap"]; nrl0_fc=F["nrl_fc"]
assert len(obs)==len(sub), (len(obs),len(sub))
ape=lambda p,m:(np.abs(p[m]-obs[m])/obs[m]).mean()*100; allm=np.ones(len(sub),bool)
print("NRLMSIS forecast (n=%d):  00 all=%.2f storm=%.2f  |  2.0 all=%.2f storm=%.2f"%(
    len(sub),ape(nrl0_fc,allm),ape(nrl0_fc,allm&(Apo>=30)),ape(nrl2_fc,allm),ape(nrl2_fc,allm&(Apo>=30))))
print("  (DANTE fc-SWPC all=17.56 storm=22.03 ; fc-persist all=15.62 storm=27.16)")
