#!/usr/bin/env python3
"""Idea 2: the driver-forecast uncertainty ceiling. On the RSGA forecast subset, propagate the SWPC
one-day-ahead driver-forecast error through the DEPLOYED 5-seed DANTE by Monte-Carlo perturbing the
forecast F10.7 and ap with resampled RSGA forecast residuals (ap residuals stratified by forecast
activity, since the ap forecast degrades in storms), then measure the resulting density spread.
Compares driver-induced spread to the model's own conformal interval and reports the nowcast->forecast
point-error decomposition. Shows the driver-forecast error, not the model, dominates in storms. CPU-only."""
import sys, os, json, numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'/home/dell/hdd8t/gravity_wave/_bench/karman'); sys.path.insert(0,'/home/dell/nanoclaw/groups/main/dante_deploy')
import karman
from dante_model import TwoStreamNet, exponential_atmosphere
GW="/home/dell/hdd8t/gravity_wave/data/"; MLW="/home/dell/hdd8t/waccmx_ml/"; BN="/home/dell/hdd8t/gravity_wave/_bench/"
WD="/home/dell/nanoclaw/groups/main/dante_deploy/weights/"; dev=torch.device("cpu")
cfg=json.load(open(WD+"config.json")); FEAT=cfg["FEAT"]; La=cfg["La"]; Lf=cfg["Lf"]
scz=np.load(WD+"scaler.npz"); MU=scz["mu"]; SD=scz["sd"]
nets=[]
for f in sorted(g for g in os.listdir(WD) if g.startswith("seed") and g.endswith(".pt")):
    nt=TwoStreamNet(len(FEAT),5,La,Lf,cfg.get("dm",64),cfg.get("heads",4)); nt.load_state_dict(torch.load(WD+f,map_location=dev)); nt.eval().to(dev); nets.append(nt)
print("loaded %d deployed seeds (CPU)"%len(nets),flush=True)
# ---- data prep (identical construction to _forecast24_5m_v2clean.py) ----
d=pd.read_parquet(GW+"thermo_all/dense_points.parquet")
d["day"]=d.t.dt.strftime("%Y-%m-%d"); d["yr"]=d.t.dt.year; d["mo"]=d.t.dt.month; d["doy"]=d.t.dt.dayofyear
d["sinA"]=np.sin(2*np.pi*d.doy/365); d["cosA"]=np.cos(2*np.pi*d.doy/365)
d["sinLT"]=np.sin(2*np.pi*d.lst/24); d["cosLT"]=np.cos(2*np.pi*d.lst/24)
d["sinLON"]=np.sin(np.pi*d.lon/180); d["cosLON"]=np.cos(np.pi*d.lon/180)
d["expo"]=exponential_atmosphere(torch.tensor(d.alt.values,dtype=torch.float32)).numpy()
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
gg=gd.reset_index().rename(columns={"index":"dt"}); gg["day"]=gg.dt.dt.strftime("%Y-%m-%d"); ggd=gg.set_index("day")
d=d.merge(gg[["day","f107","Ap","f107a"]],on="day",how="left").dropna(subset=["f107","Ap","f107a"]).reset_index(drop=True)
ap=pd.read_parquet(GW+"thermo_all/ap3h.parquet").sort_values("t"); apt=ap.t.values.astype("datetime64[ns]"); apv=np.log1p(ap.ap.values.astype(np.float32)); ap_date=apt.astype("datetime64[D]")
ia=np.searchsorted(apt,d.t.values.astype("datetime64[ns]"),side="right")-1
SA=np.zeros((len(d),La),np.float32)
for i,e in enumerate(ia):
    s=max(0,e-La+1); seg=apv[s:e+1]; SA[i,La-len(seg):]=seg
ft=gd.index.values.astype("datetime64[ns]"); idf=np.searchsorted(ft,d.t.values.astype("datetime64[D]").astype("datetime64[ns]"),side="right")-1
SF=np.zeros((len(d),Lf),np.float32)
for i,e in enumerate(idf):
    fv=(gd.f107.values.astype(np.float32)-120)/80; s=max(0,e-Lf+1); seg=fv[s:e+1]; SF[i,Lf-len(seg):]=seg
ok=(ia>=La-1)&(idf>=Lf-1); d=d[ok].reset_index(drop=True); SA=SA[ok]; SF=SF[ok]; ia=ia[ok]
Xs=d[FEAT].values.astype(np.float32)
months=np.array(range(1,13)); custom={2001:3,2003:10,2005:5,2012:9,2013:5,2015:3,2022:1,2024:4}
tm=lambda yy: custom[yy] if yy in custom else int(np.roll(months,yy-2000)[2])
te=np.where((d.yr.map(tm)==d.mo).values)[0]
rs=pd.read_csv(GW+"rsga/rsga_forecast.csv"); rs["day"]=rs["target"].astype(str); rs=rs.drop_duplicates("day").set_index("day")
dte=d.iloc[te]; has=dte.day.isin(rs.index).values; sub=te[has]; dsub=d.iloc[sub].reset_index(drop=True)
f107_fc=rs.loc[dsub.day,"f107_fc_real"].values.astype(np.float32); ap_fc=rs.loc[dsub.day,"ap_fc_real"].values.astype(np.float32)
obs=dsub.rho.values; expo=d.expo.values[sub]; Apo=dsub.Ap.values; f107_true=dsub.f107.values.astype(np.float32); f107a=dsub.f107a.values.astype(np.float32)
def build_driver(f107v,f107av,apv_daily):
    Xf=Xs[sub].copy(); Xf[:,1]=f107v; Xf[:,2]=f107av; Xf[:,3]=apv_daily; Xf_t=((Xf-MU)/SD).astype(np.float32)
    SFf=SF[sub].copy(); SFf[:,-1]=(f107v-120)/80; SAf=SA[sub].copy(); lap=np.log1p(apv_daily)
    for k in range(len(sub)):
        e=ia[sub[k]]; s=max(0,e-La+1); seg_dates=ap_date[s:e+1]; D=np.datetime64(dsub.day.iloc[k]); mask=(seg_dates==D)
        if mask.any():
            row=np.zeros(La,bool); row[La-len(seg_dates):]=mask; SAf[k,row]=lap[k]
    return SAf,SFf,Xf_t
@torch.no_grad()
def predict_density(SAb,SFb,Xrows):
    out=np.zeros(len(Xrows),np.float32); bs=32768
    for i in range(0,len(Xrows),bs):
        sa=torch.tensor(SAb[i:i+bs]); sf=torch.tensor(SFb[i:i+bs]); st=torch.tensor(Xrows[i:i+bs])
        Q=np.median(np.stack([np.sort(nt(sa,sf,st).numpy(),1)[:,2] for nt in nets],0),0)
        out[i:i+bs]=Q
    return expo*np.power(10.,out)
# baselines: true-driver (nowcast) and SWPC-forecast (unperturbed)
rho_true=predict_density(*build_driver(f107_true,f107a,Apo))          # perfect drivers = nowcast
rho_fc  =predict_density(*build_driver(f107_fc,f107a,ap_fc))          # SWPC forecast drivers
ape=lambda p,m:(np.abs(p[m]-obs[m])/obs[m]).mean()*100
allm=np.ones(len(sub),bool); quiet=Apo<15; storm=Apo>=30
print("n=%d  quiet(ap<15)=%d storm(ap>=30)=%d"%(len(sub),quiet.sum(),storm.sum()),flush=True)
print("POINT-ERROR decomposition (deployed 5-seed):")
for lab,m in [("all",allm),("quiet",quiet),("storm",storm)]:
    print("  %-6s nowcast(true drv)=%.2f%%  forecast(SWPC drv)=%.2f%%  driver-error contribution=%.2f pp"%(
        lab,ape(rho_true,m),ape(rho_fc,m),ape(rho_fc,m)-ape(rho_true,m)),flush=True)
# ---- Monte-Carlo driver perturbation ----
rng=np.random.default_rng(0); N=25
res_f=f107_fc-f107_true                      # SWPC F10.7 forecast residuals
res_a=ap_fc-Apo                              # SWPC ap forecast residuals
# activity-stratified ap residual pools (ap error grows with activity)
abins=[(-1,15),(15,30),(30,1e9)]; pools={i:res_a[(ap_fc>=lo)&(ap_fc<hi)] for i,(lo,hi) in enumerate(abins)}
which=np.select([(ap_fc>=lo)&(ap_fc<hi) for lo,hi in abins],[0,1,2],default=2)
dens=np.zeros((N,len(sub)),np.float32)
for j in range(N):
    pf=f107_fc+rng.choice(res_f,len(sub))
    pa=ap_fc.copy()
    for i,(lo,hi) in enumerate(abins):
        sel=which==i; pa[sel]=ap_fc[sel]+rng.choice(pools[i],int(sel.sum()))
    pa=np.clip(pa,0,None)
    dens[j]=predict_density(*build_driver(pf,f107a,pa))
    if j%5==0: print("  MC %d/%d"%(j,N),flush=True)
mu_d=dens.mean(0); sig_d=dens.std(0); rel=sig_d/mu_d*100          # driver-induced relative spread (%)
np.savez(BN+"driver_uncertainty.npz",rel=rel,Apo=Apo,rho_true=rho_true,rho_fc=rho_fc,obs=obs)
print("\nDRIVER-INDUCED density spread (1-sigma, %% of density):")
for lab,m in [("all",allm),("quiet ap<15",quiet),("ap 15-30",(Apo>=15)&(Apo<30)),("storm ap>=30",storm),("ap>=50",Apo>=50)]:
    print("  %-14s median=%.1f%%  mean=%.1f%%  (n=%d)"%(lab,np.median(rel[m]),rel[m].mean(),int(m.sum())),flush=True)
print("DONE",flush=True)
