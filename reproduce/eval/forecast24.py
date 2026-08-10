#!/usr/bin/env python3
"""4.85M 24h-forecast: train transformer on dense5 TRAIN (3-seed ensemble), evaluate on dense (1.2M)
test set in 3 driver modes (nowcast=true, forecast=RSGA, persist), vs Karman(true/persist) & NRLMSIS.
Mirrors _forecast24b protocol but trained on 4.85M."""
import sys; sys.path.insert(0,'/home/dell/hdd8t/gravity_wave/_bench/karman')
import numpy as np, pandas as pd, torch, torch.nn as nn, warnings; warnings.filterwarnings("ignore")
import karman; from nrlmsise00 import msise_flat
from sklearn.preprocessing import StandardScaler
GW="/home/dell/hdd8t/gravity_wave/data/"; MLW="/home/dell/hdd8t/waccmx_ml/"; BN="/home/dell/hdd8t/gravity_wave/_bench/"
dev=torch.device("cuda"); QS=np.array([0.05,0.25,0.5,0.75,0.95],np.float32); NSEED=3
d=pd.read_parquet(GW+"thermo_all/dense_points.parquet")
d["day"]=d.t.dt.strftime("%Y-%m-%d"); d["yr"]=d.t.dt.year; d["mo"]=d.t.dt.month; d["doy"]=d.t.dt.dayofyear
d["sinA"]=np.sin(2*np.pi*d.doy/365); d["cosA"]=np.cos(2*np.pi*d.doy/365)
d["sinLT"]=np.sin(2*np.pi*d.lst/24); d["cosLT"]=np.cos(2*np.pi*d.lst/24)
d["sinLON"]=np.sin(np.pi*d.lon/180); d["cosLON"]=np.cos(np.pi*d.lon/180)
d["expo"]=karman.util.exponential_atmosphere(torch.tensor(d.alt.values,dtype=torch.float32)).numpy(); d["y"]=np.log10(d.rho)-np.log10(d.expo)
gi=[]
for ln in open(MLW+"indices/Kp_ap_Ap_SN_F107_since_1932.txt"):
    if ln.startswith("#") or not ln.strip(): continue
    c=ln.split()
    if len(c)<27: continue
    try: gi.append(("%04d-%02d-%02d"%(int(c[0]),int(c[1]),int(c[2])),float(c[26]),float(c[23])))
    except: continue
gd=pd.DataFrame(gi,columns=["day","f107","Ap"]).drop_duplicates("day").set_index("day").sort_index()
gd.index=pd.to_datetime(gd.index); gd=gd.reindex(pd.date_range(gd.index.min(),gd.index.max(),freq="D")).ffill()
_bad=(gd.f107<=0)|(gd.f107>300); gd.loc[_bad,"f107"]=np.nan; gd["f107"]=gd.f107.interpolate("time",limit_direction="both")  # v2clean despike+fill
gd["f107a"]=gd.f107.rolling(81,min_periods=40).mean()
gg=gd.reset_index().rename(columns={"index":"dt"}); gg["day"]=gg.dt.dt.strftime("%Y-%m-%d"); ggd=gg.set_index("day")
d=d.merge(gg[["day","f107","Ap","f107a"]],on="day",how="left").dropna(subset=["f107","Ap","f107a"]).reset_index(drop=True)
ap=pd.read_parquet(GW+"thermo_all/ap3h.parquet").sort_values("t"); apt=ap.t.values.astype("datetime64[ns]"); apv=np.log1p(ap.ap.values.astype(np.float32)); ap_date=apt.astype("datetime64[D]")
La=40; ia=np.searchsorted(apt,d.t.values.astype("datetime64[ns]"),side="right")-1
SA=np.zeros((len(d),La),np.float32)
for i,e in enumerate(ia):
    s=max(0,e-La+1); seg=apv[s:e+1]; SA[i,La-len(seg):]=seg
ft=gd.index.values.astype("datetime64[ns]"); Lf=27; idf=np.searchsorted(ft,d.t.values.astype("datetime64[D]").astype("datetime64[ns]"),side="right")-1
SF=np.zeros((len(d),Lf),np.float32)
for i,e in enumerate(idf):
    fv=(gd.f107.values.astype(np.float32)-120)/80; s=max(0,e-Lf+1); seg=fv[s:e+1]; SF[i,Lf-len(seg):]=seg
ok=(ia>=La-1)&(idf>=Lf-1); d=d[ok].reset_index(drop=True); SA=SA[ok]; SF=SF[ok]; ia=ia[ok]
FEAT=["alt","f107","f107a","Ap","sinA","cosA","lat","sinLT","cosLT","sinLON","cosLON"]
Xs=d[FEAT].values.astype(np.float32)
months=np.array(range(1,13)); custom={2001:3,2003:10,2005:5,2012:9,2013:5,2015:3,2022:1,2024:4}
tm=lambda yy: custom[yy] if yy in custom else int(np.roll(months,yy-2000)[2])
te=np.where((d.yr.map(tm)==d.mo).values)[0]
# ---- dense5 TRAIN + scaler ----
Z=np.load(BN+"arch_cache5_train_v2clean.npz"); Xtr5=Z["Xs"]; SAtr5=Z["SA"]; SFtr5=Z["SF"]; ytr5=Z["y"]
sc=StandardScaler().fit(Xtr5); Xt=sc.transform(Xs).astype(np.float32); Xtr5_t=sc.transform(Xtr5).astype(np.float32)
class Stream(nn.Module):
    def __init__(s,Ln,dm,h,l):
        super().__init__(); s.emb=nn.Linear(1,dm); s.pos=nn.Parameter(torch.randn(1,Ln,dm)*0.02)
        s.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(dm,h,dm*2,0.1,batch_first=True),l); s.attn=nn.MultiheadAttention(dm,h,batch_first=True)
    def forward(s,seq,q): h=s.tr(s.emb(seq.unsqueeze(-1))+s.pos); c,_=s.attn(q,h,h); return c.squeeze(1)
class Net(nn.Module):
    def __init__(s,ns,nq,dm=64,h=4):
        super().__init__(); s.q=nn.Sequential(nn.Linear(ns,dm),nn.GELU()); s.sa=Stream(La,dm,h,2); s.sf=Stream(Lf,dm,h,2)
        s.head=nn.Sequential(nn.Linear(2*dm+ns,96),nn.GELU(),nn.Dropout(0.1),nn.Linear(96,nq))
    def forward(s,sa,sf,st): q=s.q(st).unsqueeze(1); return s.head(torch.cat([s.sa(sa,q),s.sf(sf,q),st],1))
qt=torch.tensor(QS).to(dev)
def pinball(p,t): e=t.unsqueeze(1)-p; return torch.maximum(qt*e,(qt-1)*e).mean()
# ---- test subset + driver constructions (seed-independent) ----
rs=pd.read_csv(GW+"rsga/rsga_forecast.csv"); rs["day"]=rs["target"].astype(str); rs=rs.drop_duplicates("day").set_index("day")
dte=d.iloc[te]; has=dte.day.isin(rs.index).values; sub=te[has]; dsub=d.iloc[sub].reset_index(drop=True)
f107_fc=rs.loc[dsub.day,"f107_fc_real"].values.astype(np.float32); ap_fc=rs.loc[dsub.day,"ap_fc_real"].values.astype(np.float32)
prevday=(pd.to_datetime(dsub.day)-pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
f107_p=ggd.reindex(prevday)["f107"].values.astype(np.float32); Ap_p=ggd.reindex(prevday)["Ap"].values.astype(np.float32); f107a_p=ggd.reindex(prevday)["f107a"].values.astype(np.float32)
def build_driver(f107v,f107av,apv_daily):
    Xf=Xs[sub].copy(); Xf[:,1]=f107v; Xf[:,2]=f107av; Xf[:,3]=apv_daily; Xf_t=sc.transform(Xf).astype(np.float32)
    SFf=SF[sub].copy(); SFf[:,-1]=(f107v-120)/80; SAf=SA[sub].copy(); lap=np.log1p(apv_daily)
    for k in range(len(sub)):
        e=ia[sub[k]]; s=max(0,e-La+1); seg_dates=ap_date[s:e+1]; D=np.datetime64(dsub.day.iloc[k]); mask=(seg_dates==D)
        if mask.any():
            row=np.zeros(La,bool); row[La-len(seg_dates):]=mask; SAf[k,row]=lap[k]
    return SAf,SFf,Xf_t
SAr,SFr,Xr=build_driver(f107_fc,dsub.f107a.values.astype(np.float32),ap_fc)
SAp2,SFp2,Xp2=build_driver(f107_p,f107a_p,Ap_p)
print("test w/ RSGA n=%d ; dense5-train n=%d"%(len(sub),len(ytr5)),flush=True)
def predict(net,SAb,SFb,Xrows):
    out=np.zeros((len(Xrows),5),np.float32)
    with torch.no_grad():
        for i in range(0,len(Xrows),16384):
            out[i:i+16384]=np.sort(net(torch.tensor(SAb[i:i+16384]).to(dev),torch.tensor(SFb[i:i+16384]).to(dev),torch.tensor(Xrows[i:i+16384]).to(dev)).cpu().numpy(),1)
    return out
Qn=Qr=Qp=None
for seed in range(NSEED):
    torch.manual_seed(seed); np.random.seed(seed)
    net=Net(len(FEAT),len(QS)).to(dev); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30)
    At=torch.tensor(SAtr5).to(dev); Ft=torch.tensor(SFtr5).to(dev); Xtt=torch.tensor(Xtr5_t).to(dev); yt=torch.tensor(ytr5).to(dev); n=len(ytr5)
    for ep in range(30):
        net.train(); pm=torch.randperm(n)
        for i in range(0,n,16384):
            j=pm[i:i+16384]; opt.zero_grad(); pinball(net(At[j],Ft[j],Xtt[j]),yt[j]).backward(); opt.step()
        sch.step()
    net.eval(); print("seed %d trained"%seed,flush=True)
    qn=predict(net,SA[sub],SF[sub],Xt[sub]); qr=predict(net,SAr,SFr,Xr); qp=predict(net,SAp2,SFp2,Xp2)
    Qn=qn if Qn is None else Qn+qn; Qr=qr if Qr is None else Qr+qr; Qp=qp if Qp is None else Qp+qp
Qn/=NSEED; Qr/=NSEED; Qp/=NSEED
med_now=d.expo.values[sub]*np.power(10.,Qn[:,2]); med_fc=d.expo.values[sub]*np.power(10.,Qr[:,2]); med_pe=d.expo.values[sub]*np.power(10.,Qp[:,2])
# Karman true + persist ; NRLMSIS fc
sw=pd.read_csv(BN+"karman/karman/satellites_data_subsampled_1d.csv"); sw["day"]=pd.to_datetime(sw["all__dates_datetime__"]).dt.strftime("%Y-%m-%d"); perdate=sw.drop_duplicates("day")
persh=perdate.copy(); persh["all__dates_datetime__"]=pd.to_datetime(persh["all__dates_datetime__"])+pd.Timedelta(days=1); persh["day"]=persh["all__dates_datetime__"].dt.strftime("%Y-%m-%d")
mo=karman.density_models.NowcastingModel(num_instantaneous_features=18,hidden_layer_dims=128,hidden_layers=3); mo.load_model(model_path=BN+"karman/models/karman_nowcast_model_log_exp_residual_valid_mape_15.14_params_35585.torch")
def krun(dd,pdf):
    out=np.full(len(dd),np.nan); hk=dd.day.isin(pdf.day).values; ix=np.where(hk)[0]
    if len(ix)==0: return out,hk
    dts=list(pd.to_datetime(dd.t.values[ix]).astype(str)); al=list((dd.alt.values[ix]*1000).astype(np.float32)); lo=list(dd.lon.values[ix].astype(np.float32)); la=list(dd.lat.values[ix].astype(np.float32))
    for j0 in range(0,len(ix),16384):
        r=range(j0,min(j0+16384,len(ix)))
        try: out[ix[j0:j0+16384]]=np.asarray(mo(dates=[dts[q] for q in r],altitudes=[al[q] for q in r],longitudes=[lo[q] for q in r],latitudes=[la[q] for q in r],device=dev,df_thermo=pdf)).ravel()
        except Exception as ex: print("kar err",ex)
    return out,hk
kar_t,hk=krun(dsub,perdate); kar_p,_=krun(dsub,persh)
nrl_fc=msise_flat(pd.to_datetime(dsub.t).dt.to_pydatetime(),dsub.alt.values,dsub.lat.values,dsub.lon.values,dsub.f107a.values,f107_fc,ap_fc)[:,5]*1e3
obs=dsub.rho.values; Apo=dsub.Ap.values; mk=hk&np.isfinite(kar_t)&np.isfinite(kar_p)
ape=lambda p,m:(np.abs(p[m]-obs[m])/obs[m]).mean()*100; allm=np.ones(len(sub),bool)
print("\n=== 4.85M 24h FORECAST (3-seed ens, n=%d, Karman-common=%d) ==="%(len(sub),mk.sum()))
for nm,p,m in [("Ours nowcast(true)",med_now,allm),("Ours fc-RSGA",med_fc,allm),("Ours fc-PERSIST",med_pe,allm),
               ("Karman nowcast(true)",kar_t,mk),("Karman fc-PERSIST",kar_p,mk),("NRLMSIS fc-RSGA",nrl_fc,allm)]:
    print("  %-22s all=%.2f%%  storm(Ap>=30)=%.2f%%"%(nm,ape(p,m),ape(p,m&(Apo>=30))))
np.savez(BN+"forecast5m_v2.npz",med_now=med_now,med_fc=med_fc,med_pe=med_pe,kar_t=kar_t,kar_p=kar_p,nrl_fc=nrl_fc,obs=obs,Ap=Apo,mk=mk)
print("saved forecast5m_v2.npz",flush=True)
