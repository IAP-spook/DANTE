#!/usr/bin/env python3
"""Physics-anchor ABLATION (reviewer). IDENTICAL to the headline pipeline (4.85M dense5 train cache,
same features/protocol: 30 ep, AdamW lr1e-3 wd1e-4, cosine, pinball, bs16384, 5 seeds), except the
Vallado exponential ANCHOR is removed: the network predicts absolute log10(rho) directly instead of the
anomaly y = log10(rho) - log10(rho_exp). Recover rho = 10**q (no exponential multiply). Everything else
matches the headline so the comparison is clean. Isolated outputs; never overwrites the headline.
  predictions -> check5v2_noanchor_s{seed}_Q.npy  ; usage: _check5_noanchor.py <seed>"""
import sys, time, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'/home/dell/hdd8t/gravity_wave/_bench/karman'); sys.path.insert(0,'/home/dell/nanoclaw/groups/main/arch_bench')
sys.path.insert(0,'/home/dell/nanoclaw/groups/main/dante_deploy')
from models import build_nn
from dante_model import exponential_atmosphere
BN="/home/dell/hdd8t/gravity_wave/_bench/"; dev=torch.device("cuda")
QS=np.array([0.05,0.25,0.5,0.75,0.95],np.float32); seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
FEAT=["alt","f107","f107a","Ap","sinA","cosA","lat","sinLT","cosLT","sinLON","cosLON"]; La=40; Lf=27
# ---- 4.85M train cache; rebuild ABSOLUTE-density target (remove the anchor) ----
Z=np.load(BN+"arch_cache5_train_v2clean.npz"); Xtr=Z["Xs"]; SAtr=Z["SA"]; SFtr=Z["SF"]; y_anom=Z["y"]
expo_tr=exponential_atmosphere(torch.tensor(Xtr[:,0],dtype=torch.float32)).numpy()   # rho_exp(alt)
ytr=(y_anom + np.log10(expo_tr)).astype(np.float32)                                   # = log10(rho_obs)
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr_t=((Xtr-mu)/sd).astype(np.float32)
# ---- SAME 1.2M test set ----
A=np.load(BN+"arch_cache_v2clean.npz",allow_pickle=True); te=A["te"]
Xte=(A["Xs"][te]-mu)/sd; SAte=A["SA"][te]; SFte=A["SF"][te]; obs=A["rho"][te]; Dst=A["Dst"][te]; Ap=A["Ap"][te]
torch.manual_seed(seed); np.random.seed(seed)
net=build_nn("transformer",len(FEAT),5,La,Lf).to(dev)
opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30)
qt=torch.tensor(QS).to(dev)
def pinball(p,t): e=t.unsqueeze(1)-p; return torch.maximum(qt*e,(qt-1)*e).mean()
At=torch.tensor(SAtr).to(dev); Ft=torch.tensor(SFtr).to(dev); Xt=torch.tensor(Xtr_t).to(dev); yt=torch.tensor(ytr).to(dev)
n=len(ytr); bs=16384; t0=time.time()
def eval_mape():
    net.eval(); Q=np.zeros((len(te),5),np.float32)
    with torch.no_grad():
        for i in range(0,len(te),bs):
            Q[i:i+bs]=np.sort(net(torch.tensor(SAte[i:i+bs]).to(dev),torch.tensor(SFte[i:i+bs]).to(dev),torch.tensor(Xte[i:i+bs]).to(dev)).cpu().numpy(),1)
    med=np.power(10.,Q[:,2])   # NO anchor multiply
    return (np.abs(med-obs)/obs).mean()*100, Q
print("NO-ANCHOR train n=%d seed=%d (target=absolute log10 rho)"%(n,seed),flush=True)
for ep in range(30):
    net.train(); pm=torch.randperm(n)
    for i in range(0,n,bs):
        j=pm[i:i+bs]; opt.zero_grad(); pinball(net(At[j],Ft[j],Xt[j]),yt[j]).backward(); opt.step()
    sch.step()
    if seed==0:   # per-epoch convergence curve on seed 0 only
        m,_=eval_mape(); print("  ep%02d  test MAPE=%.2f%%  (%.1fmin)"%(ep,m,(time.time()-t0)/60),flush=True)
mape,Q=eval_mape(); np.save(BN+"check5v2_noanchor_s%d_Q.npy"%seed,Q)
st={"ALL":np.ones(len(te),bool),"Dst<=-50":Dst<=-50,"Dst<=-100":Dst<=-100,"ap>=100":Ap>=100}
ape=lambda m: (np.abs(np.power(10.,Q[:,2])[m]-obs[m])/obs[m]).mean()*100
print("\n=== NO-ANCHOR (seed %d) on SAME 1.2M test set (headline anchored ref: ALL 13.91) ==="%seed)
for k,m in st.items():
    if m.sum()>20: print("  %-12s MAPE=%.2f%%"%(k,ape(m)))
print("  DONE seed%d %.1fmin"%(seed,(time.time()-t0)/60),flush=True)
