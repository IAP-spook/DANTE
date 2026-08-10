#!/usr/bin/env python3
"""Deployable retrain-and-export for DANTE. Reproduces the headline 5-seed ensemble by training with
the IDENTICAL code path (arch_bench.models.build_nn), the same v2-clean train cache, seeds and
protocol (30 ep, AdamW lr1e-3 wd1e-4, cosine, pinball), and SAVES each seed's state_dict plus the
shared scaler (mu, sd) and a config. Isolated: writes only under dante_deploy/weights/; leaves the
existing caches, check5v2 predictions and all other artifacts untouched.  usage: train_export.py <seed>"""
import sys, time, os, json, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'/home/dell/hdd8t/gravity_wave/_bench/karman'); sys.path.insert(0,'/home/dell/nanoclaw/groups/main/arch_bench')
from models import build_nn
BN="/home/dell/hdd8t/gravity_wave/_bench/"; OUT="/home/dell/nanoclaw/groups/main/dante_deploy/weights/"; os.makedirs(OUT,exist_ok=True)
dev=torch.device("cuda"); QS=np.array([0.05,0.25,0.5,0.75,0.95],np.float32)
FEAT=["alt","f107","f107a","Ap","sinA","cosA","lat","sinLT","cosLT","sinLON","cosLON"]; La=40; Lf=27
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0

Z=np.load(BN+"arch_cache5_train_v2clean.npz"); Xtr=Z["Xs"]; SAtr=Z["SA"]; SFtr=Z["SF"]; ytr=Z["y"]
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
if not os.path.exists(OUT+"config.json"):
    np.savez(OUT+"scaler.npz",mu=mu,sd=sd)
    json.dump({"FEAT":FEAT,"La":La,"Lf":Lf,"QS":QS.tolist(),"arch":"transformer","dm":64,"heads":4,"n_seeds":5,
               "anchor":"vallado_exponential","target":"log10(rho[kg/m3]) - log10(rho_exp)","recover":"rho = rho_exp * 10**q",
               "train_cache":"arch_cache5_train_v2clean.npz (F10.7 despiked)","protocol":"30ep AdamW lr1e-3 wd1e-4 cosine pinball bs16384"},
              open(OUT+"config.json","w"),indent=2)
Xtr_t=((Xtr-mu)/sd).astype(np.float32)
torch.manual_seed(seed); np.random.seed(seed)
net=build_nn("transformer",len(FEAT),5,La,Lf).to(dev)
opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30)
qt=torch.tensor(QS).to(dev)
def pinball(p,t): e=t.unsqueeze(1)-p; return torch.maximum(qt*e,(qt-1)*e).mean()
At=torch.tensor(SAtr).to(dev); Ft=torch.tensor(SFtr).to(dev); Xt=torch.tensor(Xtr_t).to(dev); yt=torch.tensor(ytr).to(dev)
n=len(ytr); bs=16384; t0=time.time(); print("train n=%d seed=%d"%(n,seed),flush=True)
for ep in range(30):
    net.train(); pm=torch.randperm(n)
    for i in range(0,n,bs):
        j=pm[i:i+bs]; opt.zero_grad(); pinball(net(At[j],Ft[j],Xt[j]),yt[j]).backward(); opt.step()
    sch.step()
    if ep%5==0: print("  seed%d ep%d %.1fmin"%(seed,ep,(time.time()-t0)/60),flush=True)
torch.save(net.state_dict(), OUT+"seed%d.pt"%seed)
print("SAVED %sseed%d.pt (%.1fmin)"%(OUT,seed,(time.time()-t0)/60),flush=True)
