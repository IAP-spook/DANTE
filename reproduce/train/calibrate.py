#!/usr/bin/env python3
"""After the deployable weights are saved, (1) verify the exported 5-seed ensemble reproduces the
headline accuracy on the held-out test set, and (2) fit + save the conformal calibrators (d90, d50)
so the deployed model returns calibrated 90%/50% intervals. Writes weights/conformal.json."""
import os, json, glob, numpy as np, torch
from dante_model import TwoStreamNet
BN="/home/dell/hdd8t/gravity_wave/_bench/"; W="/home/dell/nanoclaw/groups/main/dante_deploy/weights/"
cfg=json.load(open(W+"config.json")); FEAT=cfg["FEAT"]; La=cfg["La"]; Lf=cfg["Lf"]; QS=np.array(cfg["QS"])
sc=np.load(W+"scaler.npz"); mu=sc["mu"]; sd=sc["sd"]
nets=[]
for f in sorted(glob.glob(W+"seed*.pt")):
    net=TwoStreamNet(len(FEAT),len(QS),La,Lf); net.load_state_dict(torch.load(f,map_location="cpu")); net.eval(); nets.append(net)
print("loaded %d seed nets"%len(nets))
A=np.load(BN+"arch_cache_v2clean.npz",allow_pickle=True); te=A["te"]
Xn=((A["Xs"][te]-mu)/sd).astype(np.float32); SA=A["SA"][te]; SF=A["SF"][te]
obs=A["rho"][te]; expo=A["expo"][te]; Dst=A["Dst"][te]; Ap=A["Ap"][te]
bs=16384; Qs=[]
with torch.no_grad():
    Qacc=np.zeros((len(te),len(QS)),np.float32)
    for net in nets:
        Q=np.zeros((len(te),len(QS)),np.float32)
        for i in range(0,len(te),bs):
            Q[i:i+bs]=net(torch.tensor(SA[i:i+bs]),torch.tensor(SF[i:i+bs]),torch.tensor(Xn[i:i+bs])).numpy()
        Qacc+=Q
    Qe=np.sort(Qacc/len(nets),1)
med=expo*np.power(10.,Qe[:,2]); ape=lambda m:(np.abs(med[m]-obs[m])/obs[m]).mean()*100
st={"ALL":np.ones(len(te),bool),"Dst<=-50":Dst<=-50,"Dst<=-100":Dst<=-100,"ap>=100":Ap>=100}
print("=== exported ensemble on held-out test (should match headline 13.91 / 17.6 / 22.45 / 21.66) ===")
for k,m in st.items(): print("  %-10s MAPE=%.2f%%"%(k,ape(m)))
# conformal (CQR) on a 20% in-distribution split, same recipe as the headline evaluation
le=np.log10(expo); rr=np.log10(obs); rng=np.random.default_rng(0)
idx=rng.permutation(len(te)); nc=int(0.15*len(te)); cal=idx[:nc]
def cqr(lo,hi,y,al): E=np.maximum(lo-y,y-hi); k=min(int(np.ceil((len(E)+1)*(1-al))),len(E)); return float(np.sort(E)[k-1])
d90=cqr(le[cal]+Qe[cal,0],le[cal]+Qe[cal,4],rr[cal],0.10)
d50=cqr(le[cal]+Qe[cal,1],le[cal]+Qe[cal,3],rr[cal],0.50)
json.dump({"d90":d90,"d50":d50,"note":"CQR deltas in log10-density space; interval = expo*10**(q +/- d)"},open(W+"conformal.json","w"),indent=2)
ev=idx[nc:]
c90=100*np.mean((rr[ev]>=le[ev]+Qe[ev,0]-d90)&(rr[ev]<=le[ev]+Qe[ev,4]+d90))
print("saved conformal.json  d90=%.4f d50=%.4f  -> eval 90%% coverage=%.1f%%"%(d90,d50,c90))
