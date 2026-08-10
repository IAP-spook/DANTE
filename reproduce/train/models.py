#!/usr/bin/env python3
"""Model zoo for the architecture bake-off. All sequence models share ONE unified scaffold
(two-stream, static-query cross-attention pooling, quantile head); ONLY the per-stream ENCODER
differs -> a clean architecture-only comparison. Plus MLP control and a TFT-lite (GRN+LSTM+attn).
GBDT is handled separately in run_model.py (tree, tabular features)."""
import torch, torch.nn as nn, torch.nn.functional as F

# ---------- per-stream encoders: [B,L,dm] -> [B,L,dm] ----------
class TransformerEnc(nn.Module):
    def __init__(s,L,dm,h=4,l=2):
        super().__init__(); s.pos=nn.Parameter(torch.randn(1,L,dm)*0.02)
        s.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(dm,h,dm*2,0.1,batch_first=True),l)
    def forward(s,x): return s.tr(x+s.pos)

class LSTMEnc(nn.Module):
    def __init__(s,L,dm,l=2):
        super().__init__(); s.rnn=nn.LSTM(dm,dm,l,batch_first=True,dropout=0.1)
    def forward(s,x): o,_=s.rnn(x); return o

class TCNEnc(nn.Module):
    def __init__(s,L,dm,dil=(1,2,4,8)):
        super().__init__(); s.dil=dil
        s.convs=nn.ModuleList([nn.Conv1d(dm,dm,3,dilation=d) for d in dil])
        s.norm=nn.ModuleList([nn.GroupNorm(1,dm) for _ in dil])
    def forward(s,x):
        h=x.transpose(1,2)
        for c,d,n in zip(s.convs,s.dil,s.norm):
            y=c(F.pad(h,(2*d,0)))            # causal (left) padding only
            h=h+F.gelu(n(y))
        return h.transpose(1,2)

class S4DEnc(nn.Module):
    """Diagonal state-space (S4D-lite): per-channel learnable-decay integrator == physical
    'integrated heating with relaxation' (Joule buildup + NO-cooling-like decay)."""
    def __init__(s,L,dm):
        super().__init__()
        s.logA=nn.Parameter(torch.rand(dm)*2-1)          # -> decay a=sigmoid(logA) in (0,1)
        s.B=nn.Parameter(torch.randn(dm)*0.2); s.C=nn.Parameter(torch.randn(dm)*0.2)
        s.D=nn.Parameter(torch.zeros(dm)); s.mix=nn.Linear(dm,dm)
    def forward(s,x):                                    # x:[B,L,dm]
        a=torch.sigmoid(s.logA); Bd,L,Cd=x.shape
        h=torch.zeros(Bd,Cd,device=x.device); ys=[]
        for t in range(L):
            h=a*h+s.B*x[:,t,:]; ys.append(s.C*h)
        y=torch.stack(ys,1)+s.D*x
        return F.gelu(s.mix(y))

ENC={"transformer":TransformerEnc,"lstm":LSTMEnc,"tcn":TCNEnc,"s4d":S4DEnc}

# ---------- unified two-stream scaffold ----------
class Stream(nn.Module):
    def __init__(s,L,dm,enc,h=4):
        super().__init__(); s.emb=nn.Linear(1,dm); s.enc=enc; s.attn=nn.MultiheadAttention(dm,h,batch_first=True)
    def forward(s,seq,q):
        e=s.enc(s.emb(seq.unsqueeze(-1))); c,_=s.attn(q,e,e); return c.squeeze(1)

class TwoStreamNet(nn.Module):
    def __init__(s,ns,nq,La,Lf,enc_name,dm=64,h=4):
        super().__init__(); E=ENC[enc_name]
        s.q=nn.Sequential(nn.Linear(ns,dm),nn.GELU())
        s.sa=Stream(La,dm,E(La,dm),h); s.sf=Stream(Lf,dm,E(Lf,dm),h)
        s.head=nn.Sequential(nn.Linear(2*dm+ns,96),nn.GELU(),nn.Dropout(0.1),nn.Linear(96,nq))
    def forward(s,sa,sf,st):
        q=s.q(st).unsqueeze(1); return s.head(torch.cat([s.sa(sa,q),s.sf(sf,q),st],1))

# ---------- MLP control (flatten) ----------
class MLPNet(nn.Module):
    def __init__(s,ns,nq,La,Lf,dm=256):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(ns+La+Lf,dm),nn.GELU(),nn.Dropout(0.1),
                            nn.Linear(dm,dm),nn.GELU(),nn.Dropout(0.1),nn.Linear(dm,nq))
    def forward(s,sa,sf,st): return s.net(torch.cat([st,sa,sf],1))

# ---------- TFT-lite (GRN + static-enriched LSTM + interpretable attention) ----------
class GRN(nn.Module):
    def __init__(s,d,dctx=None):
        super().__init__(); s.l1=nn.Linear(d,d); s.ctx=nn.Linear(dctx,d,bias=False) if dctx else None
        s.l2=nn.Linear(d,d); s.g=nn.Linear(d,2*d); s.norm=nn.LayerNorm(d)
    def forward(s,x,c=None):
        h=s.l1(x)+(s.ctx(c) if (s.ctx is not None and c is not None) else 0.0)
        h=s.l2(F.elu(h)); a,b=s.g(h).chunk(2,-1); return s.norm(x+a*torch.sigmoid(b))

class TFTLite(nn.Module):
    def __init__(s,ns,nq,La,Lf,dm=64,h=4):
        super().__init__()
        s.embA=nn.Linear(1,dm); s.embF=nn.Linear(1,dm)
        s.sctx=nn.Sequential(nn.Linear(ns,dm),nn.GELU()); s.sgrn=GRN(dm)
        s.grnA=GRN(dm,dm); s.grnF=GRN(dm,dm)
        s.lstmA=nn.LSTM(dm,dm,1,batch_first=True); s.lstmF=nn.LSTM(dm,dm,1,batch_first=True)
        s.enr=GRN(dm,dm); s.attn=nn.MultiheadAttention(dm,h,batch_first=True)
        s.post=GRN(dm); s.head=nn.Sequential(nn.Linear(dm+ns,96),nn.GELU(),nn.Linear(96,nq))
    def forward(s,sa,sf,st):
        c=s.sgrn(s.sctx(st))                                   # static context
        a=s.grnA(s.embA(sa.unsqueeze(-1)),c.unsqueeze(1)); a,_=s.lstmA(a)
        f=s.grnF(s.embF(sf.unsqueeze(-1)),c.unsqueeze(1)); f,_=s.lstmF(f)
        seq=torch.cat([a,f],1); seq=s.enr(seq,c.unsqueeze(1))  # static-enriched temporal
        q=c.unsqueeze(1); ctx,_=s.attn(q,seq,seq)              # interpretable attention pooling
        z=s.post(ctx.squeeze(1)); return s.head(torch.cat([z,st],1))

def build_nn(name,ns,nq,La,Lf):
    if name in ENC: return TwoStreamNet(ns,nq,La,Lf,name)
    if name=="mlp": return MLPNet(ns,nq,La,Lf)
    if name=="tftlite": return TFTLite(ns,nq,La,Lf)
    raise ValueError(name)
