#!/usr/bin/env python3
"""Self-contained DANTE architecture + Vallado exponential-atmosphere anchor for deployment.
The TwoStreamNet / Stream / TransformerEnc classes are byte-for-byte the headline architecture
(arch_bench/models.py, transformer backbone), so the saved state_dicts load without modification.
No dependency on the karman package or the training pipeline."""
import torch, torch.nn as nn, torch.nn.functional as F

# ---------- headline architecture (transformer two-stream, static-query cross-attention) ----------
class TransformerEnc(nn.Module):
    def __init__(s, L, dm, h=4, l=2):
        super().__init__(); s.pos = nn.Parameter(torch.randn(1, L, dm) * 0.02)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, h, dm * 2, 0.1, batch_first=True), l)
    def forward(s, x): return s.tr(x + s.pos)

class Stream(nn.Module):
    def __init__(s, L, dm, enc, h=4):
        super().__init__(); s.emb = nn.Linear(1, dm); s.enc = enc
        s.attn = nn.MultiheadAttention(dm, h, batch_first=True)
    def forward(s, seq, q):
        e = s.enc(s.emb(seq.unsqueeze(-1))); c, _ = s.attn(q, e, e); return c.squeeze(1)

class TwoStreamNet(nn.Module):
    def __init__(s, ns, nq, La, Lf, dm=64, h=4):
        super().__init__()
        s.q = nn.Sequential(nn.Linear(ns, dm), nn.GELU())
        s.sa = Stream(La, dm, TransformerEnc(La, dm), h)
        s.sf = Stream(Lf, dm, TransformerEnc(Lf, dm), h)
        s.head = nn.Sequential(nn.Linear(2 * dm + ns, 96), nn.GELU(), nn.Dropout(0.1), nn.Linear(96, nq))
    def forward(s, sa, sf, st):
        q = s.q(st).unsqueeze(1); return s.head(torch.cat([s.sa(sa, q), s.sf(sf, q), st], 1))

# ---------- Vallado (2013) piecewise-exponential reference atmosphere, kg/m^3 ----------
_ZB = torch.tensor([0., 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 180, 200, 250,
                    300, 350, 400, 450, 500, 600, 700, 800, 900, 1000])
_RHOB = torch.tensor([1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 8.770e-5, 1.905e-5,
                      3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, 2.070e-9, 5.464e-10,
                      2.789e-10, 7.248e-11, 2.418e-11, 9.518e-12, 3.725e-12, 1.585e-12, 6.967e-13,
                      1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15])
_ZS = torch.tensor([7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 5.877, 7.263, 9.473,
                    12.636, 16.149, 22.523, 29.740, 37.105, 45.546, 53.628, 53.298, 58.515, 60.828,
                    63.822, 71.835, 88.667, 124.64, 181.05, 268.00])

def exponential_atmosphere(alt_km):
    """alt_km: torch.Tensor of geodetic altitudes [km] -> density [kg/m^3]."""
    zb_e = _ZB.clone(); zb_e[0], zb_e[-1] = -float("inf"), float("inf")
    i = torch.searchsorted(_ZB, alt_km, right=True) - 1
    i = i.clamp(0, len(_ZB) - 1)
    return _RHOB[i] * torch.exp(-(alt_km - zb_e[i]) / _ZS[i])
