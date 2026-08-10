#!/usr/bin/env python3
"""DANTE deployable inference API.

    from dante import DANTE
    m = DANTE("weights")                 # loads 5-seed ensemble + scaler + conformal calibrators
    out = m.predict_profile(
            alt_km      = np.arange(230, 531, 2.0),   # altitude grid [km]
            lat_deg     = -12.0, lon_deg = 100.0, local_solar_time_h = 14.0, day_of_year = 78,
            f107_now    = 135.0, f107a = 128.0, ap_now = 15.0,   # current daily indices
            f107_hist27 = <27 daily F10.7, oldest..newest>,      # solar-rotation history [sfu]
            ap_hist40   = <40 three-hourly ap, oldest..newest>)  # 5-day ap history
    # out: dict with alt_km, rho (median), rho_lo90, rho_hi90  [kg/m^3]

Nowcast = feed observed histories/indices; 24-h forecast = feed the SWPC-predicted F10.7/ap for the
target day. Model = headline 5-seed ensemble on despiked F10.7; uncertainty = conformalised 90% band.
"""
import os, json, numpy as np, torch
from .model import TwoStreamNet, exponential_atmosphere

class DANTE:
    def __init__(self, weights_dir, device="cpu"):
        self.dir = weights_dir; self.dev = torch.device(device)
        cfg = json.load(open(os.path.join(weights_dir, "config.json")))
        self.FEAT = cfg["FEAT"]; self.La = cfg["La"]; self.Lf = cfg["Lf"]
        self.QS = np.array(cfg["QS"], np.float32); self.dm = cfg.get("dm", 64); self.h = cfg.get("heads", 4)
        sc = np.load(os.path.join(weights_dir, "scaler.npz")); self.mu = sc["mu"]; self.sd = sc["sd"]
        self.nets = []
        for f in sorted(g for g in os.listdir(weights_dir) if g.startswith("seed") and g.endswith(".pt")):
            net = TwoStreamNet(len(self.FEAT), len(self.QS), self.La, self.Lf, self.dm, self.h)
            net.load_state_dict(torch.load(os.path.join(weights_dir, f), map_location=self.dev))
            net.eval().to(self.dev); self.nets.append(net)
        if not self.nets:
            raise FileNotFoundError("no seed*.pt weights found in " + weights_dir)
        cf = os.path.join(weights_dir, "conformal.json")
        self.d90, self.d50 = (json.load(open(cf))["d90"], json.load(open(cf))["d50"]) if os.path.exists(cf) else (0.0, 0.0)

    def _static(self, alt_km, lat, lst, doy, f107, f107a, ap_daily, lon):
        n = len(alt_km); one = np.ones(n, np.float32)
        cols = {"alt": alt_km, "f107": f107 * one, "f107a": f107a * one, "Ap": ap_daily * one,
                "sinA": np.sin(2*np.pi*doy/365)*one, "cosA": np.cos(2*np.pi*doy/365)*one, "lat": lat*one,
                "sinLT": np.sin(2*np.pi*lst/24)*one, "cosLT": np.cos(2*np.pi*lst/24)*one,
                "sinLON": np.sin(np.pi*lon/180)*one, "cosLON": np.cos(np.pi*lon/180)*one}
        X = np.stack([cols[k] for k in self.FEAT], 1).astype(np.float32)
        return (X - self.mu) / self.sd

    @torch.no_grad()
    def predict_profile(self, alt_km, lat_deg, lon_deg, local_solar_time_h, day_of_year,
                        f107_now, f107a, ap_now, f107_hist27, ap_hist40):
        alt_km = np.asarray(alt_km, np.float32); n = len(alt_km)
        Xn = self._static(alt_km, lat_deg, local_solar_time_h, day_of_year, f107_now, f107a, ap_now, lon_deg)
        # history streams: SA = log1p(ap) (40 x 3h), SF = (F10.7-120)/80 (27 daily); tiled over the alt grid
        SA = np.log1p(np.asarray(ap_hist40, np.float32))[None, :].repeat(n, 0)
        SF = ((np.asarray(f107_hist27, np.float32) - 120.0) / 80.0)[None, :].repeat(n, 0)
        assert SA.shape[1] == self.La and SF.shape[1] == self.Lf, "history length mismatch"
        sa = torch.tensor(SA, device=self.dev); sf = torch.tensor(SF, device=self.dev); st = torch.tensor(Xn, device=self.dev)
        Q = np.mean([net(sa, sf, st).cpu().numpy() for net in self.nets], 0)   # (n,5) ensemble-averaged quantiles
        Q = np.sort(Q, 1)
        expo = exponential_atmosphere(torch.tensor(alt_km)).numpy()            # kg/m^3 baseline
        rho_med = expo * 10.0 ** Q[:, 2]
        rho_lo = expo * 10.0 ** (Q[:, 0] - self.d90)                            # conformalised 90% band
        rho_hi = expo * 10.0 ** (Q[:, 4] + self.d90)
        return {"alt_km": alt_km, "rho": rho_med, "rho_lo90": rho_lo, "rho_hi90": rho_hi,
                "rho_lo50": expo*10.0**(Q[:,1]-self.d50), "rho_hi50": expo*10.0**(Q[:,3]+self.d50)}
