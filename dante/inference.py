#!/usr/bin/env python3
"""DANTE inference API — turn two space-weather drivers into a calibrated density profile.

This is the one class most users need. It loads the trained 5-seed ensemble, the input
scaler, and the conformal calibrators that ship inside the package, and exposes a single
method, :meth:`DANTE.predict_profile`, that returns a thermospheric neutral-density profile
with a calibrated uncertainty band.

Minimal example
---------------
    import numpy as np
    from dante import DANTE

    model = DANTE()                                   # uses the bundled trained weights
    out = model.predict_profile(
        alt_km             = np.arange(230, 531, 5.0),# altitude grid [km]
        lat_deg            = -12.0,                   # geographic latitude  [deg]
        lon_deg            = 100.0,                   # geographic longitude [deg]
        local_solar_time_h = 14.0,                    # local solar time     [h, 0-24]
        day_of_year        = 78,                      # 1-366
        f107_now           = 135.0,                   # current daily F10.7  [sfu]
        f107a              = 128.0,                   # 81-day F10.7 average  [sfu]
        ap_now             = 15.0,                    # current daily Ap
        f107_hist27        = np.full(27, 135.0),      # 27 daily F10.7 (oldest..newest) [sfu]
        ap_hist40          = np.full(40, 15.0),       # 40 three-hourly ap (oldest..newest, 5 days)
    )
    out["rho"]                          # median density profile           [kg/m^3]
    out["rho_lo90"], out["rho_hi90"]    # calibrated 90% uncertainty band  [kg/m^3]

Nowcast vs 24-hour forecast
---------------------------
The model is identical in both modes; only the drivers differ.
  * Nowcast   : feed the *observed* driver histories and current indices.
  * 24-h fcst : feed the SWPC *predicted* F10.7 and ap for the target day
                (see :func:`dante.drivers.fetch_forecast_indices`).

Author: Yinan Wang (wangyinan@mail.iap.ac.cn), Institute of Atmospheric Physics, CAS.
"""
import os
import json
import numpy as np
import torch

from .model import TwoStreamNet, exponential_atmosphere

# Trained weights bundled inside the package, so ``DANTE()`` works with no arguments.
_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


class DANTE:
    """The deployed DANTE model: a 5-seed deep ensemble with conformal calibration.

    Parameters
    ----------
    weights_dir : str, optional
        Directory holding ``seed*.pt``, ``scaler.npz``, ``config.json`` and
        ``conformal.json``. Defaults to the weights bundled with the package.
    device : str, optional
        Torch device string (``"cpu"`` or e.g. ``"cuda"``). The model is tiny
        (~186k parameters) and runs in well under 10 ms on CPU, so ``"cpu"`` is
        a sensible default.
    """

    def __init__(self, weights_dir=None, device="cpu"):
        self.dir = weights_dir or _DEFAULT_WEIGHTS
        self.dev = torch.device(device)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError("weights directory not found: %s" % self.dir)

        # --- model / feature configuration ---
        cfg = json.load(open(os.path.join(self.dir, "config.json")))
        self.FEAT = cfg["FEAT"]                       # ordered static feature names
        self.La = cfg["La"]                           # ap history length   (40 three-hourly steps)
        self.Lf = cfg["Lf"]                           # F10.7 history length (27 daily steps)
        self.QS = np.array(cfg["QS"], np.float32)     # the 5 predicted quantile levels
        self.dm = cfg.get("dm", 64)
        self.h = cfg.get("heads", 4)

        # --- input standardiser (fit on the training set only) ---
        sc = np.load(os.path.join(self.dir, "scaler.npz"))
        self.mu, self.sd = sc["mu"], sc["sd"]

        # --- load the 5 ensemble members ---
        self.nets = []
        for f in sorted(g for g in os.listdir(self.dir) if g.startswith("seed") and g.endswith(".pt")):
            net = TwoStreamNet(len(self.FEAT), len(self.QS), self.La, self.Lf, self.dm, self.h)
            net.load_state_dict(torch.load(os.path.join(self.dir, f), map_location=self.dev))
            net.eval().to(self.dev)
            self.nets.append(net)
        if not self.nets:
            raise FileNotFoundError("no seed*.pt weights found in " + self.dir)

        # --- conformal calibration deltas (log10-density space) ---
        cf = os.path.join(self.dir, "conformal.json")
        if os.path.exists(cf):
            cj = json.load(open(cf))
            self.d90, self.d50 = cj["d90"], cj["d50"]
        else:
            self.d90, self.d50 = 0.0, 0.0

    # ------------------------------------------------------------------ #
    def _static_features(self, alt_km, lat, lst, doy, f107, f107a, ap_daily, lon):
        """Assemble and standardise the static feature matrix (n_alt x n_features)."""
        n = len(alt_km)
        one = np.ones(n, np.float32)
        cols = {
            "alt": alt_km,
            "f107": f107 * one,
            "f107a": f107a * one,
            "Ap": ap_daily * one,
            "sinA": np.sin(2 * np.pi * doy / 365) * one,
            "cosA": np.cos(2 * np.pi * doy / 365) * one,
            "lat": lat * one,
            "sinLT": np.sin(2 * np.pi * lst / 24) * one,
            "cosLT": np.cos(2 * np.pi * lst / 24) * one,
            "sinLON": np.sin(np.pi * lon / 180) * one,
            "cosLON": np.cos(np.pi * lon / 180) * one,
        }
        X = np.stack([cols[k] for k in self.FEAT], 1).astype(np.float32)
        return (X - self.mu) / self.sd

    @torch.no_grad()
    def predict_profile(self, alt_km, lat_deg, lon_deg, local_solar_time_h, day_of_year,
                        f107_now, f107a, ap_now, f107_hist27, ap_hist40):
        """Predict the neutral-density profile with calibrated uncertainty.

        All location/time/index arguments are scalars describing one prediction
        column; the profile is returned on the supplied ``alt_km`` grid.

        Returns
        -------
        dict of numpy arrays (all densities in kg/m^3):
            ``alt_km``                 : the altitude grid (echoed back)
            ``rho``                    : median density profile
            ``rho_lo90``, ``rho_hi90`` : calibrated 90% interval
            ``rho_lo50``, ``rho_hi50`` : calibrated 50% interval
        """
        alt_km = np.asarray(alt_km, np.float32)
        f107_hist27 = np.asarray(f107_hist27, np.float32)
        ap_hist40 = np.asarray(ap_hist40, np.float32)
        # Validate the two history streams up front, with actionable messages.
        if f107_hist27.shape[-1] != self.Lf:
            raise ValueError("f107_hist27 must have %d daily values (oldest..newest), got %d"
                             % (self.Lf, f107_hist27.shape[-1]))
        if ap_hist40.shape[-1] != self.La:
            raise ValueError("ap_hist40 must have %d three-hourly values (oldest..newest), got %d"
                             % (self.La, ap_hist40.shape[-1]))

        n = len(alt_km)
        Xn = self._static_features(alt_km, lat_deg, local_solar_time_h, day_of_year,
                                   f107_now, f107a, ap_now, lon_deg)
        # History streams, tiled across the altitude grid, in the exact encoding used in training:
        #   ap stream    = log1p(ap)            (40 x 3-hourly)
        #   F10.7 stream = (F10.7 - 120) / 80   (27 daily)
        SA = np.log1p(ap_hist40)[None, :].repeat(n, 0)
        SF = ((f107_hist27 - 120.0) / 80.0)[None, :].repeat(n, 0)

        sa = torch.tensor(SA, device=self.dev)
        sf = torch.tensor(SF, device=self.dev)
        st = torch.tensor(Xn, device=self.dev)

        # Ensemble-average the 5 seeds' quantiles, then sort to enforce monotone quantiles.
        Q = np.mean([net(sa, sf, st).cpu().numpy() for net in self.nets], 0)   # (n, 5)
        Q = np.sort(Q, 1)

        # Recover physical density from the anchored log-anomaly: rho = rho_exp * 10**q,
        # then widen the raw quantiles by the conformal deltas to get calibrated intervals.
        expo = exponential_atmosphere(torch.tensor(alt_km)).numpy()            # kg/m^3 baseline
        rho_med = expo * 10.0 ** Q[:, 2]
        rho_lo90 = expo * 10.0 ** (Q[:, 0] - self.d90)
        rho_hi90 = expo * 10.0 ** (Q[:, 4] + self.d90)
        rho_lo50 = expo * 10.0 ** (Q[:, 1] - self.d50)
        rho_hi50 = expo * 10.0 ** (Q[:, 3] + self.d50)
        return {"alt_km": alt_km, "rho": rho_med,
                "rho_lo90": rho_lo90, "rho_hi90": rho_hi90,
                "rho_lo50": rho_lo50, "rho_hi50": rho_hi50}
