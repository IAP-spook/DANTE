#!/usr/bin/env python3
"""Example 1 — a basic density profile (quiet vs storm "what-if").

Runs entirely on the bundled weights, no internet needed. Shows the core call and
that DANTE inflates density during a geomagnetic storm.

    python examples/01_basic_profile.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run without installing
import numpy as np
from dante import DANTE

model = DANTE()                       # bundled 5-seed ensemble + conformal calibrators
alt = np.arange(250, 511, 30.0)       # altitude grid [km]


def scenario(name, f107, ap):
    out = model.predict_profile(
        alt_km=alt, lat_deg=0.0, lon_deg=100.0, local_solar_time_h=14.0, day_of_year=80,
        f107_now=f107, f107a=f107, ap_now=ap,
        f107_hist27=np.full(27, f107), ap_hist40=np.full(40, ap))
    print("\n=== %s (F10.7=%d, ap=%d) ===" % (name, f107, ap))
    print(" alt[km]   rho[kg/m3]         90% band")
    for i, a in enumerate(alt):
        print("  %3d    %.3e    [%.3e, %.3e]" % (a, out["rho"][i], out["rho_lo90"][i], out["rho_hi90"][i]))
    return out


quiet = scenario("QUIET", 100, 5)
storm = scenario("STORM", 160, 150)
print("\nstorm/quiet density ratio by altitude:", np.round(storm["rho"] / quiet["rho"], 2))
