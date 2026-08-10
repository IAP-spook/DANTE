#!/usr/bin/env python3
"""End-to-end smoke test of the deployable DANTE inference API: builds a quiet and a storm 'what-if'
driver scenario and prints the predicted density profile with its calibrated 90% band."""
import os, sys, numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, _ROOT)
from dante import DANTE
m = DANTE(os.path.join(_ROOT, "weights"))
alt = np.arange(250, 511, 30.0)

def scenario(name, f107, ap):
    out = m.predict_profile(alt_km=alt, lat_deg=0.0, lon_deg=100.0, local_solar_time_h=14.0, day_of_year=80,
                            f107_now=f107, f107a=f107, ap_now=ap,
                            f107_hist27=np.full(27, f107), ap_hist40=np.full(40, ap))
    print("\n=== %s (F10.7=%d, ap=%d) ==="%(name, f107, ap))
    print(" alt[km]   rho[kg/m3]        90%% band")
    for i,a in enumerate(alt):
        print("  %3d    %.3e    [%.3e, %.3e]"%(a, out["rho"][i], out["rho_lo90"][i], out["rho_hi90"][i]))
    return out

q = scenario("QUIET", 100, 5)
s = scenario("STORM", 160, 150)
print("\nstorm/quiet density ratio by altitude:", np.round(s["rho"]/q["rho"], 2))
