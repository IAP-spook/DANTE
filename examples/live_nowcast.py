#!/usr/bin/env python3
"""Operational live nowcast: fetch the current SWPC/GFZ drivers, run the deployed DANTE ensemble,
and output the density profile with calibrated uncertainty for a chosen location, plus a plot.

    python live_nowcast.py [lat] [lon] [local_solar_time_h]     # defaults: 0 100 14
"""
import sys, numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False})
from dante import DANTE
from fetch_drivers import fetch_nowcast_drivers

lat = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
lon = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
lst = float(sys.argv[3]) if len(sys.argv) > 3 else 14.0

b = fetch_nowcast_drivers()
m = DANTE("weights")
alt = np.arange(230, 531, 5.0)
out = m.predict_profile(alt_km=alt, lat_deg=lat, lon_deg=lon, local_solar_time_h=lst,
                        day_of_year=b["day_of_year"], f107_now=b["f107_now"], f107a=b["f107a"],
                        ap_now=b["ap_now"], f107_hist27=b["f107_hist27"], ap_hist40=b["ap_hist40"])
print("=== DANTE live nowcast ===")
print("drivers @ %s (doy %d): F10.7=%.1f  F10.7a=%.1f  Ap=%.1f" % (
    b["target_day"], b["day_of_year"], b["f107_now"], b["f107a"], b["ap_now"]))
print("location: lat=%.1f lon=%.1f LST=%.1f h" % (lat, lon, lst))
for a, r, lo, hi in list(zip(alt, out["rho"], out["rho_lo90"], out["rho_hi90"]))[::6]:
    print("  %3d km  %.3e kg/m3  [%.3e, %.3e]" % (a, r, lo, hi))

fig, ax = plt.subplots(figsize=(4.2, 4.6))
ax.fill_betweenx(alt, out["rho_lo90"], out["rho_hi90"], color="#2171b5", alpha=0.2, lw=0, label="90% interval")
ax.plot(out["rho"], alt, "-", color="#2171b5", lw=2, label="DANTE median")
ax.set_xscale("log"); ax.set_xlabel("neutral density (kg m$^{-3}$)"); ax.set_ylabel("altitude (km)")
ax.set_title("DANTE live nowcast  %s\nlat %.0f, lon %.0f, LST %.0f h  |  F10.7=%.0f, Ap=%.0f" % (
    b["target_day"], lat, lon, lst, b["f107_now"], b["ap_now"]), fontsize=8)
ax.legend(fontsize=7, frameon=False); fig.tight_layout()
fig.savefig("figs/live_nowcast.png", dpi=150, bbox_inches="tight")
print("saved figs/live_nowcast.png")
