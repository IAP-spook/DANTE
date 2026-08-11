#!/usr/bin/env python3
"""Example 2 — a live nowcast from the real public feeds.

Fetches the current F10.7/ap drivers (GFZ Potsdam), runs the deployed DANTE ensemble,
prints the density profile with its calibrated band, and saves a plot. Needs pandas +
internet (and matplotlib for the plot):

    pip install "dante-thermo[drivers,plot]"
    python examples/02_live_nowcast.py [lat] [lon] [local_solar_time_h]   # defaults: 0 100 14
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run without installing
import numpy as np
from dante import DANTE, fetch_nowcast_drivers

lat = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
lon = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
lst = float(sys.argv[3]) if len(sys.argv) > 3 else 14.0

drivers = fetch_nowcast_drivers()               # observed F10.7/ap window for the latest day
model = DANTE()
alt = np.arange(230, 531, 5.0)
out = model.predict_profile(
    alt_km=alt, lat_deg=lat, lon_deg=lon, local_solar_time_h=lst,
    day_of_year=drivers["day_of_year"], f107_now=drivers["f107_now"], f107a=drivers["f107a"],
    ap_now=drivers["ap_now"], f107_hist27=drivers["f107_hist27"], ap_hist40=drivers["ap_hist40"])

print("=== DANTE live nowcast ===")
print("drivers @ %s (doy %d): F10.7=%.1f  F10.7a=%.1f  Ap=%.1f"
      % (drivers["target_day"], drivers["day_of_year"], drivers["f107_now"], drivers["f107a"], drivers["ap_now"]))
print("location: lat=%.1f lon=%.1f LST=%.1f h" % (lat, lon, lst))
for a, r, lo, hi in list(zip(alt, out["rho"], out["rho_lo90"], out["rho_hi90"]))[::6]:
    print("  %3d km  %.3e kg/m3  [%.3e, %.3e]" % (a, r, lo, hi))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    ax.fill_betweenx(alt, out["rho_lo90"], out["rho_hi90"], color="#2171b5", alpha=0.2, lw=0, label="90% interval")
    ax.plot(out["rho"], alt, "-", color="#2171b5", lw=2, label="DANTE median")
    ax.set_xscale("log")
    ax.set_xlabel("neutral density (kg m$^{-3}$)")
    ax.set_ylabel("altitude (km)")
    ax.set_title("DANTE live nowcast  %s\nlat %.0f, lon %.0f, LST %.0f h  |  F10.7=%.0f, Ap=%.0f"
                 % (drivers["target_day"], lat, lon, lst, drivers["f107_now"], drivers["ap_now"]), fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig("live_nowcast.png", dpi=150, bbox_inches="tight")
    print("saved live_nowcast.png")
except ImportError:
    print("(install matplotlib to also save a plot)")
