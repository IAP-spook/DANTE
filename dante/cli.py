#!/usr/bin/env python3
"""Command-line interface for DANTE — predict a density profile without writing any Python.

Examples
--------
Steady-state "what-if" nowcast at given indices (histories are filled with the current values):
    dante-predict --f107 135 --ap 15 --lat -12 --lon 100 --lst 14 --doy 78

Live nowcast from the public GFZ/SWPC feeds (needs pandas + internet):
    dante-predict --live --lat 0 --lon 100 --lst 14

Write the full profile to CSV and a PNG plot:
    dante-predict --f107 160 --ap 120 --alt-min 230 --alt-max 530 --alt-step 5 \
                  --out profile.csv --plot profile.png

Run ``dante-predict --help`` for all options.
"""
import argparse
import sys
import numpy as np

from .inference import DANTE


def _parse_list(s, n, name):
    """Parse a comma-separated list of exactly n floats."""
    vals = np.array([float(x) for x in s.split(",")], np.float32)
    if len(vals) != n:
        raise SystemExit("error: --%s must have %d comma-separated values, got %d" % (name, n, len(vals)))
    return vals


def build_parser():
    p = argparse.ArgumentParser(
        prog="dante-predict",
        description="Predict a thermospheric neutral-density profile with calibrated uncertainty.")
    # --- drivers ---
    g = p.add_argument_group("drivers (choose --live OR give indices)")
    g.add_argument("--live", action="store_true",
                   help="fetch current F10.7/ap from the public GFZ/SWPC feeds (needs pandas + internet)")
    g.add_argument("--f107", type=float, help="current daily F10.7 [sfu]")
    g.add_argument("--f107a", type=float, help="81-day F10.7 average [sfu] (default: = --f107)")
    g.add_argument("--ap", type=float, help="current daily Ap")
    g.add_argument("--f107-hist27", type=str,
                   help="27 daily F10.7 (oldest..newest), comma-separated; default: constant = --f107")
    g.add_argument("--ap-hist40", type=str,
                   help="40 three-hourly ap (oldest..newest), comma-separated; default: constant = --ap")
    # --- location / time ---
    g2 = p.add_argument_group("location and time")
    g2.add_argument("--lat", type=float, default=0.0, help="geographic latitude [deg] (default 0)")
    g2.add_argument("--lon", type=float, default=100.0, help="geographic longitude [deg] (default 100)")
    g2.add_argument("--lst", type=float, default=14.0, help="local solar time [h] (default 14)")
    g2.add_argument("--doy", type=int, default=80, help="day of year 1-366 (default 80)")
    # --- altitude grid ---
    g3 = p.add_argument_group("altitude grid [km]")
    g3.add_argument("--alt-min", type=float, default=230.0)
    g3.add_argument("--alt-max", type=float, default=530.0)
    g3.add_argument("--alt-step", type=float, default=20.0)
    # --- output ---
    g4 = p.add_argument_group("output")
    g4.add_argument("--out", type=str, help="write the full profile to this CSV file")
    g4.add_argument("--plot", type=str, help="save a density-profile plot to this PNG file")
    g4.add_argument("--weights", type=str, default=None, help="custom weights directory (default: bundled)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    model = DANTE(args.weights)
    alt = np.arange(args.alt_min, args.alt_max + 1e-6, args.alt_step, dtype=np.float32)

    # --- resolve the drivers ---
    if args.live:
        from .drivers import fetch_nowcast_drivers
        b = fetch_nowcast_drivers()
        f107_now, f107a, ap_now = b["f107_now"], b["f107a"], b["ap_now"]
        f107_hist27, ap_hist40, doy = b["f107_hist27"], b["ap_hist40"], b["day_of_year"]
        print("# live drivers @ %s: F10.7=%.1f  F10.7a=%.1f  Ap=%.1f" % (b["target_day"], f107_now, f107a, ap_now))
    else:
        if args.f107 is None or args.ap is None:
            raise SystemExit("error: give --f107 and --ap (or use --live)")
        f107_now = args.f107
        f107a = args.f107a if args.f107a is not None else args.f107
        ap_now = args.ap
        f107_hist27 = (_parse_list(args.f107_hist27, 27, "f107-hist27")
                       if args.f107_hist27 else np.full(27, f107_now, np.float32))
        ap_hist40 = (_parse_list(args.ap_hist40, 40, "ap-hist40")
                     if args.ap_hist40 else np.full(40, ap_now, np.float32))
        doy = args.doy

    out = model.predict_profile(
        alt_km=alt, lat_deg=args.lat, lon_deg=args.lon, local_solar_time_h=args.lst, day_of_year=doy,
        f107_now=f107_now, f107a=f107a, ap_now=ap_now, f107_hist27=f107_hist27, ap_hist40=ap_hist40)

    # --- print a readable table ---
    print("# lat=%.1f lon=%.1f LST=%.1fh doy=%d | F10.7=%.1f F10.7a=%.1f Ap=%.1f"
          % (args.lat, args.lon, args.lst, doy, f107_now, f107a, ap_now))
    print("# alt[km]   rho[kg/m3]     rho_lo90       rho_hi90")
    for i, a in enumerate(alt):
        print("  %6.1f   %.4e   %.4e   %.4e" % (a, out["rho"][i], out["rho_lo90"][i], out["rho_hi90"][i]))

    # --- optional CSV ---
    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["alt_km", "rho", "rho_lo90", "rho_hi90", "rho_lo50", "rho_hi50"])
            for i in range(len(alt)):
                w.writerow([alt[i], out["rho"][i], out["rho_lo90"][i], out["rho_hi90"][i],
                            out["rho_lo50"][i], out["rho_hi50"][i]])
        print("# wrote", args.out)

    # --- optional plot ---
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise SystemExit("error: --plot needs matplotlib (pip install matplotlib)")
        fig, ax = plt.subplots(figsize=(4.2, 4.6))
        ax.fill_betweenx(alt, out["rho_lo90"], out["rho_hi90"], color="#2171b5", alpha=0.2, lw=0,
                         label="90% interval")
        ax.plot(out["rho"], alt, "-", color="#2171b5", lw=2, label="DANTE median")
        ax.set_xscale("log")
        ax.set_xlabel("neutral density (kg m$^{-3}$)")
        ax.set_ylabel("altitude (km)")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        print("# wrote", args.plot)

    return 0


if __name__ == "__main__":
    sys.exit(main())
