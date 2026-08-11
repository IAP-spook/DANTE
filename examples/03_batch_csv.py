#!/usr/bin/env python3
"""Example 3 — batch prediction over many conditions from a CSV.

Reads a CSV where each row is one prediction column (location, time, drivers) and writes
the median density (and 90% band) at a single reference altitude for every row. This is a
template for driving DANTE from your own tables; adapt the altitude handling as needed.

Input CSV columns (header required):
    lat_deg, lon_deg, lst_h, doy, f107, f107a, ap, alt_km

    python examples/03_batch_csv.py conditions.csv out.csv

If no input file is given, a tiny demo table is generated and used.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run without installing
import csv
import numpy as np
from dante import DANTE


def demo_table(path):
    rows = [
        # lat, lon, lst, doy, f107, f107a, ap, alt
        (0.0, 100.0, 14.0, 80, 100, 100, 5, 400.0),
        (0.0, 100.0, 2.0, 80, 100, 100, 5, 400.0),
        (60.0, 100.0, 14.0, 80, 160, 150, 120, 400.0),
        (-30.0, 20.0, 10.0, 200, 130, 125, 20, 350.0),
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lat_deg", "lon_deg", "lst_h", "doy", "f107", "f107a", "ap", "alt_km"])
        w.writerows(rows)


def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "demo_conditions.csv"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "demo_predictions.csv"
    if len(sys.argv) <= 1:
        demo_table(in_csv)
        print("# no input given -> wrote demo table to", in_csv)

    model = DANTE()
    with open(in_csv) as fh:
        rows = list(csv.DictReader(fh))

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lat_deg", "lon_deg", "lst_h", "doy", "f107", "ap", "alt_km",
                    "rho", "rho_lo90", "rho_hi90"])
        for r in rows:
            f107 = float(r["f107"])
            ap = float(r["ap"])
            alt = np.array([float(r["alt_km"])], np.float32)
            # Steady-state history (constant drivers). For a true nowcast/forecast, pass the
            # real 27-day F10.7 and 40-slot ap histories instead of constant fills.
            out = model.predict_profile(
                alt_km=alt, lat_deg=float(r["lat_deg"]), lon_deg=float(r["lon_deg"]),
                local_solar_time_h=float(r["lst_h"]), day_of_year=int(float(r["doy"])),
                f107_now=f107, f107a=float(r.get("f107a", f107)), ap_now=ap,
                f107_hist27=np.full(27, f107), ap_hist40=np.full(40, ap))
            w.writerow([r["lat_deg"], r["lon_deg"], r["lst_h"], r["doy"], f107, ap, r["alt_km"],
                        "%.4e" % out["rho"][0], "%.4e" % out["rho_lo90"][0], "%.4e" % out["rho_hi90"][0]])
    print("# wrote", out_csv, "(%d rows)" % len(rows))


if __name__ == "__main__":
    main()
