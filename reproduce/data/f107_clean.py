"""Shared F10.7 cleaning for the v2clean retrain (isolated from the original v1 pipeline).

Physical rationale: F10.7 (10.7 cm radio flux) is a proxy for solar EUV/UV irradiance, the
actual thermospheric heat source. On solar radio-burst days (Type II/IV bursts from flares) the
10.7 cm flux gets a large NON-THERMAL contribution that does NOT correspond to a proportional EUV
heating, so those daily values (all >400 sfu in this record, e.g. 2003-11-04, 2006-12-06,
2011-03-07) are radio artifacts, not thermospheric forcing. The GFZ index file also uses -1 as a
missing-data sentinel. We therefore flag F10.7 <= 0 (missing) or > 300 sfu (radio burst; well
above the ~295 sfu ceiling of legitimate thermal solar maximum) and replace them by linear
time-interpolation across flanking clean days, which best recovers the slowly-varying underlying
EUV proxy. f107a (81-day mean) and the F10.7 history stream are then recomputed from the cleaned
series. Affects ~0.35% of days (2000-2024); all other drivers (ap, Ap) and rho/coords are clean.
"""
import numpy as np, pandas as pd

F107_CAP = 300.0   # sfu; legitimate thermal solar-max daily F10.7 stays below ~295, bursts are >400

def clean_f107(gd):
    """Clean the daily F10.7 column of a DatetimeIndex-ed DataFrame `gd` IN PLACE and return it.
    `gd` must have a 'f107' column and a daily DatetimeIndex. Recompute f107a AFTER calling this."""
    bad = (gd["f107"] <= 0) | (gd["f107"] > F107_CAP)
    n_bad = int(bad.sum())
    gd.loc[bad, "f107"] = np.nan
    gd["f107"] = gd["f107"].interpolate("time", limit_direction="both")
    gd.attrs["f107_n_cleaned"] = n_bad
    return gd
