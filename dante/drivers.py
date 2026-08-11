#!/usr/bin/env python3
"""Fetch the live space-weather drivers DANTE needs, straight from the public sources.

DANTE needs only two drivers, and this module builds them in exactly the encoding the
model was trained on:

  * ``fetch_nowcast_drivers()``  -> the observed F10.7/ap window for a **nowcast**
    (GFZ Potsdam Kp/ap/F10.7 files: near-real-time nowcast + full archive).
  * ``fetch_forecast_indices()`` -> the SWPC one-day-ahead+ predicted F10.7/Ap for a
    **24-hour (or multi-day) forecast** (SWPC 27-day outlook).

Consistency with training (do not change without re-checking against the paper):
    F10.7      = F10.7adj  (GFZ column 26, the training index)
    ap history = 3-hourly ap (GFZ columns 15-22)
    daily Ap   = GFZ column 23
    despiking  = drop F10.7 > 300 sfu or <= 0, then interpolate (same rule as training)

Requires ``pandas`` (an optional dependency; only these live-driver helpers use it).
"""
import numpy as np

try:
    import urllib.request
    import pandas as pd
except ImportError as _e:  # pragma: no cover
    raise ImportError("dante.drivers needs pandas (pip install pandas)") from _e

GFZ_NOW = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_nowcast.txt"
GFZ_ALL = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
SWPC_27 = "https://services.swpc.noaa.gov/text/27-day-outlook.txt"


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "DANTE/1.0"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")


def _parse_gfz(text):
    """Parse a GFZ Kp_ap_Ap_SN_F107 file into a daily DataFrame."""
    rows = []
    for ln in text.splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        c = ln.split()
        if len(c) < 27:
            continue
        try:
            day = pd.Timestamp(int(c[0]), int(c[1]), int(c[2]))
            ap8 = [float(c[15 + k]) for k in range(8)]     # ap1..ap8 (3-hourly)
            Ap = float(c[23])                              # daily Ap
            f107 = float(c[26])                            # F10.7adj (training column)
            rows.append((day, f107, Ap, *ap8))
        except Exception:
            continue
    cols = ["day", "f107", "Ap"] + ["ap%d" % k for k in range(1, 9)]
    return pd.DataFrame(rows, columns=cols)


def fetch_nowcast_drivers(target_day=None):
    """Return the driver bundle for a nowcast at ``target_day`` (default: latest available).

    The returned dict plugs straight into :meth:`dante.DANTE.predict_profile`:
        ``f107_now``, ``f107a``, ``ap_now``, ``f107_hist27``, ``ap_hist40``, ``day_of_year``.
    """
    df = pd.concat([_parse_gfz(_get(GFZ_ALL)), _parse_gfz(_get(GFZ_NOW))], ignore_index=True)
    df = df.drop_duplicates("day", keep="last").sort_values("day").reset_index(drop=True)

    # daily F10.7: despike (same rule as training) + interpolate missing/sentinel values
    f = df.set_index("day")["f107"].copy()
    f[(f <= 0) | (f > 300)] = np.nan
    f = f.interpolate("time", limit_direction="both")
    df["f107c"] = f.values

    # 3-hourly ap series (flatten ap1..ap8 by day); sentinel -1 -> NaN -> fill
    apw = df.melt(id_vars="day", value_vars=["ap%d" % k for k in range(1, 9)],
                  var_name="slot", value_name="ap")
    apw["slot"] = apw["slot"].str[2:].astype(int)
    apw = apw.sort_values(["day", "slot"]).reset_index(drop=True)
    apw.loc[apw["ap"] < 0, "ap"] = np.nan
    apw["ap"] = apw["ap"].ffill().bfill()

    # latest complete day = last day whose F10.7 is a real (not future-interpolated) value
    if target_day is None:
        target_day = df[df["f107"].between(1, 300)]["day"].max()
    target_day = pd.Timestamp(target_day)

    dsel = df[df["day"] <= target_day]
    f107_hist27 = dsel["f107c"].values[-27:].astype(np.float32)
    f107a = float(np.mean(dsel["f107c"].values[-81:]))
    f107_now = float(dsel["f107c"].values[-1])
    ap_now = float(dsel["Ap"].values[-1])
    apsel = apw[apw["day"] <= target_day]
    ap_hist40 = apsel["ap"].values[-40:].astype(np.float32)
    return {"target_day": str(target_day.date()), "day_of_year": int(target_day.dayofyear),
            "f107_now": f107_now, "f107a": f107a, "ap_now": ap_now,
            "f107_hist27": f107_hist27, "ap_hist40": ap_hist40}


def fetch_forecast_indices():
    """SWPC 27-day outlook -> ``{date_str: (f107, Ap)}`` one-day-ahead+ forecasts.

    For a 24-hour forecast, replace the target day's ``f107_now``/``ap_now`` (and the last
    slots of the histories) with the forecast values for that day.
    """
    out = {}
    for ln in _get(SWPC_27).splitlines():
        c = ln.split()
        if len(c) >= 6 and c[0].isdigit() and c[1].isalpha():
            try:
                d = pd.Timestamp("%s %s %s" % (c[0], c[1], c[2]))
                out[str(d.date())] = (float(c[3]), float(c[4]))  # Radio Flux 10.7, Planetary A
            except Exception:
                continue
    return out


if __name__ == "__main__":
    b = fetch_nowcast_drivers()
    print("nowcast drivers @", b["target_day"], "doy", b["day_of_year"])
    print("  F10.7_now=%.1f  F10.7a(81d)=%.1f  Ap_now=%.1f" % (b["f107_now"], b["f107a"], b["ap_now"]))
    print("  F10.7 hist27 last5:", np.round(b["f107_hist27"][-5:], 1))
    print("  ap hist40 last8:", np.round(b["ap_hist40"][-8:], 1))
