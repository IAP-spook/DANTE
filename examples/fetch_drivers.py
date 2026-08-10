#!/usr/bin/env python3
"""Real-time driver fetcher for operational DANTE. Pulls the public GFZ Kp/ap/F10.7 files
(near-real-time nowcast + full archive), builds the exact driver inputs DANTE was trained on
(F10.7adj, 3-hourly ap), applies the same F10.7 despiking, and returns the nowcast driver bundle.
Optional 24-h/multi-day forecast drivers come from the SWPC 27-day outlook.

Consistency with training: F10.7 = F10.7adj (GFZ column 26, as in the training index file);
ap history stream = 3-hourly ap (columns 15-22); daily Ap = column 23; despike >300 sfu or <=0."""
import io, urllib.request, numpy as np, pandas as pd

GFZ_NOW = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_nowcast.txt"
GFZ_ALL = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
SWPC_27 = "https://services.swpc.noaa.gov/text/27-day-outlook.txt"

def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "DANTE/1.0"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")

def _parse_gfz(text):
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
            Ap = float(c[23]); f107 = float(c[26])          # daily Ap ; F10.7adj (training column)
            rows.append((day, f107, Ap, *ap8))
        except Exception:
            continue
    cols = ["day", "f107", "Ap"] + ["ap%d" % k for k in range(1, 9)]
    return pd.DataFrame(rows, columns=cols)

def fetch_nowcast_drivers(target_day=None):
    """Return the driver bundle for a nowcast at `target_day` (default: latest available)."""
    df = pd.concat([_parse_gfz(_get(GFZ_ALL)), _parse_gfz(_get(GFZ_NOW))], ignore_index=True)
    df = df.drop_duplicates("day", keep="last").sort_values("day").reset_index(drop=True)
    # daily F10.7: despike (same rule as training) + interpolate missing/sentinel
    f = df.set_index("day")["f107"].copy()
    f[(f <= 0) | (f > 300)] = np.nan
    f = f.interpolate("time", limit_direction="both")
    df["f107c"] = f.values
    # 3-hourly ap series (flatten ap1..ap8 by day), sentinel -1 -> NaN -> ffill
    apw = df.melt(id_vars="day", value_vars=["ap%d" % k for k in range(1, 9)], var_name="slot", value_name="ap")
    apw["slot"] = apw["slot"].str[2:].astype(int)
    apw = apw.sort_values(["day", "slot"]).reset_index(drop=True)
    apw.loc[apw["ap"] < 0, "ap"] = np.nan
    apw["ap"] = apw["ap"].ffill().bfill()
    # latest complete calendar day = last day whose F10.7 is real (not just interpolated future)
    if target_day is None:
        valid = df[df["f107"].between(1, 300)]["day"]
        target_day = valid.max()
    target_day = pd.Timestamp(target_day)
    dsel = df[df["day"] <= target_day]
    f107_hist27 = dsel["f107c"].values[-27:].astype(np.float32)
    f107a = float(np.mean(dsel["f107c"].values[-81:]))
    f107_now = float(dsel["f107c"].values[-1])
    ap_now = float(dsel["Ap"].values[-1])
    apsel = apw[apw["day"] <= target_day]
    ap_hist40 = apsel["ap"].values[-40:].astype(np.float32)
    doy = int(target_day.dayofyear)
    return {"target_day": str(target_day.date()), "day_of_year": doy,
            "f107_now": f107_now, "f107a": f107a, "ap_now": ap_now,
            "f107_hist27": f107_hist27, "ap_hist40": ap_hist40}

def fetch_forecast_indices():
    """SWPC 27-day outlook: {date -> (f107, Ap)} one-day-ahead+ forecasts for forecast mode."""
    out = {}
    for ln in _get(SWPC_27).splitlines():
        c = ln.split()
        if len(c) >= 6 and c[0].isdigit() and c[1].isalpha():
            try:
                d = pd.Timestamp("%s %s %s" % (c[0], c[1], c[2]))
                out[str(d.date())] = (float(c[3]), float(c[4]))  # Radio Flux 10.7, Planetary A Index
            except Exception:
                continue
    return out

if __name__ == "__main__":
    b = fetch_nowcast_drivers()
    print("nowcast drivers @", b["target_day"], "doy", b["day_of_year"])
    print("  F10.7_now=%.1f  F10.7a(81d)=%.1f  Ap_now=%.1f" % (b["f107_now"], b["f107a"], b["ap_now"]))
    print("  F10.7 hist27 last5:", np.round(b["f107_hist27"][-5:], 1))
    print("  ap hist40 last8:", np.round(b["ap_hist40"][-8:], 1))
