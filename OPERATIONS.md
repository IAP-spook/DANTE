# DANTE — operational live nowcast/forecast

End-to-end: public real-time drivers → deployed 5-seed ensemble → current density profile with
calibrated uncertainty.

## Real-time driver sources
- **GFZ Kp/ap/F10.7** (single source, same format as training):
  - nowcast (latest ~30 days): `https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_nowcast.txt`
  - full archive: `https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt`
  - columns used: F10.7adj (col 26, matches training), ap1..ap8 3-hourly (cols 15-22), daily Ap (col 23).
- **SWPC 27-day outlook** (forecast F10.7 + planetary A): `https://services.swpc.noaa.gov/text/27-day-outlook.txt`

## Loop
1. `dante.drivers.fetch_nowcast_drivers()` merges the GFZ nowcast + archive, uses F10.7adj, despikes
   (>300 sfu or ≤0 → interpolate, the same rule as training), and builds: the 27-day F10.7 history,
   the 81-day F10.7 average, the current F10.7, the 40-slot 3-hourly ap history, and the current daily Ap.
2. `dante.DANTE().predict_profile(...)` (5-seed ensemble + conformal) maps drivers + (lat, lon,
   local solar time, day-of-year, altitude grid) → median density + calibrated 90 %/50 % bands.
3. `examples/02_live_nowcast.py [lat lon LST]` runs the whole loop now and writes `live_nowcast.png`.

## Run
```bash
python examples/02_live_nowcast.py 0 100 14     # equator, lon 100E, 14h LST, current drivers
# or the CLI:
dante-predict --live --lat 0 --lon 100 --lst 14
```

## Nowcast vs forecast
- **Nowcast:** observed drivers up to the latest available day (the most recent ap/F10.7 are
  provisional GFZ nowcast values, refined later).
- **24-h / multi-day forecast:** replace the target-day F10.7/Ap with `fetch_forecast_indices()`
  (SWPC 27-day outlook). The SWPC F10.7 forecast is accurate (~4 % MAE); the ap forecast is
  unreliable in storms, so forecast-mode storm uncertainty is optimistic (see the paper). Note: the
  paper's 24-hour benchmark used the SWPC RSGA next-day forecasts; this helper uses the
  machine-readable 27-day outlook for convenience, so deployed forecast skill may differ slightly
  from the paper's reported numbers.

## To productionise
- A scheduler (cron) to refresh and cache drivers hourly.
- Wrap `DANTE` in a small FastAPI/Flask endpoint; a front-end form (time, lat, lon, altitude range)
  → profile plot. The model is tiny (~186k params, <10 ms CPU), so latency is trivial.
