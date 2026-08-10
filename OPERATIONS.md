# DANTE — operational live nowcast/forecast

Proven end-to-end on 2026-08-06: public real-time drivers -> deployed 5-seed ensemble ->
current density profile with calibrated uncertainty.

## Real-time driver sources (verified live)
- **GFZ Kp/ap/F10.7** (single source, same format as training):
  - nowcast (latest ~30 days): `https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_nowcast.txt`
  - full archive: `https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt`
  - columns used: F10.7adj (col 26, matches training), ap1..ap8 3-hourly (cols 15-22), daily Ap (col 23).
- **SWPC 27-day outlook** (forecast F10.7 + planetary A): `https://services.swpc.noaa.gov/text/27-day-outlook.txt`
- (also available: SWPC 3-day forecast, 10cm-flux-30-day.json, noaa-planetary-k-index.json)

## Loop
1. `fetch_drivers.py` merges GFZ nowcast + archive, uses F10.7adj, despikes (>300 or <=0 -> interpolate,
   same rule as training), and builds: 27-day F10.7 history, 81-day F10.7 average, current F10.7,
   40-slot 3-hourly ap history, current daily Ap.
2. `dante.py` (5-seed ensemble + conformal) maps drivers + (lat, lon, local solar time, day-of-year,
   altitude grid) -> median density + calibrated 90%/50% bands.
3. `live_nowcast.py [lat lon LST]` runs the loop now and writes `figs/live_nowcast.png`.

## Run
```
python live_nowcast.py 0 100 14      # equator, lon 100E, 14h LST, current drivers
```

## Nowcast vs forecast
- Nowcast: uses observed drivers up to the latest available day (the most recent ap/F10.7 are
  provisional GFZ nowcast values, refined later).
- 24-h / multi-day forecast: replace the current-day F10.7/Ap with `fetch_forecast_indices()` (SWPC
  27-day outlook) for the target day. The SWPC F10.7 forecast is accurate (~4% MAE); the ap forecast
  is unreliable in storms, so forecast-mode storm uncertainty is optimistic (see paper).

## To productionise
- A scheduler (cron) to refresh drivers hourly and cache them.
- Wrap `dante.py` in a small FastAPI/Flask endpoint; a front-end form (time, lat, lon, altitude range)
  -> profile plot. Model is tiny (35k params, <10 ms CPU), so latency is trivial.
