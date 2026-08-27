# DANTE — Density via ANchored Transformer Ensemble

**A physics-anchored Transformer that predicts thermospheric neutral density with calibrated uncertainty.**

DANTE returns a continuous thermospheric mass-density profile from **225 to 540 km** with a
conformally calibrated uncertainty band, using **only the two space-weather drivers that are
operationally forecast a day ahead** — the solar radio flux **F10.7** and the geomagnetic index
**ap**. Because both inputs are forecastable, the same model runs as a *nowcast* and as a genuine
*24-hour forecast*.

- Nowcast accuracy ≈ **13.9 %** mean absolute percentage error (about **2.5×** better than
  NRLMSIS 2.0 and **1.8×** better than the drag-optimised JB2008).
- 24-hour forecast accuracy ≈ **17.6 %**, driven by the real one-day-ahead SWPC forecasts of F10.7 and ap.
- Every prediction carries a conformally calibrated **90 % uncertainty band**.
- Tiny and fast: ~186k parameters (186,405), well under 10 ms on a CPU.

> This repository is the **ready-to-use inference package** — the trained model and a clean API so
> you can predict density from two drivers. It is the code accompanying the paper (Wang & Lyu).

**Author / contact:** Yinan Wang (wangyinan@mail.iap.ac.cn), Institute of Atmospheric Physics,
Chinese Academy of Sciences.

---

## Install

```bash
pip install .                 # core inference (torch + numpy), weights bundled
pip install ".[drivers,plot]" # + live driver fetching (pandas) and plotting (matplotlib)
```

or, without installing, just add the repo to your path (`requirements.txt` lists the deps).

Python ≥ 3.9. The trained weights ship inside the package, so nothing else needs downloading.

---

## Use it in three lines

```python
import numpy as np
from dante import DANTE

model = DANTE()                                        # bundled 5-seed ensemble + conformal calibrators
out = model.predict_profile(
    alt_km             = np.arange(230, 531, 5.0),     # altitude grid [km]
    lat_deg            = -12.0, lon_deg = 100.0,       # geographic latitude / longitude [deg]
    local_solar_time_h = 14.0,  day_of_year = 78,      # local solar time [h], day-of-year
    f107_now           = 135.0, f107a = 128.0,         # current daily F10.7 and its 81-day average [sfu]
    ap_now             = 15.0,                          # current daily Ap
    f107_hist27        = np.full(27, 135.0),           # 27 daily F10.7 (oldest..newest) [sfu]
    ap_hist40          = np.full(40, 15.0))            # 40 three-hourly ap (oldest..newest, 5 days)

out["rho"]                          # median density profile           [kg/m^3]
out["rho_lo90"], out["rho_hi90"]    # calibrated 90 % band
out["rho_lo50"], out["rho_hi50"]    # calibrated 50 % band
```

- **Nowcast:** feed the *observed* driver histories and current indices.
- **24-hour forecast:** feed the **SWPC one-day-ahead predicted** F10.7 and ap for the target day
  (`dante.drivers.fetch_forecast_indices()`).

### Command line

A `dante-predict` command is installed with the package:

```bash
# steady-state "what-if" at given indices
dante-predict --f107 135 --ap 15 --lat -12 --lon 100 --lst 14 --doy 78

# live nowcast from the public GFZ/SWPC feeds (needs pandas + internet)
dante-predict --live --lat 0 --lon 100 --lst 14

# full profile to CSV + a plot
dante-predict --f107 160 --ap 120 --alt-min 230 --alt-max 530 --alt-step 5 --out profile.csv --plot profile.png
```

### Examples

| Script | What it shows |
|---|---|
| `examples/01_basic_profile.py` | minimal quiet-vs-storm profile (offline) |
| `examples/02_live_nowcast.py`  | live nowcast from the real GFZ/SWPC feeds + plot |
| `examples/03_batch_csv.py`     | batch prediction over many conditions from a CSV |

---

## Repository layout

```
DANTE/
├── dante/                 # the inference package
│   ├── model.py           #   TwoStreamNet architecture + Vallado exponential physics anchor
│   ├── inference.py       #   DANTE().predict_profile(...) — the main API
│   ├── drivers.py         #   fetch live F10.7/ap drivers for a real-time nowcast/forecast
│   ├── cli.py             #   `dante-predict` command-line tool
│   └── weights/           #   trained model: seed0-4.pt, scaler.npz, config.json, conformal.json
├── examples/              # runnable usage examples (01_basic, 02_live, 03_batch)
├── tests/                 # strict tests: golden-value regression + physics/calibration sanity
├── MODELCARD.md           # model card (inputs, accuracy, scope, limitations)
├── OPERATIONS.md          # notes for running a live operational nowcast/forecast
├── pyproject.toml         # pip-installable; installs the `dante-predict` command
├── requirements.txt
└── LICENSE                # MIT
```

---

## Inputs, in one place

| Input | Meaning | Notes |
|---|---|---|
| `alt_km` | altitude grid [km] | any grid within ~225–540 km |
| `lat_deg`, `lon_deg` | geographic latitude / longitude [deg] | |
| `local_solar_time_h` | local solar time [h] | 0–24 |
| `day_of_year` | 1–366 | seasonal term |
| `f107_now`, `f107a` | current daily F10.7 and its 81-day average [sfu] | |
| `ap_now` | current daily Ap | |
| `f107_hist27` | 27 daily F10.7 (oldest→newest) [sfu] | one solar rotation of memory |
| `ap_hist40` | 40 three-hourly ap (oldest→newest) | 5 days of memory |

For a nowcast these are all *observed*; for a 24-hour forecast, the target-day F10.7/ap are the
**SWPC one-day-ahead predictions**. `dante.drivers.fetch_nowcast_drivers()` builds the whole bundle
for you from the public feeds.

---

## Tests

```bash
pip install pytest
pytest -q
```

The suite pins the shipped weights to a **golden reference profile** (so any accidental change of
weights or feature encoding is caught) and checks the physics/statistics the model must always obey
(density decreasing with altitude, storms inflating density, correctly nested calibrated intervals,
and deterministic output).

---

## Provenance & tests

The bundled weights are the exact headline 5-seed ensemble reported in the paper (the Transformer
backbone, 186,405 parameters). The held-out test set, evaluation protocol, and the reported
**13.9 % nowcast MAPE** are described in the paper; this repository ships the trained model and a
clean inference API, not the training/evaluation pipeline or any private data. To guard the shipped
weights, the bundled tests pin them to a golden reference profile, so any accidental change to the
weights or feature encoding is caught (`pytest -q`).

---

## Data sources (for building your own live drivers)

| Data | Source | Link |
|---|---|---|
| Thermospheric density (POD) — training target | TU Delft TOLEOS v02 (CC BY 4.0) | http://thermosphere.tudelft.nl |
| Operational one-day-ahead F10.7/ap forecasts | NOAA SWPC | https://www.swpc.noaa.gov |
| Daily F10.7 and 3-hourly ap / Kp indices | GFZ Potsdam | https://kp.gfz-potsdam.de |
| Hourly OMNI Dst / ap / solar wind | NASA OMNIWeb | https://omniweb.gsfc.nasa.gov |

---

## Citation

```
Wang, Y., & Lyu, D. (2026). A physics-anchored Transformer forecasts thermospheric neutral density
with calibrated uncertainty.
```

## License

Released under the MIT License (see `LICENSE`).
