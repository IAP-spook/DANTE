# DANTE — Density via ANchored Transformer Ensemble

**A physics-anchored Transformer that forecasts thermospheric neutral density with calibrated uncertainty.**

DANTE predicts a continuous thermospheric mass-density profile from **225 to 540 km** with
conformally calibrated uncertainty, using **only the two space-weather drivers that are operationally
forecast a day ahead** — the solar radio flux **F10.7** and the geomagnetic index **ap**. Because both
inputs are forecastable, DANTE runs both as a *nowcast* and as a *genuine 24-hour forecast*.

- Nowcast accuracy ≈ **13.9 %** mean absolute percentage error (about 2.5× better than NRLMSIS 2.0,
  1.8× better than the drag-optimised JB2008).
- 24-hour forecast accuracy ≈ **17.6 %**, driven by the real one-day-ahead SWPC forecasts of F10.7 and ap.
- Every prediction carries a conformally calibrated 90 % uncertainty band.

This repository accompanies the paper (Wang & Lyu, submitted to *Space Weather*).

**Author / contact:** Yinan Wang (wangyinan@mail.iap.ac.cn), Institute of Atmospheric Physics,
Chinese Academy of Sciences.

---

## Repository layout

```
DANTE/
├── dante/                 # self-contained, runnable inference package (no external data needed)
│   ├── model.py           #   TwoStreamNet architecture + Vallado exponential physics anchor
│   └── inference.py       #   DANTE.predict_profile(...) API (5-seed ensemble + conformal band)
├── weights/               # trained headline model: seed0-4.pt, scaler.npz, config.json, conformal.json
├── examples/
│   ├── run_inference.py   # end-to-end smoke test (quiet vs storm scenario) — verified runnable
│   ├── fetch_drivers.py   # fetch live F10.7/ap indices for a real-time nowcast
│   └── live_nowcast.py    # build a driver window and run a live density profile
├── reproduce/             # scripts that reproduce every number and figure in the paper
│   ├── data/              #   build_cache.py (TOLEOS + indices -> feature cache), f107_clean.py
│   ├── train/             #   train.py (5-seed 4.85M training), models.py, calibrate.py (CQR)
│   ├── baselines/         #   nrlmsis2.py, nrlmsis2_forecast.py, jb2008.py
│   ├── eval/              #   train_eval_4p85M.py, forecast24.py, holdout_generalization.py,
│   │                      #   driver_uncertainty.py, anchor_ablation.py
│   └── figures/           #   fig1-9.py, figS_*.py (paper figures)
├── MODELCARD.md           # model card (inputs, scope, limitations)
├── OPERATIONS.md          # operational-use notes
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
python -m venv venv && source venv/bin/activate    # Python >= 3.9
pip install -r requirements.txt
```

Only `torch` and `numpy` are needed for inference. The `reproduce/` scripts additionally use
`pandas`, `pyarrow`, `matplotlib`, `scikit-learn`, and `pymsis` (NRLMSIS 2.0 baseline).

---

## Quick start — run DANTE (inference)

The trained model is bundled in `weights/`, so inference works out of the box:

```bash
python examples/run_inference.py
```

or from Python:

```python
import numpy as np
from dante import DANTE

model = DANTE("weights")                              # 5-seed ensemble + conformal calibrators
out = model.predict_profile(
    alt_km      = np.arange(230, 531, 2.0),           # altitude grid [km]
    lat_deg     = -12.0, lon_deg = 100.0,
    local_solar_time_h = 14.0, day_of_year = 78,
    f107_now    = 135.0, f107a = 128.0, ap_now = 15.0,# current daily indices
    f107_hist27 = np.full(27, 135.0),                 # 27 daily F10.7 (oldest..newest) [sfu]
    ap_hist40   = np.full(40, 15.0))                  # 40 three-hourly ap (oldest..newest, 5 days)

out["rho"]        # median density profile           [kg/m^3]
out["rho_lo90"], out["rho_hi90"]   # calibrated 90 % band
```

- **Nowcast:** feed the *observed* driver histories and current indices.
- **24-hour forecast:** feed the **SWPC one-day-ahead predicted** F10.7 and ap for the target day
  (see `examples/fetch_drivers.py` / `examples/live_nowcast.py`).

---

## Reproducing the paper

The `reproduce/` scripts regenerate every result. They require the underlying data (see below) and, for
some steps, a CUDA GPU. Each script carries a header docstring describing its role.

1. **Data & indices** — download the datasets in *Data availability* below.
2. **Build the feature cache** — `reproduce/data/build_cache.py` (features, driver histories, splits).
3. **Train** — `reproduce/train/train.py <seed>` for seeds 0-4 (30 epochs each), then
   `reproduce/train/calibrate.py` for the conformal calibrators.
4. **Baselines** — `reproduce/baselines/{nrlmsis2,jb2008}.py`.
5. **Evaluate** — `reproduce/eval/train_eval_4p85M.py`, `forecast24.py`,
   `holdout_generalization.py`, `driver_uncertainty.py`, `anchor_ablation.py`.
6. **Figures** — `reproduce/figures/fig*.py`, `figS_*.py`.

> **Paths.** The reproduction scripts use absolute paths to the local data cache and to a small
> `karman` helper (for the exponential reference atmosphere and accelerometer density utilities).
> Edit the `BN=...`, `GW=...` and `sys.path` lines at the top of each script to point at your own data
> location before running. The self-contained `dante/` inference package has **no** such dependency.

---

## Data availability

| Data | Source | Link |
|---|---|---|
| Thermospheric density (POD) | TU Delft TOLEOS v02 (CC BY 4.0) | http://thermosphere.tudelft.nl |
| Operational one-day-ahead F10.7/ap forecasts | NOAA SWPC RSGA | https://www.swpc.noaa.gov |
| Hourly OMNI Dst / ap / solar wind | NASA OMNIWeb | https://omniweb.gsfc.nasa.gov |
| Daily F10.7 and ap/Kp indices | GFZ Potsdam / NRCan | https://kp.gfz-potsdam.de |

---

## Citation

If you use DANTE, please cite:

> Wang, Y., & Lyu, D. (2026). A physics-anchored Transformer forecasts thermospheric neutral density
> with calibrated uncertainty. *Space Weather* (submitted).

---

## License

Released under the MIT License (see `LICENSE`).
