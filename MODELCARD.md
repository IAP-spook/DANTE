# DANTE — deployable model card

**What it is.** DANTE (Density via ANchored Transformer Ensemble) predicts thermospheric neutral
mass density (kg/m³) and a calibrated uncertainty band as a continuous profile over ~225–540 km,
from only two space-weather drivers (F10.7 and ap). This package is the headline 5-seed ensemble
retrained on the F10.7-despiked (v2) data, with weights saved for real-time inference.

**Accuracy (held-out test set, n=105,461).** Ensemble MAPE 13.9% overall; storm strata ~17.6 / 22.5 /
21.7% (Dst≤−50 / Dst≤−100 / ap≥100). Better than NRLMSISE-00 (~3×) and JB2008 (~1.8×); on par with
or better than the open ML SOTA (KML). Single-seed is ~0.5 pp worse overall but up to ~5 pp worse in
storms, so the 5-seed ensemble is used.

**Inputs (per query).**
- static: altitude grid [km], latitude [deg], longitude [deg], local solar time [h], day-of-year,
  current F10.7 [sfu], 81-day F10.7 average, current daily Ap.
- histories: 27 daily F10.7 (solar-rotation memory) and 40 three-hourly ap (5 days).
- Nowcast = observed histories/indices; 24-h forecast = SWPC-predicted F10.7/ap for the target day.

**Output.** median density and calibrated 90% (and 50%) interval at each altitude.

## Files
- `dante_model.py` — self-contained architecture (transformer two-stream, static-query cross-attention)
  + Vallado (2013) piecewise-exponential anchor. No karman/pipeline dependency.
- `dante.py` — `DANTE(weights_dir).predict_profile(...)` inference API.
- `train_export.py <seed>` — reproduces a headline seed on `arch_cache5_train_v2clean.npz` and saves weights.
- `fit_conformal.py` — verifies the exported ensemble reproduces the headline and fits/saves the CQR calibrators.
- `test_infer.py` — end-to-end smoke test (quiet vs storm what-if).
- `weights/` — `seed0..4.pt` (state_dicts), `scaler.npz` (mu, sd), `config.json`, `conformal.json`.

## Reproduce
```
python train_export.py 0   # ... through seed 4  (or run_deploy_train.sh)
python fit_conformal.py    # verify + write conformal.json
python test_infer.py       # smoke test
```

## Physics anchor / target
Target y = log10(ρ) − log10(ρ_exp); recover ρ = ρ_exp · 10^q, where ρ_exp is the Vallado
piecewise-exponential atmosphere. This guarantees positivity and the correct order-of-magnitude
vertical structure.

## Scope & limits
Total mass density only (no species/temperature), ~225–540 km; downward extrapolation below the
lowest training altitudes is the weakest regime (Fig S4). Forecast-mode intervals do not yet propagate
the SWPC driver-forecast error, so forecast-mode coverage in storms is optimistic.
