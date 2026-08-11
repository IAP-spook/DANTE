# DANTE — model card

**What it is.** DANTE (Density via ANchored Transformer Ensemble) predicts thermospheric neutral
mass density (kg/m³) and a calibrated uncertainty band as a continuous profile over ~225–540 km,
from only two space-weather drivers (F10.7 and ap). This package is the headline 5-seed ensemble
with weights saved for real-time inference.

**Architecture.** A lightweight two-stream Transformer with static-query cross-attention (~35k
parameters), anchored to the Vallado piecewise-exponential reference atmosphere. The network
predicts the log-space anomaly `y = log10(ρ) − log10(ρ_exp)` and density is recovered as
`ρ = ρ_exp · 10^y`; this guarantees positivity and the correct order-of-magnitude vertical
structure. Predictions are five quantiles, ensembled over five seeds and calibrated with
conformalized quantile regression (CQR).

**Accuracy (held-out test set, n = 105,461).** Ensemble MAPE **13.9 %** overall; storm strata
**17.6 / 22.5 / 21.7 %** (Dst ≤ −50 / Dst ≤ −100 / ap ≥ 100). About **2.5×** more accurate than
NRLMSIS 2.0 and **1.8×** more accurate than the drag-optimised JB2008; most accurate model in every
storm stratum. The 5-seed ensemble is ~0.5 pp better overall than a single seed, and up to ~5 pp
better in storms.

**24-hour forecast.** Driven by the real SWPC one-day-ahead F10.7/ap forecasts, DANTE retains
**17.6 %** MAPE at 24-hour lead (22.0 % in storms), roughly 2.3× better than an empirical forecast.

**Inputs (per query).**
- static: altitude grid [km], latitude [deg], longitude [deg], local solar time [h], day-of-year,
  current F10.7 [sfu], 81-day F10.7 average, current daily Ap.
- histories: 27 daily F10.7 (solar-rotation memory) and 40 three-hourly ap (5 days).
- Nowcast = observed histories/indices; 24-h forecast = SWPC-predicted F10.7/ap for the target day.

**Output.** Median density and calibrated 90 % (and 50 %) intervals at each altitude.

## Files
- `dante/model.py` — self-contained architecture (two-stream Transformer, static-query
  cross-attention) + Vallado (2013) piecewise-exponential anchor. No external-pipeline dependency.
- `dante/inference.py` — `DANTE().predict_profile(...)` inference API.
- `dante/drivers.py` — live F10.7/ap driver fetching from the public GFZ/SWPC feeds.
- `dante/cli.py` — the `dante-predict` command-line tool.
- `dante/weights/` — `seed0..4.pt` (state dicts), `scaler.npz` (mu, sd), `config.json`, `conformal.json`.

## Physics anchor / target
Target `y = log10(ρ) − log10(ρ_exp)`; recover `ρ = ρ_exp · 10^y`, where `ρ_exp` is the Vallado
piecewise-exponential atmosphere. Ablating the anchor (predicting absolute log-density directly)
slows convergence and worsens the optimum from 13.9 % to ~19.8 % MAPE, so the anchor is essential.

## Scope & limits
- Total mass density only (no composition or temperature), ~225–540 km. Not a full replacement for
  NRLMSIS.
- Downward extrapolation below the lowest training altitudes is the weakest regime.
- The conformal guarantee is **marginal**, not conditional: the 90 % intervals are mildly
  undercovered in the strongest storms (~84 % at Dst ≤ −100 nT).
- In forecast mode the intervals reflect model uncertainty at the predicted drivers; they do not yet
  propagate the SWPC driver-forecast error, so forecast-mode storm coverage is optimistic.
- A single solar proxy (F10.7); multi-band EUV indices are not used because they are not issued as
  operational day-ahead forecasts.
