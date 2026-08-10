# Reproducing the DANTE paper

These scripts regenerate every number and figure in Wang & Lyu (submitted to *Space Weather*).

> **Before running:** obtain the data (see the main `README.md` → *Data availability*) and edit the path
> variables at the top of each script (`BN=...` the feature-cache directory, `GW=...`/`MLW=...` the raw
> data directories, and the `sys.path.insert(...)` line that points at the local `karman` helper). The
> self-contained `dante/` inference package does **not** need any of this.

## Pipeline

| Step | Script | Produces |
|---|---|---|
| 1. Feature cache | `data/build_cache.py` | features, F10.7/ap driver histories, train/val/test split (1.2 M cache); `data/f107_clean.py` despikes F10.7 |
| 2. Train (headline) | `train/train.py <seed>` (seeds 0-4) | 5-seed 4.85 M-sample ensemble weights (`weights/seed*.pt`); `train/models.py` = architecture bake-off backbones |
| 3. Calibrate | `train/calibrate.py` | conformal 90 %/50 % calibrators (`weights/conformal.json`) |
| 4. Baselines | `baselines/nrlmsis2.py`, `baselines/jb2008.py`, `baselines/nrlmsis2_forecast.py` | NRLMSIS 2.0 and JB2008 predictions on the identical test points |
| 5. Evaluate | `eval/train_eval_4p85M.py` | headline nowcast MAPE, storm strata (Table 1, Fig 3) |
| | `eval/forecast24.py` | 24-hour forecast, nowcast vs persistence vs SWPC drivers (Table 2, Fig 11) |
| | `eval/holdout_generalization.py` | leave-one-satellite/altitude-band-out generalization (Fig S3) |
| | `eval/driver_uncertainty.py` | driver-forecast-error ceiling and Monte-Carlo density spread (§4.5) |
| | `eval/anchor_ablation.py <seed>` | no-anchor ablation: absolute-log density target (Fig S7) |
| 6. Figures | `figures/fig1.py … fig9.py` | main figures |
| | `figures/figS_condcov.py` | Fig S6 (conditional coverage by storm severity) |
| | `figures/figS_anchor.py` | Fig S7 (physics-anchor ablation) |
| | `figures/figS_residlat.py` | Fig S8 (anchor residual vs latitude) |
| | `figures/figS_xcal.py`, `figS_cadence.py` | Fig 2 (cross-calibration), Fig S2 (sampling cadence) |

## Notes

- Training used a single CUDA GPU; each seed is 30 epochs (AdamW, cosine schedule, pinball/quantile loss,
  batch size 16 384). The headline ensemble is the median of the five seeds' quantiles.
- The physics anchor predicts the log-space density *anomaly* `y = log10(rho_obs) − log10(rho_exp)` and
  recovers `rho = rho_exp · 10^y`; `anchor_ablation.py` removes this anchor to justify the design choice.
- All figure scripts write PNG/PDF/SVG; update the `FIGS=...` output path at the top of each.
