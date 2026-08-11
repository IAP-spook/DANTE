"""DANTE — Density via ANchored Transformer Ensemble.

A physics-anchored, quantile-regression Transformer that predicts a continuous thermospheric
neutral-density profile (225-540 km) with conformally calibrated uncertainty, from only the two
operationally forecastable space-weather drivers, the solar radio flux F10.7 and the geomagnetic
index ap. It runs both as a nowcast and as a genuine 24-hour forecast.

Quick start
-----------
    import numpy as np
    from dante import DANTE

    model = DANTE()                                # bundled 5-seed ensemble + conformal calibrators
    out = model.predict_profile(
        alt_km=np.arange(230, 531, 5.0), lat_deg=-12.0, lon_deg=100.0,
        local_solar_time_h=14.0, day_of_year=78,
        f107_now=135.0, f107a=128.0, ap_now=15.0,
        f107_hist27=np.full(27, 135.0),            # 27 daily F10.7 (oldest..newest)
        ap_hist40=np.full(40, 15.0))               # 40 three-hourly ap (oldest..newest)
    out["rho"], out["rho_lo90"], out["rho_hi90"]   # kg/m^3

For a live nowcast, ``dante.drivers.fetch_nowcast_drivers()`` builds the driver bundle
straight from the public GFZ/SWPC feeds. A command-line tool is also installed as
``dante-predict`` (see ``python -m dante.cli --help``).

Author: Yinan Wang (wangyinan@mail.iap.ac.cn), Institute of Atmospheric Physics, CAS.
"""
from .model import TwoStreamNet, exponential_atmosphere
from .inference import DANTE

# Live-driver helpers depend on pandas; import lazily so ``from dante import DANTE`` never
# fails just because pandas is absent in a pure-inference environment.
try:
    from .drivers import fetch_nowcast_drivers, fetch_forecast_indices
    _HAVE_DRIVERS = True
except Exception:  # pragma: no cover - pandas not installed
    _HAVE_DRIVERS = False

__all__ = ["DANTE", "TwoStreamNet", "exponential_atmosphere"]
if _HAVE_DRIVERS:
    __all__ += ["fetch_nowcast_drivers", "fetch_forecast_indices"]
__version__ = "1.0.0"
