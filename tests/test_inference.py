#!/usr/bin/env python3
"""Strict tests for the deployed DANTE model.

These lock the shipped weights + code to known-good behaviour:

  * ``test_golden_values`` is a regression test — the bundled 5-seed ensemble must
    reproduce the exact density profile it produced when the paper's numbers were
    frozen. Any change to the weights, the feature encoding, the anchor, or the
    conformal deltas will trip this.
  * the remaining tests check the physics/statistics the model must always obey
    (density decreases with altitude, storms inflate density, the calibrated
    intervals are correctly ordered, and predictions are deterministic).

Run with:  pytest -q     (needs torch + numpy)
"""
import numpy as np
import pytest

from dante import DANTE

# --- one fixed reference scenario and its frozen output (generated from the shipped weights) ---
_SCENARIO = dict(
    alt_km=np.array([250., 350., 450.], np.float32),
    lat_deg=-12.0, lon_deg=100.0, local_solar_time_h=14.0, day_of_year=78,
    f107_now=135.0, f107a=128.0, ap_now=15.0,
    f107_hist27=np.full(27, 135.0, np.float32), ap_hist40=np.full(40, 15.0, np.float32),
)
_GOLDEN_RHO = [7.746584623768982e-11, 1.226139762638967e-11, 2.5365842239166714e-12]
_GOLDEN_LO90 = [6.64633348357313e-11, 1.0375070594315083e-11, 2.09670497110237e-12]
_GOLDEN_HI90 = [9.039027859536475e-11, 1.4511014785612009e-11, 3.0791328688162256e-12]


@pytest.fixture(scope="module")
def model():
    return DANTE()


def test_golden_values(model):
    """Shipped model must reproduce the frozen reference profile (regression guard)."""
    out = model.predict_profile(**_SCENARIO)
    np.testing.assert_allclose(out["rho"], _GOLDEN_RHO, rtol=1e-4)
    np.testing.assert_allclose(out["rho_lo90"], _GOLDEN_LO90, rtol=1e-4)
    np.testing.assert_allclose(out["rho_hi90"], _GOLDEN_HI90, rtol=1e-4)


def test_output_shapes(model):
    out = model.predict_profile(**_SCENARIO)
    for k in ("alt_km", "rho", "rho_lo90", "rho_hi90", "rho_lo50", "rho_hi50"):
        assert out[k].shape == (3,)
    assert np.all(np.isfinite(out["rho"]))
    assert np.all(out["rho"] > 0)                       # density is strictly positive


def test_interval_ordering(model):
    """Calibrated intervals must nest correctly: lo90 <= lo50 <= median <= hi50 <= hi90."""
    out = model.predict_profile(**_SCENARIO)
    assert np.all(out["rho_lo90"] <= out["rho_lo50"] + 1e-30)
    assert np.all(out["rho_lo50"] <= out["rho"] + 1e-30)
    assert np.all(out["rho"] <= out["rho_hi50"] + 1e-30)
    assert np.all(out["rho_hi50"] <= out["rho_hi90"] + 1e-30)


def test_density_decreases_with_altitude(model):
    """Physics: neutral density must fall monotonically with altitude."""
    alt = np.arange(250, 511, 20.0)
    s = dict(_SCENARIO)
    s["alt_km"] = alt
    out = model.predict_profile(**s)
    assert np.all(np.diff(out["rho"]) < 0)


def test_storm_inflates_density(model):
    """Physics: a geomagnetic storm must raise density relative to quiet conditions."""
    alt = np.arange(300, 501, 50.0)
    common = dict(lat_deg=0.0, lon_deg=100.0, local_solar_time_h=14.0, day_of_year=80)
    quiet = model.predict_profile(alt_km=alt, f107_now=100, f107a=100, ap_now=5,
                                  f107_hist27=np.full(27, 100.0), ap_hist40=np.full(40, 5.0), **common)
    storm = model.predict_profile(alt_km=alt, f107_now=160, f107a=150, ap_now=150,
                                  f107_hist27=np.full(27, 160.0), ap_hist40=np.full(40, 150.0), **common)
    assert np.all(storm["rho"] > quiet["rho"])


def test_determinism(model):
    """Two identical calls must give bit-identical results (no hidden randomness)."""
    a = model.predict_profile(**_SCENARIO)["rho"]
    b = model.predict_profile(**_SCENARIO)["rho"]
    np.testing.assert_array_equal(a, b)


def test_history_length_validation(model):
    """Wrong history lengths must raise a clear ValueError, not silently mispredict."""
    s = dict(_SCENARIO)
    s["ap_hist40"] = np.full(39, 15.0, np.float32)     # one short
    with pytest.raises(ValueError):
        model.predict_profile(**s)
