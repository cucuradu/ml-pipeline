"""
GARCH Scaling Unit Test
=======================
Validates the *100 → /10000 → √252 chain that converts arch library's
scaled variance output to annualized volatility percentage.

This is a known source of silent errors in published research.
A fixed synthetic series locks in the expected behavior.
"""

import numpy as np
import pytest
from arch import arch_model


def test_garch_scaling_chain():
    """
    Fit GARCH on synthetic returns, verify the scaling chain produces
    annualized vol in a sane range (not 0, not 1000%).
    """
    np.random.seed(42)
    # Synthetic daily log returns: mean ~0, daily vol ~1%
    n = 500
    returns = np.random.normal(0, 0.01, n)

    # Scale by 100 (same as production code)
    scaled_returns = returns * 100

    am = arch_model(scaled_returns, vol="GARCH", p=1, q=1, dist="t", mean="Constant")
    res = am.fit(disp="off", update_freq=0)

    # --- 1-day forecast ---
    fcast_1d = res.forecast(horizon=1, align="origin")
    var_scaled = fcast_1d.variance.iloc[-1, 0]

    # De-scale: divide by 100^2
    var_unscaled = var_scaled / 10000.0

    # Daily vol → annualized
    daily_vol = np.sqrt(var_unscaled)
    annual_vol = daily_vol * np.sqrt(252)

    # Synthetic data has ~1% daily vol → ~15.87% annualized
    # Allow generous range because GARCH estimates vary with seed
    assert 0.05 < annual_vol < 0.50, (
        f"1d annualized vol {annual_vol:.4f} outside [5%, 50%] for 1% daily vol input"
    )

    # --- 5-day forecast ---
    fcast_5d = res.forecast(horizon=5, align="origin")
    vars_5d_scaled = fcast_5d.variance.iloc[-1, :].values
    vars_5d = vars_5d_scaled / 10000.0
    cum_5d_vol = np.sqrt(np.sum(vars_5d))
    annual_5d = cum_5d_vol * np.sqrt(252.0 / 5.0)

    # 5-day annualized should be in similar ballpark as 1-day annualized
    assert 0.05 < annual_5d < 0.50, (
        f"5d annualized vol {annual_5d:.4f} outside [5%, 50%] for 1% daily vol input"
    )

    # 5d should be within 2x of 1d (they measure the same underlying process)
    ratio = annual_5d / annual_vol
    assert 0.5 < ratio < 2.0, (
        f"5d/1d ratio {ratio:.2f} is outside [0.5, 2.0] — possible scaling error"
    )


def test_garch_descale_identity():
    """
    Verify that scaling returns by 100 and then de-scaling variance by 10000
    produces the same result as fitting on unscaled returns directly.
    """
    np.random.seed(123)
    returns = np.random.normal(0, 0.01, 300)

    # Path A: scale → fit → de-scale (production path)
    scaled = returns * 100
    am_a = arch_model(scaled, vol="GARCH", p=1, q=1, dist="normal", mean="Constant")
    res_a = am_a.fit(disp="off", update_freq=0)
    var_a = res_a.forecast(horizon=1, align="origin").variance.iloc[-1, 0] / 10000.0

    # Path B: fit directly on unscaled (reference path)
    am_b = arch_model(returns, vol="GARCH", p=1, q=1, dist="normal", mean="Constant")
    res_b = am_b.fit(disp="off", update_freq=0)
    var_b = res_b.forecast(horizon=1, align="origin").variance.iloc[-1, 0]

    # They should agree to within 15% (optimizer paths diverge slightly at
    # different scales — arch even warns about poor convergence on unscaled data.
    # This tolerance still catches true scaling bugs like missing /10000).
    np.testing.assert_allclose(var_a, var_b, rtol=0.15,
                               err_msg="Scaled/de-scaled variance doesn't match direct fit")
