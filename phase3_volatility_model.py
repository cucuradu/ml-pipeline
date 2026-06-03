#!/usr/bin/env python3
"""
Phase 3 — Volatility Forecasting (GARCH)
========================================
Fits a GARCH(1,1) model on SPY log returns to forecast the 
mathematical variance for the upcoming trading day.

Key outputs:
  • Trained GARCH model object (saved to models/garch_model.pkl)
  • `garch_1d_forecast_pct`: Annualized expected volatility for T+1.
"""

import logging
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Literal
from arch import arch_model

import config as cfg

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOGS_DIR / "phase3.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Load Data
# ============================================================
def load_returns(features_dir: Path) -> pd.Series:
    """Loads SPY features and extracts log returns."""
    csv_path = features_dir / "SPY_features.csv"
    if not csv_path.exists():
        logger.error(f"Missing {csv_path}. Run Phase 1 first.")
        sys.exit(1)

    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    df = df.sort_index()
    
    # We don't shift returns here because arch handles forecasting internally.
    # We fit on all data up to T to forecast T+1.
    if "log_return" not in df.columns:
        logger.error("Missing log_return column in SPY features.")
        sys.exit(1)
        
    return df["log_return"].dropna()


# ============================================================
# 2. Train GARCH and Forecast
# ============================================================
def fit_garch_and_forecast(returns: pd.Series, p: int = 1, q: int = 1, dist: Literal["normal", "t"] = "normal"):
    """
    Fits GARCH(p,q) and returns annualized vol forecasts (in PERCENT) at the
    1-day and 5-day horizons.

    The 1d horizon is sqrt(var_{T+1}); the 5d horizon is the average daily
    variance over T+1..T+5, sqrt'd and annualized. These diverge whenever vol is
    away from its long-run mean (e.g. mean-reverting after a shock) — which is
    exactly the term-structure information the bot wants for sizing. The old
    code forecast only horizon=1, so a "5d" field copied from 1d looked
    identical (followup S4).
    """
    logger.info(f"Fitting GARCH({p},{q}) model (Distribution: {dist})...")

    # Scale returns by 100 for optimizer stability; de-scale variance by 100^2.
    scaled_returns = returns * 100
    am = arch_model(scaled_returns, vol="GARCH", p=p, q=q, dist=dist, mean="Constant")
    res = am.fit(disp="off", update_freq=0)
    logger.info(f"Model fitted. AIC: {res.aic:.2f}, BIC: {res.bic:.2f}")

    # Multi-step variance path for T+1..T+5 (per-day variances, scaled).
    forecasts = res.forecast(horizon=5, align="origin", reindex=False)
    var_path_scaled = forecasts.variance.iloc[-1].to_numpy()  # shape (5,)
    var_path = var_path_scaled / 10000.0                      # de-scale to return units

    daily_var_1d = float(var_path[0])
    daily_var_5d_avg = float(np.mean(var_path[:5]))

    ann_1d_pct = float(np.sqrt(daily_var_1d) * np.sqrt(252) * 100)
    ann_5d_pct = float(np.sqrt(daily_var_5d_avg) * np.sqrt(252) * 100)

    logger.info(
        f"Annualized vol forecast -> 1d: {ann_1d_pct:.2f}%, 5d(avg): {ann_5d_pct:.2f}% "
        f"(term {'rising' if ann_5d_pct > ann_1d_pct else 'falling'})"
    )
    return res, ann_1d_pct, ann_5d_pct


# ============================================================
# Main
# ============================================================
def main() -> None:
    logger.info("=" * 50)
    logger.info("PHASE 3 — Volatility Forecasting (GARCH)")
    logger.info("=" * 50)

    # 1. Load returns
    returns = load_returns(cfg.DATA_DIR / "features")
    logger.info(f"Loaded {len(returns)} days of log returns for GARCH.")

    # 2. Fit & Forecast (annualized vol, already in PERCENT)
    model_result, vol_1d_pct, vol_5d_pct = fit_garch_and_forecast(
        returns,
        p=cfg.GARCH_P,
        q=cfg.GARCH_Q,
        dist=cfg.GARCH_DIST,
    )

    # Save Model
    model_path = cfg.MODELS_DIR / "garch_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_result, f)
    logger.info(f"Saved GARCH model result to {model_path.name}")

    # Save forecast. Values are annualized vol IN PERCENT (e.g. 12.18), so Phase 4
    # consumes them directly — no extra ×100 (the old field was a mislabeled decimal).
    forecast_path = cfg.DATA_DIR / "features" / "garch_forecast.json"
    with open(forecast_path, "w") as f:
        json.dump(
            {"garch_1d_forecast_pct": round(vol_1d_pct, 2),
             "garch_5d_forecast_pct": round(vol_5d_pct, 2)},
            f,
        )
    logger.info(f"Saved 1d/5d forecast to {forecast_path.name}")

    logger.info("Phase 3 complete. ✓")

if __name__ == "__main__":
    main()
