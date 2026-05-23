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
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
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
def fit_garch_and_forecast(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "normal"):
    """
    Fits GARCH(p,q) model and returns the forecast for T+1.
    Arch models optimize better when data is scaled (e.g., returns * 100).
    """
    logger.info(f"Fitting GARCH({p},{q}) model (Distribution: {dist})...")
    
    # Scale returns by 100 for optimizer stability
    scaled_returns = returns * 100
    
    # Define model
    # mean="Zero" since daily stock returns have mean ~0. 
    # Alternatively, "Constant" can be used. We'll use "Constant".
    am = arch_model(scaled_returns, vol="Garch", p=p, q=q, dist=dist, mean="Constant")
    
    # Fit model (suppress verbose output)
    res = am.fit(disp="off", update_freq=0)
    logger.info(f"Model fitted. AIC: {res.aic:.2f}, BIC: {res.bic:.2f}")
    
    # Forecast 1 step ahead (for day T+1)
    forecasts = res.forecast(horizon=1, align="origin")
    
    # Get the variance forecast for the last day
    # res.forecast() returns a DataFrame where the last row contains the next step forecast
    next_day_var_scaled = forecasts.variance.iloc[-1, 0]
    
    # De-scale variance (divide by 100^2)
    next_day_var = next_day_var_scaled / 10000.0
    
    # Convert daily variance to daily vol, then annualize
    daily_vol = np.sqrt(next_day_var)
    annualized_vol = daily_vol * np.sqrt(252)
    
    logger.info(f"T+1 Forecast -> Daily Vol: {daily_vol*100:.2f}%, Annualized Vol: {annualized_vol*100:.2f}%")
    
    return res, annualized_vol


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

    # 2. Fit & Forecast
    model_result, forecast_annual_vol = fit_garch_and_forecast(
        returns, 
        p=cfg.GARCH_P, 
        q=cfg.GARCH_Q, 
        dist=cfg.GARCH_DIST
    )
    
    # Save Model
    model_path = cfg.MODELS_DIR / "garch_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_result, f)
    logger.info(f"Saved GARCH model result to {model_path.name}")
    
    # Save Forecast to a simple text file so Phase 4 can read it easily
    # (Though Phase 4 will just call this function or re-run)
    forecast_path = cfg.DATA_DIR / "features" / "garch_forecast.json"
    pd.Series({"garch_1d_forecast_pct": float(forecast_annual_vol)}).to_json(forecast_path)
    logger.info(f"Saved 1D forecast to {forecast_path.name}")

    logger.info("Phase 3 complete. ✓")

if __name__ == "__main__":
    main()
