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
    Fits GARCH(p,q) model and returns forecasts for T+1 and T+1..T+5.
    Arch models optimize better when data is scaled (e.g., returns * 100).
    
    Returns:
        res: fitted ARCHModelResult
        annualized_vol_1d: annualized 1-day ahead vol (decimal)
        annualized_vol_5d: annualized 5-day cumulative vol (decimal)
    """
    logger.info(f"Fitting GARCH({p},{q}) model (Distribution: {dist})...")
    
    # Scale returns by 100 for optimizer stability
    scaled_returns = returns * 100
    
    # Define model
    # mean="Zero" since daily stock returns have mean ~0. 
    # Alternatively, "Constant" can be used. We'll use "Constant".
    am = arch_model(scaled_returns, vol="GARCH", p=p, q=q, dist=dist, mean="Constant")
    
    # Fit model (suppress verbose output)
    res = am.fit(disp="off", update_freq=0)
    logger.info(f"Model fitted. AIC: {res.aic:.2f}, BIC: {res.bic:.2f}")
    
    # --- 1-day forecast ---
    forecasts_1d = res.forecast(horizon=1, align="origin")
    next_day_var_scaled = forecasts_1d.variance.iloc[-1, 0]
    # De-scale variance (divide by 100^2 because we scaled returns by 100)
    next_day_var = next_day_var_scaled / 10000.0
    daily_vol = np.sqrt(next_day_var)
    annualized_vol_1d = daily_vol * np.sqrt(252)
    
    # --- 5-day forecast ---
    # horizon=5 gives variance forecasts for each of the next 5 days.
    # Cumulative 5-day vol = sqrt(sum of 5 daily variances), annualized
    # via sqrt(252/5) to express as annualized rate over a 5-day holding period.
    forecasts_5d = res.forecast(horizon=5, align="origin")
    five_day_vars_scaled = forecasts_5d.variance.iloc[-1, :].values  # 5 values
    five_day_vars = five_day_vars_scaled / 10000.0
    cumulative_5d_vol = np.sqrt(np.sum(five_day_vars))
    annualized_vol_5d = cumulative_5d_vol * np.sqrt(252.0 / 5.0)
    
    logger.info(f"T+1 Forecast -> Daily Vol: {daily_vol*100:.2f}%, Annualized Vol: {annualized_vol_1d*100:.2f}%")
    logger.info(f"T+5 Forecast -> Cumulative 5d Vol: {cumulative_5d_vol*100:.2f}%, Annualized: {annualized_vol_5d*100:.2f}%")
    
    return res, annualized_vol_1d, annualized_vol_5d


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
    model_result, forecast_annual_vol_1d, forecast_annual_vol_5d = fit_garch_and_forecast(
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
    
    # Save Forecast (both 1d and 5d) — consumed by generate_insights.py
    import json
    forecast_path = cfg.DATA_DIR / "features" / "garch_forecast.json"
    forecast_data = {
        "garch_1d_forecast_pct": float(forecast_annual_vol_1d),
        "garch_5d_forecast_pct": float(forecast_annual_vol_5d),
    }
    with open(forecast_path, "w") as f:
        json.dump(forecast_data, f, indent=2)
    logger.info(f"Saved 1d + 5d forecast to {forecast_path.name}")

    logger.info("Phase 3 complete. ✓")

if __name__ == "__main__":
    main()
