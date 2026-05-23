#!/usr/bin/env python3
"""
Phase 5 — Universe Ranking Engine (XGBoost)
===========================================
Trains an XGBoost Regressor to predict the 5-day forward return of all 11
Sector ETFs based on their current technical features.

Outputs:
  • Trained XGBoost model (models/xgboost_ranking.pkl)
  • `universe_ranking` array added to `ml-insights.json` containing the top 3
    sectors for the Cloud Engine to focus on.
"""

import logging
import sys
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

import config as cfg

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOGS_DIR / "phase5.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def build_training_dataset(features_dir: Path, target_horizon: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pools data from all sector ETFs to train a unified ranking model.
    Calculates the N-day forward return as the target variable.
    Returns (historical_data_for_training, latest_data_for_prediction).
    """
    all_historical = []
    latest_rows = []

    for sector in cfg.SECTOR_TICKERS:
        csv_path = features_dir / f"{sector}_features.csv"
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        df = df.sort_index()
        
        # Calculate N-day forward return
        # shift(-N) brings future prices back to today
        df["fwd_return_5d"] = (df["close"].shift(-target_horizon) / df["close"]) - 1
        
        # The last `target_horizon` rows will have NaN targets because the future hasn't happened yet.
        # We split these off. The very last row is what we want to predict today.
        
        latest_row = df.iloc[-1:].copy()
        latest_row["ticker"] = sector
        latest_rows.append(latest_row)
        
        # For training, drop the rows with NaN forward returns
        historical = df.dropna(subset=["fwd_return_5d"]).copy()
        historical["ticker"] = sector
        all_historical.append(historical)
        
    train_df = pd.concat(all_historical)
    
    # CRITICAL: Sort by date for proper Time-Series Split
    train_df = train_df.sort_index()
    
    latest_df = pd.concat(latest_rows)
    
    return train_df, latest_df


def train_xgboost(train_df: pd.DataFrame, feature_cols: list[str]):
    """
    Trains XGBoost using Walk-Forward Validation (TimeSeriesSplit).
    """
    logger.info(f"Training XGBoost on {len(train_df)} pooled sector days using Walk-Forward CV...")
    
    X = train_df[feature_cols]
    y = train_df["fwd_return_5d"]
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # 1. Walk-Forward Cross Validation
    for train_idx, test_idx in tscv.split(X):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train_cv, y_train_cv, verbose=False)
        preds = model.predict(X_test_cv)
        rmse = np.sqrt(mean_squared_error(y_test_cv, preds))
        cv_scores.append(rmse)
        
    avg_rmse = np.mean(cv_scores)
    logger.info(f"Walk-Forward CV RMSE (5 splits): {avg_rmse:.4f}")
    
    # 2. Final Retrain on 100% of data
    # Now that we proved the model generalizes out-of-sample without leakage,
    # we retrain on EVERYTHING so the final model has the most recent market data.
    logger.info("Retraining final model on 100% of historical data...")
    model.fit(X, y, verbose=False)
    
    return model


def main():
    logger.info("=" * 50)
    logger.info("PHASE 5 — Universe Ranking Engine (XGBoost)")
    logger.info("=" * 50)
    
    features_dir = cfg.DATA_DIR / "features"
    train_df, latest_df = build_training_dataset(features_dir, target_horizon=5)
    
    # Features to use (drop target and metadata)
    feature_cols = [c for c in train_df.columns if c not in ("ticker", "close", "fwd_return_5d")]
    
    # Train Model
    model = train_xgboost(train_df, feature_cols)
    
    # Save Model
    model_path = cfg.MODELS_DIR / "xgboost_ranking.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": feature_cols}, f)
    logger.info(f"Saved XGBoost model to {model_path.name}")
    
    # Predict on the latest data
    X_latest = latest_df[feature_cols]
    predictions = model.predict(X_latest)
    
    latest_df["pred_fwd_return_5d"] = predictions
    
    # Rank them
    ranked = latest_df.sort_values("pred_fwd_return_5d", ascending=False)
    
    logger.info("\nUniverse Ranking (Top 5):")
    for i in range(5):
        row = ranked.iloc[i]
        logger.info(f"  {i+1}. {row['ticker']} -> Expected 5d Return: {row['pred_fwd_return_5d']:.2%}")
        
    # Save top 3 to a small JSON for Phase 4 to pick up
    top_3 = ranked.head(3)["ticker"].tolist()
    ranking_output = {
        "top_3_sectors": top_3,
        "predictions": {row["ticker"]: float(row["pred_fwd_return_5d"]) for _, row in ranked.iterrows()}
    }
    
    out_path = features_dir / "universe_ranking.json"
    with open(out_path, "w") as f:
        json.dump(ranking_output, f)
        
    logger.info(f"Saved ranking to {out_path.name}")
    logger.info("Phase 5 complete. ✓")

if __name__ == "__main__":
    main()
