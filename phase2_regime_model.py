#!/usr/bin/env python3
"""
Phase 2 — Regime Classifier (GMM)
=================================
Trains a Gaussian Mixture Model to classify the SPY market environment
into 4 regimes: Bull, Neutral, Caution, Defensive.

Key features:
  • No look-ahead bias: Regime for day T is predicted using features from day T-1.
  • Persistence filter: A new regime must hold for `cfg.REGIME_PERSISTENCE_BARS` (e.g., 3 days)
    before being officially accepted.
  • Heuristic mapping: Clusters are automatically mapped to names based on their
    mean SMA distance (trend) and Realized Volatility.

Outputs:
  • Trained GMM model pipeline (saved to models/gmm_pipeline.pkl)
  • Regime predictions appended to SPY data.
"""

import logging
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline

import config as cfg

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOGS_DIR / "phase2.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Load Data & Shift for No Look-Ahead Bias
# ============================================================
def load_and_prepare_data(features_dir: Path) -> pd.DataFrame:
    """
    Loads SPY features and shifts them by 1 day.
    This ensures we use data from the END of day T-1 to predict the regime for day T.
    """
    csv_path = features_dir / "SPY_features.csv"
    if not csv_path.exists():
        logger.error(f"Missing {csv_path}. Run Phase 1 first.")
        sys.exit(1)

    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    
    # Sort just in case
    df = df.sort_index()

    # We shift all feature columns by 1 to prevent look-ahead bias.
    # The date index represents the trading day T. The shifted features represent T-1.
    feature_cols = [c for c in df.columns if c not in ("ticker", "close")]
    
    # Create shifted dataframe
    df_shifted = df.copy()
    df_shifted[feature_cols] = df[feature_cols].shift(1)
    
    # Drop the first row which will now have NaNs from the shift
    df_shifted = df_shifted.dropna(subset=feature_cols)
    
    logger.info(f"Loaded SPY features and shifted by 1 day (T-1 -> T). Rows: {len(df_shifted)}")
    return df_shifted


# ============================================================
# 2. Train GMM Pipeline
# ============================================================
def train_gmm(df: pd.DataFrame, feature_cols: list[str], n_components: int = 4) -> Pipeline:
    """
    Trains a pipeline with StandardScaler + GaussianMixture.
    """
    logger.info(f"Training GMM with {n_components} components on {len(feature_cols)} features...")
    
    X = df[feature_cols].values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("gmm", GaussianMixture(
            n_components=n_components, 
            covariance_type="full", 
            random_state=cfg.GMM_RANDOM_STATE,
            n_init=5
        ))
    ])

    pipeline.fit(X)
    logger.info("GMM training complete.")
    return pipeline


# ============================================================
# 3. Map Clusters to Human-Readable Regimes
# ============================================================
def map_clusters_to_regimes(pipeline: Pipeline, df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Automatically maps arbitrary GMM cluster labels (0, 1, 2, 3) to 
    'Bull', 'Neutral', 'Caution', 'Defensive' based on cluster properties.
    
    Heuristic: Sort clusters by average sma_dist_200 (trend) and rv_21d (vol).
    Higher trend + lower vol -> Bull
    Lower trend + higher vol -> Defensive
    """
    X = df[feature_cols].values
    preds = pipeline.predict(X)
    df_temp = df.copy()
    df_temp["cluster"] = preds

    # Calculate mean stats per cluster
    cluster_stats = df_temp.groupby("cluster")[["sma_dist_200", "rv_21d"]].mean()
    
    # We create a simple heuristic score: sma_dist_200 - rv_21d (normalized slightly)
    # Higher score = more bullish. 
    # To be safe, we'll standardize these two columns among the 4 centers to compute a score.
    trend = cluster_stats["sma_dist_200"]
    vol = cluster_stats["rv_21d"]
    
    trend_norm = (trend - trend.mean()) / trend.std()
    vol_norm = (vol - vol.mean()) / vol.std()
    
    bullish_score = trend_norm - vol_norm
    
    # Sort clusters by bullish score descending
    sorted_clusters = bullish_score.sort_values(ascending=False).index.tolist()
    
    regime_names = ["Bull", "Neutral", "Caution", "Defensive"]
    
    mapping = {}
    logger.info("Cluster Mapping Statistics:")
    for i, cluster in enumerate(sorted_clusters):
        regime = regime_names[i]
        mapping[cluster] = regime
        stats = cluster_stats.loc[cluster]
        logger.info(f"  {regime} (Cluster {cluster}) -> Trend (SMA200): {stats['sma_dist_200']:.4f}, Vol (RV21): {stats['rv_21d']:.4f}")

    return mapping


# ============================================================
# 4. Apply Persistence Filter
# ============================================================
def apply_persistence_filter(raw_regimes: pd.Series, persistence_bars: int) -> pd.Series:
    """
    A regime change is only accepted if it persists for `persistence_bars` consecutive days.
    Otherwise, we maintain the previous accepted regime.
    """
    filtered = raw_regimes.copy()
    current_regime = raw_regimes.iloc[0]
    
    for i in range(len(raw_regimes)):
        if i < persistence_bars:
            filtered.iloc[i] = current_regime
            continue
            
        # Check if the last N bars are all the same new regime
        recent_bars = raw_regimes.iloc[i - persistence_bars + 1 : i + 1]
        if len(recent_bars.unique()) == 1:
            current_regime = recent_bars.iloc[0]
            
        filtered.iloc[i] = current_regime
        
    return filtered


# ============================================================
# Main
# ============================================================
def main() -> None:
    logger.info("=" * 50)
    logger.info("PHASE 2 — Regime Classifier (GMM)")
    logger.info("=" * 50)

    # 1. Load Data
    df = load_and_prepare_data(cfg.DATA_DIR / "features")
    feature_cols = [c for c in df.columns if c not in ("ticker", "close")]

    # 2. Train Model
    pipeline = train_gmm(df, feature_cols, n_components=cfg.N_REGIMES)

    # 3. Map Clusters
    cluster_mapping = map_clusters_to_regimes(pipeline, df, feature_cols)

    # 4. Predict & Map
    X = df[feature_cols].values
    raw_clusters = pipeline.predict(X)
    raw_regimes = pd.Series(raw_clusters, index=df.index).map(cluster_mapping)
    
    # 5. Apply Persistence Filter
    filtered_regimes = apply_persistence_filter(raw_regimes, cfg.REGIME_PERSISTENCE_BARS)
    
    df["regime_raw"] = raw_regimes
    df["regime"] = filtered_regimes

    # Diagnostics
    logger.info("\nRegime Distribution (Filtered):")
    logger.info(df["regime"].value_counts().to_string())
    
    changes = (df["regime"] != df["regime"].shift(1)).sum() - 1 # subtract 1 for the first row
    logger.info(f"Total Regime Changes (after filter): {changes}")

    # Save Model
    model_path = cfg.MODELS_DIR / "gmm_pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "mapping": cluster_mapping}, f)
    logger.info(f"Saved GMM model to {model_path.name}")

    # Save outputs to data dir for downstream
    output_path = cfg.DATA_DIR / "features" / "SPY_regimes.csv"
    df[["close", "regime_raw", "regime"]].to_csv(output_path)
    logger.info(f"Saved regime predictions to {output_path.name}")

    logger.info("Phase 2 complete. ✓")

if __name__ == "__main__":
    main()
