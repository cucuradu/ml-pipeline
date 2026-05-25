#!/usr/bin/env python3
"""
Phase 6 — Systemic Fragility (PCA Absorption Ratio)
===================================================
Treats the 11 sector ETFs as a highly dimensional system and measures how
tightly coupled they are via Principal Component Analysis (PCA).

Output:
  • `systemic_fragility`: A score from 0.0 to 1.0 (Variance explained by PC1).
    High score means the entire market is tightly coupled (fragile monolith).
    Low score means sectors are moving independently (robust rotation).
"""

import logging
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

import config as cfg

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOGS_DIR / "phase6.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def build_returns_matrix(features_dir: Path, window: int = 30) -> pd.DataFrame:
    """Builds a DataFrame of daily log returns for all sectors over the last N days."""
    series_list = {}
    
    for sector in cfg.SECTOR_TICKERS:
        csv_path = features_dir / f"{sector}_features.csv"
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        # Get the last `window` rows
        ret = df["log_return"].dropna().tail(window)
        series_list[sector] = ret
        
    return pd.DataFrame(series_list)


def calculate_fragility(returns_matrix: pd.DataFrame) -> float:
    """
    Calculates the Systemic Fragility score using PCA Absorption Ratio.
    1. Computes the Pearson correlation matrix of sector returns.
    2. Extracts the eigenvalues using PCA.
    3. The fragility score is the percentage of variance explained by the 1st Principal Component.
       If PC1 explains the vast majority of variance, the market is moving as a highly correlated monolith.
    """
    # 1. Correlation Matrix
    corr = returns_matrix.corr()
    
    # 2. PCA Absorption Ratio
    # eigh calculates eigenvalues for symmetric matrices
    eigenvalues, _ = np.linalg.eigh(corr)
    
    # The largest eigenvalue represents the variance captured by the 1st Principal Component
    max_eigenvalue = np.max(eigenvalues)
    sum_eigenvalues = np.sum(eigenvalues)
    
    fragility = max_eigenvalue / sum_eigenvalues
    
    logger.info(f"Market Sectors Analyzed: {len(corr.columns)}")
    logger.info(f"Top Eigenvalue (PC1 Variance): {max_eigenvalue:.4f} / {sum_eigenvalues:.4f}")
    logger.info(f"Systemic Fragility Score (Absorption Ratio): {fragility:.4f}")
    
    return float(fragility)


def main():
    logger.info("=" * 50)
    logger.info("PHASE 6 — Systemic Fragility (PCA Absorption)")
    logger.info("=" * 50)
    
    features_dir = cfg.DATA_DIR / "features"
    returns_matrix = build_returns_matrix(features_dir, window=30)
    
    if returns_matrix.empty:
        logger.error("No sector data found.")
        sys.exit(1)
        
    fragility = calculate_fragility(returns_matrix)
    
    out_path = features_dir / "systemic_fragility.json"
    with open(out_path, "w") as f:
        json.dump({"systemic_fragility": fragility}, f)
        
    logger.info(f"Saved fragility score to {out_path.name}")
    logger.info("Phase 6 complete. ✓")

if __name__ == "__main__":
    main()
