#!/usr/bin/env python3
"""
Phase 6 — Graph-Based Market Structure (GNN / Topology)
=======================================================
Treats the 11 sector ETFs as nodes in a graph. Connects them with edges
based on their rolling 30-day correlations.

Output:
  • `systemic_fragility`: A score from 0.0 to 1.0. 
    High score means the entire market is tightly coupled (fragile).
    Low score means sectors are moving independently (robust).
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


def calculate_fragility(returns_matrix: pd.DataFrame, threshold: float = 0.7) -> float:
    """
    Calculates the Systemic Fragility score using Graph Topology.
    1. Computes the Pearson correlation matrix.
    2. Builds an adjacency matrix (Edge = 1 if correlation > threshold).
    3. Calculates graph density (actual edges / possible edges).
    """
    # 1. Correlation Matrix
    corr = returns_matrix.corr()
    
    # 2. Graph Construction
    G = nx.Graph()
    sectors = corr.columns
    G.add_nodes_from(sectors)
    
    for i in range(len(sectors)):
        for j in range(i + 1, len(sectors)):
            weight = corr.iloc[i, j]
            # We look at highly correlated pairs. 
            if weight > threshold:
                G.add_edge(sectors[i], sectors[j], weight=weight)
                
    # 3. Graph Density (The Fragility Score)
    # Density is 0 if no edges exist, and 1 if it's a complete graph.
    fragility = nx.density(G)
    
    logger.info(f"Market Graph Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    logger.info(f"Systemic Fragility Score (Density): {fragility:.4f}")
    
    return float(fragility)


def main():
    logger.info("=" * 50)
    logger.info("PHASE 6 — Graph-Based Market Structure")
    logger.info("=" * 50)
    
    features_dir = cfg.DATA_DIR / "features"
    returns_matrix = build_returns_matrix(features_dir, window=30)
    
    if returns_matrix.empty:
        logger.error("No sector data found.")
        sys.exit(1)
        
    fragility = calculate_fragility(returns_matrix, threshold=0.7)
    
    out_path = features_dir / "systemic_fragility.json"
    with open(out_path, "w") as f:
        json.dump({"systemic_fragility": fragility}, f)
        
    logger.info(f"Saved fragility score to {out_path.name}")
    logger.info("Phase 6 complete. ✓")

if __name__ == "__main__":
    main()
