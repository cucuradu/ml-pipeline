#!/usr/bin/env python3
"""
Phase 4 — Assembly & Automation Hook
====================================
Runs the full pipeline, loads all models and features, and generates
the final `ml-insights.json` deliverable.
"""

import sys
import json
import subprocess
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

def run_script(script_name: str) -> None:
    print(f"\n>>> Running {script_name}...")
    subprocess.run([sys.executable, script_name], check=True)

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def generate_json() -> None:
    # 1. Market Regime & Confidence
    features_dir = cfg.DATA_DIR / "features"
    spy_feat = pd.read_csv(features_dir / "SPY_features.csv", index_col="date", parse_dates=True)
    
    # Get last feature row
    last_row = spy_feat.iloc[-1:]
    feature_cols = [c for c in spy_feat.columns if c not in ("ticker", "close", "regime_raw", "regime")]
    X_latest = last_row[feature_cols].values
    
    # Load GMM
    with open(cfg.MODELS_DIR / "gmm_pipeline.pkl", "rb") as f:
        gmm_data = pickle.load(f)
    pipeline = gmm_data["pipeline"]
    mapping = gmm_data["mapping"]
    
    # Predict
    proba = pipeline.predict_proba(X_latest)[0]
    cluster = pipeline.predict(X_latest)[0]
    raw_regime = mapping[cluster]
    confidence = float(np.max(proba))
    
    # The actual filtered regime is in SPY_regimes.csv
    spy_regimes = pd.read_csv(features_dir / "SPY_regimes.csv", index_col="date", parse_dates=True)
    final_regime = spy_regimes["regime"].iloc[-1]
    
    # If the filtered regime differs from the raw prediction (due to persistence filter),
    # the confidence represents our confidence in the *raw* prediction, but we output the filtered regime.
    # To be consistent, let's just output the confidence of the final_regime.
    # Wait, mapping maps cluster IDs to regime names. Let's find the cluster ID of the final_regime.
    inv_map = {v: k for k, v in mapping.items()}
    final_cluster = inv_map[final_regime]
    final_confidence = float(proba[final_cluster])

    # 2. Volatility
    with open(features_dir / "garch_forecast.json", "r") as f:
        garch_data = json.load(f)
    garch_forecast = garch_data["garch_1d_forecast_pct"]
    
    # VIX Term Structure
    vix_raw = pd.read_csv(cfg.DATA_DIR / "raw" / "VIX_raw.csv", index_col="date", parse_dates=True)
    vix3m_raw = pd.read_csv(cfg.DATA_DIR / "raw" / "VIX3M_raw.csv", index_col="date", parse_dates=True)
    
    vix_latest = vix_raw["close"].iloc[-1]
    vix3m_latest = vix3m_raw["close"].iloc[-1]
    
    term_structure = "contango" if vix3m_latest >= vix_latest else "backwardation"
    
    # 3. Sectors
    sectors_dict = {}
    for sector in cfg.SECTOR_TICKERS:
        sec_feat = pd.read_csv(features_dir / f"{sector}_features.csv", index_col="date", parse_dates=True)
        sec_last = sec_feat.iloc[-1]
        
        sma50 = sec_last["sma_dist_50"]
        sma200 = sec_last["sma_dist_200"]
        
        # Sector Regime Heuristic
        if sma50 > 0 and sma200 > 0:
            sec_regime = "Trend"
        elif sma50 < 0 and sma200 < 0:
            sec_regime = "Bear"
        else:
            sec_regime = "Choppy"
            
        # Score heuristic (0 to 1)
        # Using a sigmoid on the average SMA distance * 10 
        score = float(sigmoid((sma50 + sma200) * 10))
        
        sectors_dict[sector] = {
            "regime": sec_regime,
            "score": round(score, 4)
        }
        
    # Build JSON
    deployment_info = cfg.REGIME_DEPLOYMENT_MAP[final_regime]
    
    insights = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": cfg.MODEL_VERSION,
        "market": {
            "regime": final_regime,
            "confidence": round(final_confidence, 4),
            "persistence_bars": cfg.REGIME_PERSISTENCE_BARS,
            "deployment_target": deployment_info["deployment_target"],
            "trade_slots": deployment_info["trade_slots"]
        },
        "volatility": {
            "garch_1d_forecast_pct": round(garch_forecast, 4),
            "vix_implied_term_structure": term_structure
        },
        "sectors": sectors_dict
    }
    
    # Save
    with open(cfg.INSIGHTS_PATH, "w") as f:
        json.dump(insights, f, indent=2)
        
    print(f"\n>>> Successfully generated {cfg.INSIGHTS_PATH.name}")
    print(json.dumps(insights, indent=2))

def main():
    print("=" * 50)
    print("PHASE 4 — Pipeline Assembly")
    print("=" * 50)
    
    run_script("phase1_data_pipeline.py")
    run_script("phase2_regime_model.py")
    run_script("phase3_volatility_model.py")
    
    print("\n>>> Generating Insights JSON...")
    generate_json()
    
    print("\nAll phases complete. ✓")

if __name__ == "__main__":
    main()
