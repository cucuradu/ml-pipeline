#!/usr/bin/env python3
"""
Model Evaluation & Backtesting Script
=====================================
Evaluates the real-world predictive power of the ML models on historical data.

1. XGBoost: Calculates Rank Correlation & Hit Rate for the Top 1 Sector.
2. HMM: Compares the forward returns of SPY across different predicted regimes.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr
import config as cfg

def evaluate_xgboost():
    print("\n" + "="*50)
    print("1. XGBoost Ranking Engine Evaluation")
    print("="*50)
    
    # Load all historical features (we need the actual 5-day forward returns)
    features_dir = cfg.DATA_DIR / "features"
    all_data = []
    
    for sector in cfg.SECTOR_TICKERS:
        csv_path = features_dir / f"{sector}_features.csv"
        if not csv_path.exists(): continue
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        # Actual future return
        df["actual_fwd_5d"] = (df["close"].shift(-5) / df["close"]) - 1
        df["ticker"] = sector
        all_data.append(df.dropna())
        
    if not all_data: return
    
    full_df = pd.concat(all_data)
    
    # Load model and predict
    import pickle
    with open(cfg.MODELS_DIR / "xgboost_ranking.pkl", "rb") as f:
        model_data = pickle.load(f)
        
    model = model_data["model"]
    feature_cols = model_data["features"]
    
    full_df["pred_fwd_5d"] = model.predict(full_df[feature_cols])
    
    # 1. Rank Correlation (Spearman)
    corr, p_value = spearmanr(full_df["pred_fwd_5d"], full_df["actual_fwd_5d"])
    print(f"Spearman Rank Correlation: {corr:.4f} (p-value: {p_value:.4e})")
    print("  * Higher is better. A positive correlation means the model correctly")
    print("    ranks which sectors will outperform relative to others.")
    
    
    # 2. Top-1 Hit Rate
    top1_hit = 0
    total_days = 0
    
    for date, group in full_df.groupby("date"):
        if len(group) < 11: continue # Only evaluate days we have all sectors
        top_pick = group.iloc[group["pred_fwd_5d"].argmax()]
        if top_pick["actual_fwd_5d"] > 0:
            top1_hit += 1
        total_days += 1
        
    hit_rate = top1_hit / total_days if total_days > 0 else 0
    
    print(f"\nTop-1 Sector Long-Only Hit Rate: {hit_rate:.2%} ({top1_hit}/{total_days} days)")
    
    return {
        "spearman_correlation": float(corr),
        "top_1_hit_rate_pct": float(hit_rate * 100),
        "total_eval_days": total_days
    }


def evaluate_hmm_regimes():
    print("\n" + "="*50)
    print("2. HMM Regime Classifier Evaluation")
    print("="*50)
    
    features_dir = cfg.DATA_DIR / "features"
    spy = pd.read_csv(features_dir / "SPY_regimes.csv", index_col="date", parse_dates=True)
    
    # Calculate SPY 5-day forward return
    spy["actual_fwd_5d"] = (spy["close"].shift(-5) / spy["close"]) - 1
    spy = spy.dropna()
    
    print("Average 5-Day Forward Return by Regime:")
    stats = spy.groupby("regime")["actual_fwd_5d"].agg(["mean", "std", "count"])
    stats["mean"] = stats["mean"] * 100
    stats["std"] = stats["std"] * 100
    
    # Formatting
    stats = stats.rename(columns={"mean": "Mean Return (%)", "std": "Volatility (%)", "count": "Days in Regime"})
    print(stats.round(2).to_string())
    
    # Format for JSON
    regime_stats = {}
    for regime in ["Bull", "Neutral", "Caution", "Defensive"]:
        if regime in stats.index:
            row = stats.loc[regime]
            regime_stats[regime] = {
                "mean_fwd_return_pct": float(row["Mean Return (%)"]),
                "volatility_pct": float(row["Volatility (%)"]),
                "days_in_regime": int(row["Days in Regime"])
            }
            
    # Also load the Transition Matrix from the GMM/HMM file
    import pickle
    tm_list = []
    try:
        with open(cfg.MODELS_DIR / "gmm_pipeline.pkl", "rb") as f:
            hmm_data = pickle.load(f)
            hmm = hmm_data["pipeline"]["hmm"]
            mapping = hmm_data["mapping"]
            tm = hmm.transmat_
            
            # Reorder matrix to Bull, Neutral, Caution, Defensive
            sorted_states = []
            for r in ["Bull", "Neutral", "Caution", "Defensive"]:
                for state, name in mapping.items():
                    if name == r:
                        sorted_states.append(state)
                        break
            
            tm_reordered = tm[np.ix_(sorted_states, sorted_states)]
            tm_list = tm_reordered.tolist()
    except Exception as e:
        print(f"Could not load Transition Matrix: {e}")

    return {
        "forward_returns": regime_stats,
        "transition_matrix": tm_list
    }

if __name__ == "__main__":
    xgb_metrics = evaluate_xgboost()
    hmm_metrics = evaluate_hmm_regimes()
    
    # Save the training metrics payload for the dashboard
    metrics_payload = {
        "xgboost_ranking": xgb_metrics,
        "hmm_regimes": hmm_metrics
    }
    
    out_path = cfg.DATA_DIR / "features" / "training_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
        
    print(f"\nSaved training metrics to {out_path.name}")
    print("Done.")
