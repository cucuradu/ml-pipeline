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
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
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
    
    # Sort chronologically for TimeSeriesSplit
    full_df = pd.concat(all_data).sort_index()
    
    # Load model params from the trained PKL to ensure we evaluate the Optuna-tuned model
    import pickle
    with open(cfg.MODELS_DIR / "xgboost_ranking.pkl", "rb") as f:
        model_data = pickle.load(f)
        
    best_params = model_data.get("params", {
        "n_estimators": 100, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": -1
    })
    
    model = xgb.XGBRegressor(**best_params)
    
    # We will use the same features from config or just exclude metadata
    feature_cols = [c for c in full_df.columns if c not in ("ticker", "close", "actual_fwd_5d")]
    
    X = full_df[feature_cols]
    y = full_df["actual_fwd_5d"]
    
    # Purged CV (Embargo): gap=5 ensures 0 overlap between train target (T to T+5) and test features (T+6)
    # 70/30 Chronological Split with 5-day Embargo Gap
    dates = full_df.index.unique().sort_values()
    split_idx = int(len(dates) * 0.70)
    
    train_end_date = dates[split_idx]
    # +5 days to prevent overlapping return leakage
    test_start_date = dates[min(split_idx + 5, len(dates) - 1)] 
    
    X_train = full_df.loc[full_df.index <= train_end_date][feature_cols]
    y_train = full_df.loc[full_df.index <= train_end_date]["actual_fwd_5d"]
    
    X_test = full_df.loc[full_df.index >= test_start_date][feature_cols]
    y_test = full_df.loc[full_df.index >= test_start_date]["actual_fwd_5d"]
    
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_test)
        
    oof_df = full_df.loc[full_df.index >= test_start_date].copy()
    oof_df["pred_fwd_5d"] = preds
    
    # 1. Rank Correlation (Spearman) on Out-Of-Sample data
    corr, p_value = spearmanr(oof_df["pred_fwd_5d"], oof_df["actual_fwd_5d"])
    print(f"Spearman Rank Correlation: {corr:.4f} (p-value: {p_value:.4e})")
    print("  * Higher is better. A positive correlation means the model correctly")
    print("    ranks which sectors will outperform relative to others.")
    
    
    # 2. Top-1 Hit Rate
    top1_hit = 0
    total_days = 0
    
    for date, group in oof_df.groupby("date"):
        if len(group) < 11: continue # Only evaluate days we have all sectors
        top_pick = group.iloc[group["pred_fwd_5d"].argmax()]
        if top_pick["actual_fwd_5d"] > 0:
            top1_hit += 1
        total_days += 1
    hit_rate = top1_hit / total_days if total_days > 0 else 0
    
    print(f"\nTop-1 Sector Long-Only Hit Rate: {hit_rate:.2%} ({top1_hit}/{total_days} days)")
    
    # 3. Benchmark (Base Rate & SPY)
    base_rate = (oof_df["actual_fwd_5d"] > 0).mean()
    
    # Get SPY performance for the exact same OOS dates
    spy = pd.read_csv(cfg.DATA_DIR / "features" / "SPY_features.csv", index_col="date", parse_dates=True)
    spy["actual_fwd_5d"] = (spy["close"].shift(-5) / spy["close"]) - 1
    
    # We only care about the unique dates in the OOS test set
    oos_dates = oof_df.index.unique()
    spy_oos = spy.loc[oos_dates].dropna()
    spy_hit_rate = (spy_oos["actual_fwd_5d"] > 0).mean()
    
    print(f"\nBenchmarks (Random Sector vs SPY vs Model):")
    print(f"  Random Sector Hit Rate: {base_rate:.2%}")
    print(f"  SPY 5-Day Hit Rate:     {spy_hit_rate:.2%}")
    print(f"  --> MODEL EDGE:         {hit_rate - spy_hit_rate:+.2%} over SPY")
    
    return {
        "spearman_correlation": float(corr),
        "top_1_hit_rate_pct": float(hit_rate * 100),
        "benchmark_spy_hit_rate_pct": float(spy_hit_rate * 100),
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
