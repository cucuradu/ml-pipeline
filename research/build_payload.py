import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

# Add the project root to sys.path so we can import modules
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from research.run_advanced_tests import engineer_features
from config import SECTOR_TICKERS

def build_production_payload():
    print("=========================================")
    print("Building Production Payload (Phase 4)")
    print("=========================================")
    
    file_path = os.path.join(base_dir, 'data', 'features', 'SPY_features.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please run data pipeline first.")
        
    print("Loading data...")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'ticker_id']).reset_index(drop=True)
    
    # Engineer features
    df = engineer_features(df)
    
    # Define groups for qid
    groups = df.groupby('date').ngroup().values
    
    # Discretize Target
    df['y'] = df.groupby('date')['forward_5d_return'].transform(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop') if len(x.dropna()) >= 5 else np.zeros(len(x))
    )
    y = df['y'].fillna(0).astype(int)
    
    # Feature columns map
    requested_features = [
        'log_return', 'Yield_Curve_10Y2Y', 'High_Yield_Spread', 
        'rv_21d', 'sma_dist_20', 'sma_dist_50', 'relative_strength_21d', 'ticker_id'
    ]
    feature_cols = [col for col in requested_features if col in df.columns]
    X = df[feature_cols].fillna(0)
    
    # 1. Train Production Model
    print("Training XGBRanker Production Model on FULL dataset...")
    model = xgb.XGBRanker(
        objective='rank:ndcg',
        max_depth=6,
        learning_rate=0.0123,
        reg_alpha=7.867,
        reg_lambda=0.443,
        colsample_bytree=0.574,
        n_estimators=100,
        subsample=0.8,
        random_state=42
    )
    
    # Sort entire dataset by qid for training
    train_sort_idx = np.argsort(groups)
    X_train = X.iloc[train_sort_idx]
    y_train = y.iloc[train_sort_idx]
    qid_train = groups[train_sort_idx]
    
    model.fit(X_train, y_train, qid=qid_train, verbose=False)
    
    # 2. Predict Tomorrow's Ranks
    print("Generating predictions for the most recent date cross-section...")
    most_recent_date = df['date'].max()
    recent_df = df[df['date'] == most_recent_date].copy()
    
    X_recent = recent_df[feature_cols].fillna(0)
    predictions = model.predict(X_recent)
    
    # Map predictions back to tickers
    recent_df['score'] = predictions
    ticker_scores = dict(zip(recent_df['ticker'], recent_df['score']))
    
    # Selection: Top 3 by Momentum (XGBoost Score)
    top3_df = recent_df.sort_values('score', ascending=False).head(3)
    top3_tickers = top3_df['ticker'].tolist()
    
    # Sizing: Inverse Volatility (Risk-Adjusted)
    # Prevent division by zero using a small epsilon
    rv_values = np.maximum(top3_df['rv_21d'].values, 1e-4)
    inv_vol = 1.0 / rv_values
    weights = inv_vol / inv_vol.sum()
    universe_weights = {ticker: round(float(w), 4) for ticker, w in zip(top3_tickers, weights)}
    
    print(f"Top 3 Selection: {top3_tickers}")
    print(f"Risk-Adjusted Weights: {universe_weights}")
    
    # 3. Update the JSON Payload
    json_path = os.path.join(base_dir, 'output', 'ml-insights.json')
    print(f"Updating JSON Payload at {json_path}...")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            payload = json.load(f)
    else:
        # Create skeleton if it doesn't exist
        payload = {"sectors": {}}
        
    # Overwrite top-level allocations
    payload["universe_ranking"] = top3_tickers
    payload["universe_weights"] = universe_weights
        
    # Ensure sectors dictionary exists
    if "sectors" not in payload:
        payload["sectors"] = {}
        
    for ticker in SECTOR_TICKERS:
        if ticker not in payload["sectors"]:
            payload["sectors"][ticker] = {}
        
        # Safely assign the newly predicted score, defaulting to 0.0 if not found
        pred_val = float(ticker_scores.get(ticker, 0.0))
        payload["sectors"][ticker]["score"] = pred_val
        
        # Optional: Also update the old key just in case legacy systems rely on it
        payload["sectors"][ticker]["xgb_fwd_return_pred"] = pred_val
        
    # Update timestamp
    payload["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
    
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
        
    print(f"Success! Model trained up to {most_recent_date.date()} and JSON secured.")

if __name__ == "__main__":
    build_production_payload()
