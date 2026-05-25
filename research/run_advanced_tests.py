import os
import sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from advanced_models import train_xgboost_cv, train_gnn_walkforward, SectorGAT, stress_test_gnn

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers the required Phase 1 stationary features on the fly if missing."""
    # Ensure log_return exists
    if 'log_return' not in df.columns and 'sector_close' in df.columns:
        df['log_return'] = df.groupby('ticker')['sector_close'].transform(lambda x: np.log(x / x.shift(1)))
        
    # Realized Volatility (21-day)
    if 'rv_21d' not in df.columns and 'log_return' in df.columns:
        df['rv_21d'] = df.groupby('ticker')['log_return'].transform(lambda x: x.rolling(21).std() * np.sqrt(252))
        
    # SMA Distances
    if 'sma_dist_20' not in df.columns and 'sector_close' in df.columns:
        df['sma_dist_20'] = df.groupby('ticker')['sector_close'].transform(lambda x: (x / x.rolling(20).mean()) - 1)
        
    if 'sma_dist_50' not in df.columns and 'sector_close' in df.columns:
        df['sma_dist_50'] = df.groupby('ticker')['sector_close'].transform(lambda x: (x / x.rolling(50).mean()) - 1)
        
    # Relative Strength (21-day) against SPY
    if 'relative_strength_21d' not in df.columns and 'sector_close' in df.columns and 'SPY_Close' in df.columns:
        # Calculate 21-day returns for sector and SPY
        sector_ret_21 = df.groupby('ticker')['sector_close'].transform(lambda x: x.pct_change(21))
        spy_ret_21 = df.groupby('ticker')['SPY_Close'].transform(lambda x: x.pct_change(21))
        df['relative_strength_21d'] = sector_ret_21 - spy_ret_21

    return df

def run_xgboost_tests():
    print("=========================================")
    print("1. XGBoost Sector Ranking Test")
    print("=========================================")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(base_dir, 'data', 'features', 'SPY_features.csv')
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Cannot run real test.")
        return
        
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Sort strictly by date and ticker
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'ticker_id']).reset_index(drop=True)
    
    # Engineer missing features
    df = engineer_features(df)
    
    # Define groups for qid (one group per unique date)
    groups = df.groupby('date').ngroup().values
    
    # 2. Target Discretization (Daily Cross-Section)
    print("Converting continuous targets to daily cross-sectional quintiles (0-4)...")
    df['y'] = df.groupby('date')['forward_5d_return'].transform(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop') if len(x.dropna()) >= 5 else np.zeros(len(x))
    )
    y = df['y'].fillna(0).astype(int)
    
    # 3. Features Map - expanding to requested technicals
    requested_features = [
        'log_return', 'Yield_Curve_10Y2Y', 'High_Yield_Spread', 
        'rv_21d', 'sma_dist_20', 'sma_dist_50', 'volume_trend', 
        'relative_strength_21d', 'ticker_id'
    ]
    feature_cols = [col for col in requested_features if col in df.columns]
    
    print(f"Using features: {feature_cols}")
    X = df[feature_cols].fillna(0) # naive fill for any initial rolling NaNs
    
    print("Training XGBoost Ranker with Combinatorial Purged CV...")
    model, mean_ic = train_xgboost_cv(X, y, groups=groups)
    
    print(f"\n[XGBoost Result] Final Mean Information Coefficient (IC): {mean_ic:.4f}")
    print("=========================================\n")


def prep_real_graphs_from_csv(df: pd.DataFrame) -> list:
    """
    Converts real SPY_features.csv data into chronological torch_geometric Data objects.
    Draws FULLY CONNECTED graphs using 30-day rolling correlation as edge weights.
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    # Engineer missing features
    df = engineer_features(df)
        
    requested_features = [
        'log_return', 'Yield_Curve_10Y2Y', 'High_Yield_Spread', 
        'rv_21d', 'sma_dist_20', 'sma_dist_50', 'volume_trend', 
        'relative_strength_21d', 'ticker_id'
    ]
    feature_cols = [col for col in requested_features if col in df.columns]
    
    # Pivot log returns to easily calculate rolling correlations
    wide_returns = df.pivot(index='date', columns='ticker', values='log_return')
    unique_dates = wide_returns.index
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if base_dir not in sys.path:
        sys.path.append(base_dir)
    from config import SECTOR_TICKERS
    
    # We must enforce a consistent order for nodes matching config.py exactly
    sectors = SECTOR_TICKERS
    num_sectors = len(sectors)
    
    graphs = []
    
    # We need at least 30 days of history for the first correlation matrix
    print("Building fully connected dynamic graphs with 30-day rolling correlation edge weights...")
    for i in range(30, len(unique_dates)):
        current_date = unique_dates[i]
        
        # 1. Node Features & Targets
        day_data = df[df['date'] == current_date].set_index('ticker').reindex(sectors)
        
        # Fill missing with 0 temporarily
        x_tensor = torch.tensor(day_data[feature_cols].fillna(0).values, dtype=torch.float)
        
        # Target: forward_5d_return
        y_tensor = torch.tensor(day_data['forward_5d_return'].fillna(0).values, dtype=torch.float).unsqueeze(1)
        
        # 2. Edges (30-day rolling correlation)
        # Slicing the last 30 days up to current_date
        window_returns = wide_returns.iloc[i-30:i]
        corr_matrix = window_returns.corr().reindex(index=sectors, columns=sectors).values
        
        edge_index = []
        edge_attr = []
        
        # Create a fully connected graph topology
        for r in range(num_sectors):
            for c in range(num_sectors):
                if r != c:
                    corr_val = corr_matrix[r, c]
                    # Handle NaNs dynamically by zeroing out the correlation weight
                    if np.isnan(corr_val):
                        corr_val = 0.0
                        
                    edge_index.append([r, c])
                    edge_attr.append([corr_val])
                        
        if len(edge_index) > 0:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
        else:
            # Fallback if no sectors
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, 1), dtype=torch.float)
            
        graph = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=y_tensor)
        graphs.append(graph)
        
    return graphs


def run_gnn_tests():
    print("=========================================")
    print("2. GNN Execution & Stress Test (Real Data)")
    print("=========================================")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(base_dir, 'data', 'features', 'SPY_features.csv')
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Cannot run real test.")
        return
        
    df = pd.read_csv(file_path)
    graphs = prep_real_graphs_from_csv(df)
    
    print(f"Generated {len(graphs)} real chronological graphs.")
    
    print("\nTraining SectorGAT Walk-Forward Temporal Splitter...")
    trained_gnn_model = train_gnn_walkforward(graphs)
    
    print("\nExecuting Stress Test (-5% Shock to Tech / XLK) on Most Recent Graph...")
    current_graph = graphs[-1]
    
    # Exact Node Indices:
    # ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
    stress_test_gnn(trained_gnn_model, current_graph)


if __name__ == "__main__":
    run_xgboost_tests()
    run_gnn_tests()
