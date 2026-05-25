import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from typing import List, Tuple, Any

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from validation_harness import CombinatorialPurgedCV, WalkForwardTemporalSplitter


# ==========================================
# 1. Advanced XGBoost: Training & Testing
# ==========================================

def train_xgboost_cv(X: pd.DataFrame, y: pd.Series, groups: np.ndarray = None) -> Tuple[xgb.XGBRanker, float]:
    """
    Trains an XGBoost Ranker model using Combinatorial Purged Cross-Validation (CPCV).
    Evaluates out-of-sample predictions using the Information Coefficient (Spearman Rank Correlation).
    
    Args:
        X: Chronological dataframe of sector features.
        y: Chronological series of sector forward returns (targets to rank).
        groups: Array indicating the grouping (e.g., date index or step) for ranking pairwise.
    
    Returns:
        trained_model: The XGBoost model trained on the full set (or the last fold).
        mean_ic: The average Information Coefficient across all out-of-sample folds.
    """
    # Initialize XGBRanker with ndcg objective for ranking.
    # rank:ndcg optimizes Normalized Discounted Cumulative Gain, perfect for relative sector performance.
    model = xgb.XGBRanker(
        objective='rank:ndcg',
        learning_rate=0.05,
        n_estimators=100,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # CPCV with 6 blocks, 2 test blocks, and a 5-day embargo gap
    cpcv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, embargo_td=5)
    
    out_of_sample_ics = []
    
    for fold, (train_idx, test_idx) in enumerate(cpcv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # XGBRanker requires 'qid' (query ID) which represents the grouping (e.g., the specific day)
        # If groups is not provided, we simulate a dummy grouping (assuming 1 group per day, etc.)
        qid_train = groups[train_idx] if groups is not None else np.zeros(len(train_idx))
        qid_test = groups[test_idx] if groups is not None else np.zeros(len(test_idx))
        
        # Sort by qid as required by XGBRanker
        train_sort_idx = np.argsort(qid_train)
        X_train, y_train, qid_train = X_train.iloc[train_sort_idx], y_train.iloc[train_sort_idx], qid_train[train_sort_idx]
        
        model.fit(
            X_train, y_train,
            qid=qid_train,
            verbose=False
        )
        
        # Predict on out-of-sample
        preds = model.predict(X_test)
        
        # Calculate Information Coefficient (Spearman Rank Correlation)
        ic, _ = spearmanr(preds, y_test)
        if not np.isnan(ic):
            out_of_sample_ics.append(ic)
            
    mean_ic = np.mean(out_of_sample_ics) if out_of_sample_ics else 0.0
    print(f"Mean Out-of-Sample Information Coefficient (IC): {mean_ic:.4f}")
    
    return model, mean_ic

# ==========================================
# 2. Graph Neural Network: Training & Testing
# ==========================================

class SectorGAT(torch.nn.Module):
    """
    Graph Attention Network (GAT) for Sector Modeling.
    
    Maps 11 sectors as nodes and uses Partial Correlation for edges.
    GAT is used instead of GCN to dynamically weight which sector 
    relationships matter most in the current market regime.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4):
        super(SectorGAT, self).__init__()
        # First GAT layer with multi-head attention and edge attributes enabled
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2, edge_dim=1)
        # Second GAT layer consolidating the heads to the output dimension
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.2, edge_dim=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None, return_attention_weights: bool = False):
        """
        Forward pass.
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge weights/features (e.g. partial correlations) [num_edges, num_edge_features]
            return_attention_weights: If True, returns (out, (edge_index, alpha))
        """
        x = F.dropout(x, p=0.2, training=self.training)
        # Using edge attributes if supported by specific layer configuration, otherwise standard GAT topology
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        if return_attention_weights:
            x, (edge_idx_out, alpha) = self.conv2(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            return x, (edge_idx_out, alpha)
        else:
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
            return x

def train_gnn_walkforward(graphs: List[Data]) -> SectorGAT:
    """
    Trains the GAT to predict next day's node states using a Walk-Forward Temporal Splitter.
    
    Args:
        graphs: Chronological list of PyTorch Geometric `Data` objects representing the market daily.
                Each graph must have `x` (features), `edge_index` (topology), and `y` (target node states).
    
    Returns:
        trained_model: The GAT model after completing the walk-forward training sequence.
    """
    # Assuming uniform feature size across all graphs
    in_channels = graphs[0].x.size(1)
    out_channels = graphs[0].y.size(1) if graphs[0].y.dim() > 1 else 1
    
    model = SectorGAT(in_channels=in_channels, hidden_channels=16, out_channels=out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    
    # 60 days train window, 10 days val window
    splitter = WalkForwardTemporalSplitter(train_window_size=60, val_window_size=10, step_size=10)
    
    for window_idx, (train_graphs, val_graphs) in enumerate(splitter.split(graphs)):
        model.train()
        
        # Train on the current train window
        for epoch in range(10): # 10 epochs per walk-forward step
            optimizer.zero_grad()
            total_loss = 0
            for data in train_graphs:
                out = model(data.x, data.edge_index, data.edge_attr)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        # Validate (Optional tracking)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_graphs:
                out = model(data.x, data.edge_index, data.edge_attr)
                val_loss += criterion(out, data.y).item()
                
        print(f"Walk-Forward Window {window_idx+1}: Val Loss = {val_loss / len(val_graphs):.4f}")
        
    return model

def stress_test_gnn(trained_model: SectorGAT, current_graph: Data) -> None:
    """
    Stress tests the GNN to evaluate if it understands market structure.
    
    Index Mapping as per `config.py`:
    ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
    - Index 0: XLK (Technology)
    - Index 1: XLF (Financials)
    - Index 7: XLU (Utilities)
    
    Test: Artificially shock the input features of XLK (Technology) by -5% and 
    measure the propagated impact on XLU and XLF compared to an unshocked baseline.
    """
    trained_model.eval()
    
    # Define exact indices from mapping
    IDX_XLK = 0
    IDX_XLF = 1
    IDX_XLU = 7
    
    with torch.no_grad():
        # Baseline Pass (No Shock) with attention extraction
        baseline_out, (edge_idx_out, alpha) = trained_model(current_graph.x, current_graph.edge_index, current_graph.edge_attr, return_attention_weights=True)
        
        # Extract attention weights from XLK (source) to XLF (target) and XLU (target)
        # edge_idx_out[0] is source j, edge_idx_out[1] is target i in PyG GATConv
        xlk_to_xlf_mask = (edge_idx_out[0] == IDX_XLK) & (edge_idx_out[1] == IDX_XLF)
        xlk_to_xlu_mask = (edge_idx_out[0] == IDX_XLK) & (edge_idx_out[1] == IDX_XLU)
        
        # alpha is shape [num_edges, heads], we can mean across heads
        attn_xlk_to_xlf = alpha[xlk_to_xlf_mask].mean().item() if xlk_to_xlf_mask.any() else 0.0
        attn_xlk_to_xlu = alpha[xlk_to_xlu_mask].mean().item() if xlk_to_xlu_mask.any() else 0.0
        
        # Shock Pass (Shock XLK by -5%)
        shocked_x = current_graph.x.clone()
        # Apply -0.05 to all features of XLK
        shocked_x[IDX_XLK] = shocked_x[IDX_XLK] - 0.05
        
        shocked_out = trained_model(shocked_x, current_graph.edge_index, current_graph.edge_attr)
        
        # Calculate Propagation Impact
        baseline_xlf = baseline_out[IDX_XLF].mean().item()
        shocked_xlf = shocked_out[IDX_XLF].mean().item()
        impact_xlf = shocked_xlf - baseline_xlf
        
        baseline_xlu = baseline_out[IDX_XLU].mean().item()
        shocked_xlu = shocked_out[IDX_XLU].mean().item()
        impact_xlu = shocked_xlu - baseline_xlu
        
        print("=== GNN Stress Test: -5% Shock to XLK (Technology) ===")
        print(f"Learned Attention Weight (XLK -> XLF): {attn_xlk_to_xlf:.8f}")
        print(f"Learned Attention Weight (XLK -> XLU): {attn_xlk_to_xlu:.8f}")
        print(f"Propagated Impact on XLF (Financials):  {impact_xlf:+.4f} (Baseline: {baseline_xlf:.4f} -> Shocked: {shocked_xlf:.4f})")
        print(f"Propagated Impact on XLU (Utilities):   {impact_xlu:+.4f} (Baseline: {baseline_xlu:.4f} -> Shocked: {shocked_xlu:.4f})")
        print("======================================================")
