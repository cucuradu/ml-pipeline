import numpy as np
import pandas as pd
from typing import List, Tuple, Iterable, Any
from itertools import combinations

class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV)
    
    Based on Marcos Lopez de Prado's methodology.
    Splits the time series into `n_splits` chronological blocks.
    In each iteration, `n_test_splits` blocks are selected as the test set.
    The remaining blocks constitute the train set, but we apply an embargo
    gap (a specified number of samples) immediately following any test block
    to prevent serial correlation (data leakage) from flowing into the train set.
    """
    
    def __init__(self, n_splits: int = 6, n_test_splits: int = 2, embargo_td: int = 5):
        """
        Args:
            n_splits: Total number of chronological blocks.
            n_test_splits: Number of blocks to use for testing in each combination.
            embargo_td: Number of samples to drop from the start of a training block 
                        that immediately follows a test block.
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_td = embargo_td
        
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: np.ndarray = None) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Yields train and test indices for each CPCV fold.
        
        Args:
            X: Dataframe of features, assumed to be chronologically ordered.
        Yields:
            train_indices, test_indices
        """
        num_samples = len(X)
        block_size = num_samples // self.n_splits
        
        # Create boundaries for the N blocks
        blocks = []
        for i in range(self.n_splits):
            start = i * block_size
            end = (i + 1) * block_size if i < self.n_splits - 1 else num_samples
            blocks.append((start, end))
            
        # Combinations of test blocks
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))
        
        for test_block_indices in test_combinations:
            test_indices = []
            train_indices = []
            
            # Identify which blocks are test blocks
            is_test = [i in test_block_indices for i in range(self.n_splits)]
            
            for i in range(self.n_splits):
                if is_test[i]:
                    test_indices.extend(range(blocks[i][0], blocks[i][1]))
                else:
                    start_idx = blocks[i][0]
                    end_idx = blocks[i][1]
                    
                    # If the PREVIOUS block was a test block, apply embargo gap
                    if i > 0 and is_test[i-1]:
                        start_idx = min(start_idx + self.embargo_td, end_idx)
                    
                    # If we purged too much, we just don't add to train indices
                    if start_idx < end_idx:
                        train_indices.extend(range(start_idx, end_idx))
                        
            yield np.array(train_indices), np.array(test_indices)

class WalkForwardTemporalSplitter:
    """
    Walk-Forward Temporal Splitter for Graph Neural Networks.
    
    Generates a sliding window over a chronological sequence of graphs.
    Yields (TrainWindow, ValidationWindow) tuples ensuring that the 
    validation graph topology completely occurs after the training window.
    """
    
    def __init__(self, train_window_size: int, val_window_size: int = 1, step_size: int = 1):
        """
        Args:
            train_window_size: Number of chronological steps to include in training graph sequence.
            val_window_size: Number of chronological steps to include in validation graph sequence.
            step_size: How many steps to slide the window forward each iteration.
        """
        self.train_window_size = train_window_size
        self.val_window_size = val_window_size
        self.step_size = step_size
        
    def split(self, graphs: List[Any]) -> Iterable[Tuple[List[Any], List[Any]]]:
        """
        Splits sequence of graphs chronologically.
        
        Args:
            graphs: List of PyTorch Geometric Data objects or similar, ordered chronologically.
        Yields:
            train_graphs, val_graphs
        """
        n_graphs = len(graphs)
        start_idx = 0
        
        while start_idx + self.train_window_size + self.val_window_size <= n_graphs:
            train_end = start_idx + self.train_window_size
            val_end = train_end + self.val_window_size
            
            train_graphs = graphs[start_idx:train_end]
            val_graphs = graphs[train_end:val_end]
            
            yield train_graphs, val_graphs
            
            start_idx += self.step_size
