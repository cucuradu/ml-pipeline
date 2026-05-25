import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
import optuna

# Add the project root to sys.path so we can import modules
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from research.validation_harness import CombinatorialPurgedCV
from research.run_advanced_tests import engineer_features

def load_and_prep_data():
    file_path = os.path.join(base_dir, 'data', 'features', 'SPY_features.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please run data pipeline first.")
        
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'ticker_id']).reset_index(drop=True)
    
    # Engineer the 7 stationary features
    df = engineer_features(df)
    
    # Target discretization (cross-sectional daily ranking)
    df['y'] = df.groupby('date')['forward_5d_return'].transform(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop') if len(x.dropna()) >= 5 else np.zeros(len(x))
    )
    y = df['y'].fillna(0).astype(int)
    
    # Groups for qid
    groups = df.groupby('date').ngroup().values
    
    # Feature columns
    requested_features = [
        'log_return', 'Yield_Curve_10Y2Y', 'High_Yield_Spread', 
        'rv_21d', 'sma_dist_20', 'sma_dist_50', 'relative_strength_21d', 'ticker_id'
    ]
    feature_cols = [col for col in requested_features if col in df.columns]
    X = df[feature_cols].fillna(0)
    
    return X, y, groups

def objective(trial, X, y, groups):
    # Hyperparameters heavily focused on regularization
    params = {
        'objective': 'rank:ndcg',
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True), # L1 penalty
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True), # L2 penalty
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': 0.8 # Fixed subsample for speed/robustness
    }
    
    model = xgb.XGBRanker(**params)
    cpcv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, embargo_td=5)
    
    out_of_sample_ics = []
    
    for train_idx, test_idx in cpcv.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        qid_train = groups[train_idx]
        qid_test = groups[test_idx]
        
        # Sort by qid
        train_sort_idx = np.argsort(qid_train)
        X_train, y_train, qid_train = X_train.iloc[train_sort_idx], y_train.iloc[train_sort_idx], qid_train[train_sort_idx]
        
        model.fit(X_train, y_train, qid=qid_train, verbose=False)
        
        preds = model.predict(X_test)
        
        # Calculate IC
        ic, _ = spearmanr(preds, y_test)
        if not np.isnan(ic):
            out_of_sample_ics.append(ic)
            
    return np.mean(out_of_sample_ics) if out_of_sample_ics else 0.0

def run_tuner():
    print("=========================================")
    print("XGBoost Institutional Hyperparameter Tuner")
    print("=========================================")
    
    X, y, groups = load_and_prep_data()
    print(f"Loaded X with shape {X.shape} and {len(np.unique(groups))} daily groups.")
    print("Features:", list(X.columns))
    
    # Create Optuna study to maximize Information Coefficient
    study = optuna.create_study(direction='maximize', study_name='xgboost_ranker_tune')
    
    print("\nStarting 50 trials...")
    study.optimize(lambda trial: objective(trial, X, y, groups), n_trials=50)
    
    print("\n=========================================")
    print("Tuning Complete!")
    print(f"Best Mean Out-of-Sample IC: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    run_tuner()
