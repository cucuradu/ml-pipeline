#!/usr/bin/env python3
"""
Phase 2 — Regime Classifier (HMM)
=================================
Trains a Hidden Markov Model (HMM) to classify the SPY market environment
into 4 regimes: Bull, Neutral, Caution, Defensive.

Key features:
  • HMM inherently models the transition matrix between states over time.
  • No look-ahead bias: Regime for day T is predicted using features from day T-1.
  • Heuristic mapping: Hidden states are automatically mapped to names based on their
    mean SMA distance (trend) and Realized Volatility.

Outputs:
  • Trained HMM model pipeline (saved to models/hmm_pipeline.pkl)
  • Regime predictions appended to SPY data.
"""

import logging
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

import config as cfg

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOGS_DIR / "phase2.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Load Data & Shift for No Look-Ahead Bias
# ============================================================
def load_and_prepare_data(features_dir: Path) -> pd.DataFrame:
    """
    Loads SPY features and shifts them by 1 day.
    """
    csv_path = features_dir / "SPY_features.csv"
    if not csv_path.exists():
        logger.error(f"Missing {csv_path}. Run Phase 1 first.")
        sys.exit(1)

    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    df = df.sort_index()

    # Macro-arm the regime model: merge FRED-derived credit/macro signals (Phase 7)
    # so risk-off is detected from credit + the curve, not price alone. Forward-fill
    # across FRED's publication gaps; skip silently if the overlay is unavailable.
    macro_path = features_dir / "macro_features.csv"
    if macro_path.exists():
        macro = pd.read_csv(macro_path, index_col=0, parse_dates=True).sort_index()
        for col in ("macro_risk", "hy_oas"):
            if col in macro.columns:
                df[col] = macro[col].reindex(df.index, method="ffill")
        logger.info("Merged macro overlay (macro_risk, hy_oas) into regime features.")
    else:
        logger.warning("macro_features.csv absent; regime model runs price-only.")

    feature_cols = [c for c in df.columns if c not in ("ticker", "close")]

    df_shifted = df.copy()
    df_shifted[feature_cols] = df[feature_cols].shift(1)
    df_shifted = df_shifted.dropna(subset=feature_cols)
    
    logger.info(f"Loaded SPY features and shifted by 1 day (T-1 -> T). Rows: {len(df_shifted)}")
    return df_shifted


# ============================================================
# 2. Train HMM
# ============================================================
def train_hmm(df: pd.DataFrame, feature_cols: list[str], n_components: int = 4):
    """
    Trains a StandardScaler + GaussianHMM.
    HMM requires sequential data, which fits our daily time series perfectly.
    """
    logger.info(f"Training HMM with {n_components} components on {len(feature_cols)} features...")
    
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # covariance_type="diag" (not "full") + min_covar regularization is the fix for
    # the confidence-pinned-at-1.0 bug (followup S1/P7): a full 11×11 covariance per
    # state over ~1200 samples over-fits, making one Gaussian emission so peaked that
    # the single-day posterior saturates to 1.0. Diagonal covariances keep emissions
    # broad enough that predict_proba returns calibrated posteriors (~0.6–0.95).
    hmm = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        min_covar=1e-2,
        n_iter=cfg.HMM_N_ITER,
        random_state=cfg.GMM_RANDOM_STATE,
    )

    hmm.fit(X_scaled)
    logger.info("HMM training complete.")

    return {"scaler": scaler, "hmm": hmm}


# ============================================================
# 3. Map Hidden States to Human-Readable Regimes
# ============================================================
def map_states_to_regimes(model_dict: dict, df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Automatically maps arbitrary HMM hidden states (0, 1, 2, 3) to 
    'Bull', 'Neutral', 'Caution', 'Defensive' based on state properties.
    """
    X = df[feature_cols].values
    X_scaled = model_dict["scaler"].transform(X)
    preds = model_dict["hmm"].predict(X_scaled)
    
    df_temp = df.copy()
    df_temp["state"] = preds

    # Calculate mean stats per state
    state_stats = df_temp.groupby("state")[["sma_dist_200", "rv_21d"]].mean()
    
    trend = state_stats["sma_dist_200"]
    vol = state_stats["rv_21d"]
    
    trend_norm = (trend - trend.mean()) / trend.std()
    vol_norm = (vol - vol.mean()) / vol.std()
    
    bullish_score = trend_norm - vol_norm
    
    # Sort states by bullish score descending
    sorted_states = bullish_score.sort_values(ascending=False).index.tolist()
    
    regime_names = ["Bull", "Neutral", "Caution", "Defensive"]
    
    mapping = {}
    logger.info("Hidden State Mapping Statistics:")
    for i, state in enumerate(sorted_states):
        regime = regime_names[i]
        mapping[state] = regime
        stats = state_stats.loc[state]
        logger.info(f"  {regime} (State {state}) -> Trend (SMA200): {stats['sma_dist_200']:.4f}, Vol (RV21): {stats['rv_21d']:.4f}")

    return mapping


# ============================================================
# 4. Apply Persistence Filter
# ============================================================
def apply_persistence_filter(raw_regimes: pd.Series, persistence_bars: int) -> pd.Series:
    """
    Even with HMM's transition matrix, a persistence filter adds robustness 
    against 1-day whipsaws during regime boundaries.
    """
    raw_vals = raw_regimes.values.copy()
    filtered_vals = np.empty_like(raw_vals)
    current_regime = raw_vals[0]

    for i in range(len(raw_vals)):
        if i >= persistence_bars:
            window = raw_vals[i - persistence_bars + 1 : i + 1]
            if len(set(window)) == 1:
                current_regime = window[0]

        filtered_vals[i] = current_regime

    return pd.Series(filtered_vals, index=raw_regimes.index, name=raw_regimes.name)


# ============================================================
# Main
# ============================================================
def main() -> None:
    logger.info("=" * 50)
    logger.info("PHASE 2 — Regime Classifier (HMM)")
    logger.info("=" * 50)

    # 1. Load Data
    df = load_and_prepare_data(cfg.DATA_DIR / "features")
    feature_cols = [c for c in df.columns if c not in ("ticker", "close")]

    # 2. Train Model
    model_dict = train_hmm(df, feature_cols, n_components=cfg.N_REGIMES)

    # 3. Map States
    state_mapping = map_states_to_regimes(model_dict, df, feature_cols)

    # 4. Predict & Map
    X_scaled = model_dict["scaler"].transform(df[feature_cols].values)
    raw_states = model_dict["hmm"].predict(X_scaled)
    raw_regimes = pd.Series(raw_states, index=df.index).map(state_mapping)
    
    # 5. Apply Persistence Filter
    filtered_regimes = apply_persistence_filter(raw_regimes, cfg.REGIME_PERSISTENCE_BARS)
    
    df["regime_raw"] = raw_regimes
    df["regime"] = filtered_regimes

    # Diagnostics
    logger.info("\nRegime Distribution (Filtered):")
    logger.info(df["regime"].value_counts().to_string())
    
    changes = (df["regime"] != df["regime"].shift(1)).sum() - 1 
    logger.info(f"Total Regime Changes (after filter): {changes}")

    # Print Transition Matrix
    logger.info("\nHMM Transition Matrix:")
    tm = model_dict["hmm"].transmat_
    
    # Map the transition matrix row/cols to regime names
    sorted_states = []
    for regime in ["Bull", "Neutral", "Caution", "Defensive"]:
        for state, name in state_mapping.items():
            if name == regime:
                sorted_states.append(state)
                break
                
    tm_reordered = tm[np.ix_(sorted_states, sorted_states)]
    tm_df = pd.DataFrame(
        tm_reordered, 
        index=["Bull", "Neutral", "Caution", "Defensive"],
        columns=["Bull", "Neutral", "Caution", "Defensive"]
    )
    logger.info("\n" + tm_df.round(4).to_string())

    # Save Model
    model_path = cfg.MODELS_DIR / "gmm_pipeline.pkl" # Keep filename for compatibility with generate_insights.py
    with open(model_path, "wb") as f:
        # Save scaler and hmm separately since pipeline isn't used
        pickle.dump({"pipeline": model_dict, "mapping": state_mapping}, f)
    logger.info(f"Saved HMM model to {model_path.name}")

    # Save outputs to data dir for downstream
    output_path = cfg.DATA_DIR / "features" / "SPY_regimes.csv"
    df[["close", "regime_raw", "regime"]].to_csv(output_path)
    logger.info(f"Saved regime predictions to {output_path.name}")

    logger.info("Phase 2 complete. ✓")

if __name__ == "__main__":
    main()
