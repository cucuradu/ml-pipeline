#!/usr/bin/env python3
"""
Phase 4 — Assembly & Automation Hook
====================================
Runs the full pipeline, loads all models and features, and generates
the final `ml-insights.json` deliverable.

Bug fixes applied:
  • Look-ahead bias: Features are shifted by 1 day before predicting.
  • Confidence: Reports raw model confidence + adds filtered flag.
  • GARCH scale: Outputs percentage (12.18) not decimal (0.1218).
  • Script paths: Uses absolute paths so cron doesn't break.
  • Logging: Writes to phase4.log, not just stdout.
"""

import sys
import json
import logging
import subprocess
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOGS_DIR / "phase4.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def run_script(script_name: str) -> None:
    """Run a phase script using absolute path so cron can find it."""
    script_path = cfg.PROJECT_ROOT / script_name
    logger.info(f"Running {script_path.name}...")
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=str(cfg.PROJECT_ROOT))


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def generate_json() -> None:
    features_dir = cfg.DATA_DIR / "features"

    # ----------------------------------------------------------------
    # 1. Market Regime & Confidence
    # ----------------------------------------------------------------
    spy_feat = pd.read_csv(features_dir / "SPY_features.csv", index_col="date", parse_dates=True)
    spy_feat = spy_feat.sort_index()

    # Identify model feature columns (exclude metadata)
    feature_cols = [c for c in spy_feat.columns if c not in ("ticker", "close")]

    # CRITICAL: Shift features by 1 to prevent look-ahead bias.
    # The GMM was trained on shifted features (T-1 features predicting T's regime).
    # We must do the same at inference. After shifting, iloc[-1] contains
    # the features from the SECOND-TO-LAST trading day, which is what we need
    # to make a prediction for the LAST (most recent) trading day.
    spy_shifted = spy_feat.copy()
    spy_shifted[feature_cols] = spy_feat[feature_cols].shift(1)
    spy_shifted = spy_shifted.dropna(subset=feature_cols)

    # Use the last row of shifted features (= features from T-1)
    X_latest = spy_shifted[feature_cols].iloc[-1:].values
    prediction_date = spy_shifted.index[-1].strftime("%Y-%m-%d")
    logger.info(f"Predicting regime for {prediction_date} using T-1 features.")

    # Load HMM
    with open(cfg.MODELS_DIR / "gmm_pipeline.pkl", "rb") as f:
        hmm_data = pickle.load(f)
    scaler = hmm_data["pipeline"]["scaler"]
    hmm = hmm_data["pipeline"]["hmm"]
    mapping = hmm_data["mapping"]

    # Predict raw regime & confidence
    X_scaled = scaler.transform(X_latest)
    proba = hmm.predict_proba(X_scaled)[0]
    cluster = hmm.predict(X_scaled)[0]
    raw_regime = mapping[cluster]
    raw_confidence = float(np.max(proba))

    # Get filtered regime from persistence filter output
    spy_regimes = pd.read_csv(features_dir / "SPY_regimes.csv", index_col="date", parse_dates=True)
    final_regime = spy_regimes["regime"].iloc[-1]

    # Determine if the persistence filter is actively overriding
    persistence_active = (final_regime != raw_regime)
    if persistence_active:
        logger.info(
            f"Persistence filter active: raw={raw_regime} ({raw_confidence:.2%}), "
            f"filtered={final_regime}"
        )

    # Always report the RAW model confidence (what the model actually believes today)
    # The cloud engine can use the filtered flag to know if persistence is overriding.
    logger.info(f"Regime: {final_regime} | Raw: {raw_regime} | Confidence: {raw_confidence:.4f}")

    # ----------------------------------------------------------------
    # 1.5. Breadth Divergence (RSP vs SPY)
    # ----------------------------------------------------------------
    rsp_feat = pd.read_csv(features_dir / "RSP_features.csv", index_col="date", parse_dates=True)
    rsp_feat = rsp_feat.sort_index()
    
    # Calculate rolling 20-day return for both to see intermediate trend
    spy_20d_ret = (spy_feat["close"].iloc[-1] / spy_feat["close"].iloc[-20]) - 1
    rsp_20d_ret = (rsp_feat["close"].iloc[-1] / rsp_feat["close"].iloc[-20]) - 1
    
    # Breadth divergence: SPY is up over 20 days, but RSP is down
    breadth_divergence = bool((spy_20d_ret > 0) and (rsp_20d_ret < 0))
    logger.info(f"SPY 20d: {spy_20d_ret:.2%}, RSP 20d: {rsp_20d_ret:.2%} -> Divergence: {breadth_divergence}")
    
    if breadth_divergence and final_regime == "Bull":
        logger.warning("Breadth Divergence detected! Downgrading Bull -> Caution")
        final_regime = "Caution"

    # ----------------------------------------------------------------
    # 2. Volatility
    # ----------------------------------------------------------------
    with open(features_dir / "garch_forecast.json", "r") as f:
        garch_data = json.load(f)
    # Convert from decimal to percentage (0.1218 → 12.18)
    garch_forecast_pct = round(garch_data["garch_1d_forecast_pct"] * 100, 2)

    # VIX Term Structure (VIX vs VIX3M proxy)
    vix_raw = pd.read_csv(cfg.DATA_DIR / "raw" / "VIX_raw.csv", index_col="date", parse_dates=True)
    vix3m_raw = pd.read_csv(cfg.DATA_DIR / "raw" / "VIX3M_raw.csv", index_col="date", parse_dates=True)

    vix_latest = vix_raw["close"].iloc[-1]
    vix3m_latest = vix3m_raw["close"].iloc[-1]
    term_structure = "contango" if vix3m_latest >= vix_latest else "backwardation"
    logger.info(f"VIX: {vix_latest:.2f} | VIX3M: {vix3m_latest:.2f} | Term: {term_structure}")

    # ----------------------------------------------------------------
    # 3. XGBoost Ranking & Systemic Fragility
    # ----------------------------------------------------------------
    with open(features_dir / "universe_ranking.json", "r") as f:
        ranking_data = json.load(f)
        
    top_3 = ranking_data["top_3_sectors"]
    top_3_vol = {}
    for sector in top_3:
        sec_feat = pd.read_csv(features_dir / f"{sector}_features.csv", index_col="date", parse_dates=True)
        # Calculate annualized 21-day realized volatility
        rv21 = sec_feat["log_return"].iloc[-21:].std() * np.sqrt(252)
        top_3_vol[sector] = rv21
        
    # Inverse Volatility Weighting
    inv_vol = {k: 1.0 / v for k, v in top_3_vol.items()}
    total_inv_vol = sum(inv_vol.values())
    universe_weights = {k: round(v / total_inv_vol, 4) for k, v in inv_vol.items()}
        
    with open(features_dir / "systemic_fragility.json", "r") as f:
        fragility_score = json.load(f)["systemic_fragility"]

    # ----------------------------------------------------------------
    # 4. Sectors
    # ----------------------------------------------------------------
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

        # The actual forward return prediction from XGBoost
        xgb_pred = ranking_data["predictions"].get(sector, 0.0)

        sectors_dict[sector] = {
            "regime": sec_regime,
            "xgb_fwd_return_pred": round(xgb_pred, 4)
        }

    # Build JSON (The Handshake Contract)
    # ----------------------------------------------------------------
    deployment_info = cfg.REGIME_DEPLOYMENT_MAP[final_regime]

    insights = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": cfg.MODEL_VERSION,
        "market": {
            "regime": final_regime,
            "confidence": round(raw_confidence, 4),
            "persistence_bars": cfg.REGIME_PERSISTENCE_BARS,
            "breadth_divergence": breadth_divergence,
            "systemic_fragility": fragility_score,
            "deployment_target": deployment_info["deployment_target"],
            "trade_slots": deployment_info["trade_slots"]
        },
        "universe_ranking": ranking_data["top_3_sectors"],
        "universe_weights": universe_weights,
        "volatility": {
            "garch_1d_forecast_pct": garch_forecast_pct,
            "vix_implied_term_structure": term_structure
        },
        "sectors": sectors_dict
    }

    # Save
    with open(cfg.INSIGHTS_PATH, "w") as f:
        json.dump(insights, f, indent=2)

    logger.info(f"Generated {cfg.INSIGHTS_PATH.name}")
    print(json.dumps(insights, indent=2))


def main():
    logger.info("=" * 50)
    logger.info("PHASE 4 — Pipeline Assembly")
    logger.info("=" * 50)

    run_script("phase1_data_pipeline.py")
    run_script("phase2_regime_model.py")
    run_script("phase3_volatility_model.py")
    run_script("phase5_ranking_model.py")
    run_script("phase6_gnn_topology.py")

    logger.info("Generating Insights JSON...")
    generate_json()

    logger.info("Evaluating Models for Dashboard...")
    run_script("evaluate_models.py")

    logger.info("All phases complete. ✓")


if __name__ == "__main__":
    main()
