#!/usr/bin/env python3
"""
Phase 1 — Data Pipeline & Feature Engineering
===============================================
Downloads 5 years of daily OHLCV data for SPY, VIX, and sector ETFs,
then engineers the feature matrix used by downstream models.

Features computed per ticker:
  • log_return          — ln(close_t / close_{t-1})
  • rv_5d .. rv_63d     — realized volatility (annualized std of log returns)
  • sma_dist_20/50/200  — (close - SMA) / SMA  (mean-reversion / trend signal)
  • volume_trend        — volume / rolling_mean(volume, 20)  (participation signal)
  • vix_level           — raw VIX close (added to SPY features for regime model)

No look-ahead bias:
  All features are computed using data available up to and including bar t.
  Downstream models will shift predictions by 1 bar (predict t+1 from t).

Usage:
    python phase1_data_pipeline.py              # fetch + engineer + save
    python phase1_data_pipeline.py --no-fetch   # skip download, re-engineer from cache
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import config as cfg

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOGS_DIR / "phase1.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Data Fetching
# ============================================================
def fetch_all_tickers(
    tickers: list[str],
    start: str,
    end: str,
    save_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV data from Yahoo Finance for each ticker.
    Saves raw CSVs to `save_dir` for reproducibility and caching.
    Returns dict of {ticker: DataFrame}.
    """
    raw_data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tickers:
        logger.info(f"Fetching {ticker} ({start} → {end})...")
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,   # adjusted OHLCV (splits + dividends)
                progress=False,
            )

            if df.empty:
                logger.warning(f"  ⚠ No data returned for {ticker}")
                failed.append(ticker)
                continue

            # Flatten multi-level columns if present (yfinance quirk)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Standardize column names to lowercase
            df.columns = [c.lower() for c in df.columns]

            # Ensure DatetimeIndex
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"

            # Use clean display name for storage and dict keys
            display_name = cfg.TICKER_DISPLAY_NAMES.get(ticker, ticker)

            # Save raw CSV
            csv_path = save_dir / f"{display_name}_raw.csv"
            df.to_csv(csv_path)
            logger.info(f"  ✓ {display_name}: {len(df)} bars → {csv_path.name}")

            raw_data[display_name] = df

        except Exception as e:
            logger.error(f"  ✗ Failed to fetch {ticker}: {e}")
            failed.append(ticker)

    if failed:
        logger.warning(f"Failed tickers: {failed}")

    return raw_data


def load_cached_data(save_dir: Path, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load previously saved raw CSVs from disk."""
    raw_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        display_name = cfg.TICKER_DISPLAY_NAMES.get(ticker, ticker)
        csv_path = save_dir / f"{display_name}_raw.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
            raw_data[display_name] = df
            logger.info(f"  Loaded cached {display_name}: {len(df)} bars")
        else:
            logger.warning(f"  ⚠ No cached data for {display_name}")
    return raw_data


# ============================================================
# 2. Feature Engineering
# ============================================================
def compute_log_returns(close: pd.Series) -> pd.Series:
    """Compute log returns: ln(P_t / P_{t-1})."""
    return np.log(close / close.shift(1))


def compute_realized_volatility(
    log_returns: pd.Series,
    windows: list[int],
) -> pd.DataFrame:
    """
    Rolling realized volatility (annualized).
    rv = std(log_returns, window) * sqrt(252)
    """
    rv_frames = {}
    for w in windows:
        rv_frames[f"rv_{w}d"] = log_returns.rolling(window=w).std() * np.sqrt(252)
    return pd.DataFrame(rv_frames, index=log_returns.index)


def compute_sma_distances(
    close: pd.Series,
    periods: list[int],
) -> pd.DataFrame:
    """
    Distance from price to SMA, normalized:  (close - SMA) / SMA.
    Positive = above SMA (bullish), negative = below (bearish).
    """
    dist_frames = {}
    for p in periods:
        sma = close.rolling(window=p).mean()
        dist_frames[f"sma_dist_{p}"] = (close - sma) / sma
    return pd.DataFrame(dist_frames, index=close.index)


def compute_volume_trend(
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """
    Volume relative to its rolling mean.
    >1 = above-average participation, <1 = quiet.
    """
    rolling_mean = volume.rolling(window=window).mean()
    return (volume / rolling_mean).rename("volume_trend")


def engineer_features_single(
    df: pd.DataFrame,
    ticker: str,
    vix_close: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Full feature engineering for a single ticker.
    Returns a DataFrame with date index and feature columns.
    """
    features = pd.DataFrame(index=df.index)
    features["ticker"] = ticker

    # Raw price (for reference, not a model feature)
    features["close"] = df["close"]

    # Log returns
    log_ret = compute_log_returns(df["close"])
    features["log_return"] = log_ret

    # Realized volatility at multiple horizons
    rv_df = compute_realized_volatility(log_ret, cfg.RV_WINDOWS)
    features = features.join(rv_df)

    # SMA distance (trend / mean-reversion signals)
    sma_df = compute_sma_distances(df["close"], cfg.SMA_PERIODS)
    features = features.join(sma_df)

    # Volume trend
    if "volume" in df.columns and df["volume"].notna().any():
        features["volume_trend"] = compute_volume_trend(
            df["volume"], cfg.VOLUME_TREND_WINDOW
        )
    else:
        # VIX has no volume in some feeds
        features["volume_trend"] = np.nan

    # Attach VIX level as a feature (only for SPY — the regime model's input)
    if vix_close is not None:
        features["vix_level"] = vix_close.reindex(features.index)

    return features


def engineer_all_features(
    raw_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Run feature engineering on every ticker.
    SPY gets VIX level as an extra feature.
    """
    engineered: dict[str, pd.DataFrame] = {}
    vix_close = raw_data.get("VIX", pd.DataFrame()).get("close")

    for ticker, df in raw_data.items():
        # VIX / VIX3M are not tradeable — only used as feature inputs
        if ticker in ("VIX", "VIX3M"):
            logger.info(f"Skipping {ticker} feature engineering (used as input signal only).")
            continue

        logger.info(f"Engineering features for {ticker}...")

        attach_vix = vix_close if ticker == "SPY" else None
        feat = engineer_features_single(df, ticker, attach_vix)

        # Drop rows where features are NaN (warm-up period)
        n_before = len(feat)
        feat = feat.dropna(subset=[c for c in feat.columns if c not in ("ticker", "close")])
        n_after = len(feat)
        logger.info(
            f"  {ticker}: {n_before} → {n_after} rows "
            f"(dropped {n_before - n_after} warm-up rows)"
        )

        engineered[ticker] = feat

    return engineered


# ============================================================
# 3. Save Processed Features
# ============================================================
def save_features(
    engineered: dict[str, pd.DataFrame],
    output_dir: Path,
) -> Path:
    """
    Save all engineered features:
      - One CSV per ticker in data/features/
      - One combined parquet file for easy downstream loading
    """
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []

    for ticker, df in engineered.items():
        csv_path = features_dir / f"{ticker}_features.csv"
        df.to_csv(csv_path)
        logger.info(f"  Saved {csv_path.name}")
        all_frames.append(df)

    # Combined dataset
    combined = pd.concat(all_frames)
    combined_path = features_dir / "all_features.csv"
    combined.to_csv(combined_path)
    logger.info(f"  Saved combined feature set → {combined_path.name} ({len(combined)} rows)")

    return features_dir


# ============================================================
# 4. Diagnostic Summary
# ============================================================
def print_diagnostics(engineered: dict[str, pd.DataFrame]) -> None:
    """Print a quick summary table of engineered features."""
    print("\n" + "=" * 70)
    print("PHASE 1 — FEATURE ENGINEERING DIAGNOSTICS")
    print("=" * 70)

    summary_rows = []
    for ticker, df in engineered.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary_rows.append(
            {
                "Ticker": ticker,
                "Rows": len(df),
                "Features": len(numeric_cols),
                "Date Range": f"{df.index.min().date()} → {df.index.max().date()}",
                "NaN %": f"{df[numeric_cols].isna().mean().mean() * 100:.1f}%",
                "Last Log Return": (
                    f"{df['log_return'].iloc[-1]:.4f}"
                    if "log_return" in df and len(df) > 0
                    else "N/A"
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Feature correlation preview for SPY
    if "SPY" in engineered:
        spy = engineered["SPY"]
        feature_cols = [c for c in spy.columns if c not in ("ticker", "close")]
        print("\n--- SPY Feature Correlation Matrix (last 252 bars) ---")
        corr = spy[feature_cols].tail(252).corr()
        print(corr.round(2).to_string())

    print("=" * 70 + "\n")


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Data Pipeline & Feature Engineering")
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip data download; use cached CSVs instead.",
    )
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("PHASE 1 — Data Pipeline & Feature Engineering")
    logger.info(f"  Tickers: {cfg.ALL_TICKERS}")
    logger.info(f"  Period:  {cfg.START_DATE} → {cfg.END_DATE}")
    logger.info("=" * 50)

    # --- Step 1: Fetch or load data ---
    raw_dir = cfg.DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.no_fetch:
        logger.info("Using cached data (--no-fetch flag).")
        raw_data = load_cached_data(raw_dir, cfg.ALL_TICKERS)
    else:
        raw_data = fetch_all_tickers(
            tickers=cfg.ALL_TICKERS,
            start=cfg.START_DATE,
            end=cfg.END_DATE,
            save_dir=raw_dir,
        )

    if not raw_data:
        logger.error("No data available. Exiting.")
        sys.exit(1)

    # --- Step 2: Engineer features ---
    engineered = engineer_all_features(raw_data)

    # --- Step 3: Save ---
    save_features(engineered, cfg.DATA_DIR)

    # --- Step 4: Diagnostics ---
    print_diagnostics(engineered)

    logger.info("Phase 1 complete. ✓")


if __name__ == "__main__":
    main()
