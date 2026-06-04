#!/usr/bin/env python3
"""
Phase 5 — Cross-Sectional Universe Ranker
=========================================
Ranks the FULL 70-name tradeable universe (megacaps + ETFs), not just the 11
sector ETFs. This is the field that actually drives the bot's stock selection
(`universe_ranking`); previously it only ranked sectors, so the bot fell back to
its own local `screener.py`. To add value, this model must out-rank that screener
(measured by information coefficient in the backtest harness, Phase 8).

Design choices that fix the prior model's weaknesses:
  • Features = the same battle-tested factors the screener uses (momentum at
    20/60/125d, relative strength vs sector, technical setup, 52w-high distance,
    volume surge, ATR%, realized vol) — so the comparison is learned-weights vs
    fixed-weights, apples to apples.
  • Target = CROSS-SECTIONAL forward return (5d return minus the day's universe
    mean). Predicting raw forward return was sub-noise (followup S6); demeaning
    turns it into a market-neutral ranking problem the model can actually learn.
  • Purged walk-forward CV (TimeSeriesSplit + gap) reports out-of-fold rank IC —
    the honest measure of whether the ranking has edge.
  • Model = xgboost if importable, else sklearn HistGradientBoostingRegressor
    (no libomp/GPU needed), behind one interface so it runs on any host.

Output (data/features/universe_ranking.json):
  {
    "as_of": "...",
    "oof_rank_ic": 0.0x,
    "ranking": [{"symbol","forward_return_pct","sector","ml_score"}, ...],  # desc
    "sector_predictions": {XLK: pct, ...}
  }
"""

import logging
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit

import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOGS_DIR / "phase5.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

TARGET_HORIZON = 5
FACTOR_COLS = [
    "momentum_20d", "momentum_60d", "momentum_125d", "rs_vs_sector_60d",
    "technical_setup", "dist_from_52w_high", "volume_surge", "atr_pct", "rv_21d",
]


# ============================================================
# Factor panel
# ============================================================
def _load_close_panel(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load cached raw OHLCV (lowercase cols) for every tradeable symbol."""
    bars: dict[str, pd.DataFrame] = {}
    for sym in cfg.TRADEABLE_UNIVERSE:
        path = raw_dir / f"{sym}_raw.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
        if {"close", "high", "low", "volume"}.issubset(df.columns) and len(df) >= 220:
            bars[sym] = df
    return bars


def _sector_returns_60d(bars: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """60d trailing return series for each sector ETF (for relative strength)."""
    out = {}
    for etf in cfg.SECTOR_TICKERS:
        if etf in bars:
            c = bars[etf]["close"].astype(float)
            out[etf] = c / c.shift(60) - 1.0
    return out


def _atr_pct(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return atr / close * 100.0


def compute_factor_panel(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long panel: one row per (date, symbol) with factor columns + fwd return."""
    sector_ret60 = _sector_returns_60d(bars)
    frames = []
    for sym, df in bars.items():
        close = df["close"].astype(float)
        vol = df["volume"].astype(float)
        logret = np.log(close / close.shift(1))
        sec = cfg.SECTOR_OF.get(sym)

        f = pd.DataFrame(index=df.index)
        f["symbol"] = sym
        f["sector"] = sec
        f["close"] = close
        f["momentum_20d"] = close / close.shift(20) - 1.0
        f["momentum_60d"] = close / close.shift(60) - 1.0
        f["momentum_125d"] = close / close.shift(125) - 1.0

        own60 = f["momentum_60d"]
        if sec == "BROAD" or sec is None:
            f["rs_vs_sector_60d"] = 0.0
        else:
            f["rs_vs_sector_60d"] = own60 - sector_ret60.get(sec, pd.Series(index=df.index)).reindex(df.index)

        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        golden = ((close > sma50) & (sma50 > sma200)).astype(float)
        high252 = close.rolling(252, min_periods=200).max()
        pct_below = (high252 - close) / high252
        prox = (1.0 - pct_below / 0.20).clip(0.0, 1.0)
        f["technical_setup"] = 0.5 * golden + 0.5 * prox
        f["dist_from_52w_high"] = (1.0 - (high252 - close) / close).clip(0.0, 1.0)

        v_recent = vol.rolling(5).mean()
        v_prior = vol.shift(5).rolling(15).mean()
        f["volume_surge"] = (v_recent / v_prior - 1.0).replace([np.inf, -np.inf], np.nan)

        f["atr_pct"] = _atr_pct(df)
        f["rv_21d"] = logret.rolling(21).std() * np.sqrt(252)

        # Target: 5d forward return (demeaned cross-sectionally below).
        f["fwd_return"] = close.shift(-TARGET_HORIZON) / close - 1.0
        frames.append(f)

    panel = pd.concat(frames).sort_index()
    # Cross-sectional demean of the target → market-neutral ranking objective.
    panel["target"] = panel["fwd_return"] - panel.groupby(level=0)["fwd_return"].transform("mean")
    return panel


# ============================================================
# Model (xgboost if available, else sklearn HGB)
# ============================================================
def make_model():
    try:
        import xgboost as xgb  # noqa
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=2.0, random_state=cfg.GMM_RANDOM_STATE, n_jobs=-1,
        )
        return model, "xgboost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.03, max_iter=400,
            l2_regularization=2.0, min_samples_leaf=80,
            random_state=cfg.GMM_RANDOM_STATE,
        )
        return model, "hist_gradient_boosting"


def walk_forward_ic(X: pd.DataFrame, y: pd.Series, dates: pd.Index) -> float:
    """Mean out-of-fold daily rank IC under purged walk-forward CV."""
    tscv = TimeSeriesSplit(n_splits=5, gap=TARGET_HORIZON)
    ics: list[float] = []
    for tr_idx, te_idx in tscv.split(X):
        model, _ = make_model()
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = model.predict(X.iloc[te_idx])
        te = pd.DataFrame({"date": dates[te_idx], "pred": preds, "y": y.iloc[te_idx].to_numpy()})
        daily = te.groupby("date").apply(
            lambda g: spearmanr(g["pred"], g["y"]).correlation if len(g) > 3 else np.nan
        )
        ics.append(float(np.nanmean(daily)))
    return float(np.nanmean(ics)) if ics else float("nan")


def main():
    logger.info("=" * 50)
    logger.info("PHASE 5 — Cross-Sectional Universe Ranker")
    logger.info("=" * 50)

    raw_dir = cfg.DATA_DIR / "raw"
    features_dir = cfg.DATA_DIR / "features"

    bars = _load_close_panel(raw_dir)
    logger.info(f"Loaded {len(bars)}/{len(cfg.TRADEABLE_UNIVERSE)} universe symbols with usable history.")
    if len(bars) < 10:
        logger.error("Too few symbols with cached bars; run Phase 1 first.")
        sys.exit(1)

    panel = compute_factor_panel(bars)
    train = panel.dropna(subset=FACTOR_COLS + ["target"]).copy()
    logger.info(f"Training panel: {len(train)} (date, symbol) rows.")

    X, y = train[FACTOR_COLS], train["target"]
    oof_ic = walk_forward_ic(X, y, train.index)
    logger.info(f"Out-of-fold rank IC (purged walk-forward): {oof_ic:.4f}")

    model, backend = make_model()
    model.fit(X, y)
    logger.info(f"Final model trained on 100% of panel ({backend}).")

    with open(cfg.MODELS_DIR / "ranking_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": FACTOR_COLS, "backend": backend}, f)

    # Predict the latest cross-section (last available factor row per symbol).
    latest = panel.dropna(subset=FACTOR_COLS).groupby("symbol").tail(1).copy()
    latest["pred"] = model.predict(latest[FACTOR_COLS])
    latest = latest.sort_values("pred", ascending=False)

    ranking = [
        {
            "symbol": row["symbol"],
            "forward_return_pct": round(float(row["pred"]) * 100, 4),
            "ml_score": round(float(row["pred"]), 6),  # back-compat alias during migration
            "sector": row["sector"],
        }
        for _, row in latest.iterrows()
    ]
    sector_predictions = {
        row["symbol"]: round(float(row["pred"]) * 100, 4)
        for _, row in latest.iterrows() if row["symbol"] in cfg.SECTOR_TICKERS
    }

    logger.info("Top 8 ranked names:")
    for r in ranking[:8]:
        logger.info(f"  {r['symbol']:5s} excess5d={r['forward_return_pct']:+.3f}% sector={r['sector']}")

    out = {
        "as_of": latest.index.max().strftime("%Y-%m-%d"),
        "oof_rank_ic": round(oof_ic, 4),
        "ranking": ranking,
        "sector_predictions": sector_predictions,
    }
    with open(features_dir / "universe_ranking.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved universe_ranking.json")
    logger.info("Phase 5 complete. ✓")


if __name__ == "__main__":
    main()
