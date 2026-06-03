#!/usr/bin/env python3
"""
Phase 8 — Profit Scoreboard
===========================
The gate for every model in this pipeline. A ranking is only worth shipping if
it beats (a) SPY buy-hold and (b) the bot's own fixed-weight screener, on
RISK-ADJUSTED return. Without this, "best model for profit" is unfalsifiable.

What it does:
  1. Rebuild the cross-sectional factor panel (Phase 5).
  2. Generate purged walk-forward out-of-fold predictions for the ML ranker.
  3. Build a fixed-weight "screener" score from the same factors (per-day z-scored
     × the screener's published weights) as the baseline to beat.
  4. Backtest each as a daily-rebalanced top-5 long book (matching the bot's 5-6
     position cap) and report Sharpe, Sortino, max drawdown, CAGR, hit rate,
     turnover, and rank IC — all vs SPY buy-hold.
  5. Keep the HMM regime forward-return table + transition matrix.

Saves data/features/training_metrics.json for the bot's dashboard.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as cfg
import phase5_ranking_model as p5

TOP_K = 5
GAP = p5.TARGET_HORIZON
TRADING_DAYS = 252

# Screener factor weights (mirror Trading Bot/scripts/screener.py FACTOR_WEIGHTS),
# mapped onto the factors this panel computes. vol_stability = -atr_pct.
SCREENER_WEIGHTS = {
    "momentum_125d": 0.25, "momentum_20d": 0.10, "rs_vs_sector_60d": 0.18,
    "technical_setup": 0.12, "dist_from_52w_high": 0.08, "volume_surge": 0.08,
    "atr_pct": -0.04,  # lower vol preferred
}


# ============================================================
# Backtest helpers
# ============================================================
def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else pd.Series(0.0, index=s.index)


def screener_score(panel: pd.DataFrame) -> pd.Series:
    """Per-day cross-sectional z-score × fixed screener weights → baseline score."""
    out = pd.Series(0.0, index=panel.index)
    for f, w in SCREENER_WEIGHTS.items():
        z = panel.groupby(level=0)[f].transform(_zscore).clip(-3, 3)
        out = out + w * z
    return out


def walk_forward_oof(panel: pd.DataFrame) -> pd.Series:
    """Purged, date-based expanding walk-forward OOF predictions for the ML ranker."""
    dates = np.array(sorted(panel.index.unique()))
    warmup = int(len(dates) * 0.40)
    test_blocks = np.array_split(dates[warmup:], 5)
    # The panel index is non-unique (one row per symbol per day), so assign through
    # a positional boolean mask on a plain array — `oof.loc[te.index] = ...` would
    # match every duplicate date label and blow up the length.
    oof = np.full(len(panel), np.nan)

    for block in test_blocks:
        if len(block) == 0:
            continue
        block_start = pd.Timestamp(block[0])
        train_cutoff = block_start - pd.Timedelta(days=GAP * 2)  # purge ~5 trading days
        tr = panel[panel.index < train_cutoff]
        te_mask = panel.index.isin(block)
        if len(tr) < 500 or not te_mask.any():
            continue
        model, _ = p5.make_model()
        model.fit(tr[p5.FACTOR_COLS], tr["target"])
        oof[te_mask] = model.predict(panel.loc[te_mask, p5.FACTOR_COLS])
    return pd.Series(oof, index=panel.index)


def rank_ic(score: pd.Series, fwd: pd.Series, dates: pd.Index) -> float:
    df = pd.DataFrame({"d": dates, "s": score.to_numpy(), "f": fwd.to_numpy()}).dropna()
    daily = df.groupby("d").apply(lambda g: spearmanr(g["s"], g["f"]).correlation if len(g) > 3 else np.nan)
    return float(np.nanmean(daily))


def backtest_topk(score: pd.Series, ret_1d: pd.Series, dates: pd.Index, symbols: pd.Index, k: int = TOP_K) -> dict:
    """Daily top-k equal-weight long book. Returns risk-adjusted metrics."""
    df = pd.DataFrame({"date": dates, "symbol": symbols, "score": score.to_numpy(), "ret": ret_1d.to_numpy()}).dropna()
    daily_ret, prev = [], set()
    turn = []
    idx = []
    for d, g in df.groupby("date"):
        picks = set(g.nlargest(k, "score")["symbol"])
        if not picks:
            continue
        daily_ret.append(g[g["symbol"].isin(picks)]["ret"].mean())
        idx.append(d)
        turn.append(len(picks - prev) / k if prev else 1.0)
        prev = picks
    r = pd.Series(daily_ret, index=pd.to_datetime(idx)).dropna()
    return _metrics(r, turnover=float(np.mean(turn)) if turn else None)


def _metrics(r: pd.Series, turnover=None) -> dict:
    if len(r) < 20:
        return {"error": "insufficient_return_series", "days": len(r)}
    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))
    downside = r[r < 0].std(ddof=0) * np.sqrt(TRADING_DAYS)
    equity = (1 + r).cumprod()
    dd = float((equity / equity.cummax() - 1).min())
    cagr = float(equity.iloc[-1] ** (TRADING_DAYS / len(r)) - 1)
    return {
        "days": int(len(r)),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(ann_ret / ann_vol, 3) if ann_vol else None,
        "sortino": round(ann_ret / float(downside), 3) if downside else None,
        "max_drawdown_pct": round(dd * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "hit_rate_pct": round(float((r > 0).mean()) * 100, 2),
        "turnover": round(turnover, 3) if turnover is not None else None,
    }


# ============================================================
# Sections
# ============================================================
def evaluate_ranker() -> dict:
    print("\n" + "=" * 60)
    print("1. Cross-Sectional Ranker — Profit Scoreboard (top-5 long)")
    print("=" * 60)

    bars = p5._load_close_panel(cfg.DATA_DIR / "raw")
    if len(bars) < 10:
        print("  Not enough cached symbols; run Phase 1 first.")
        return {"error": "insufficient_universe"}

    panel = p5.compute_factor_panel(bars)
    panel = panel.dropna(subset=p5.FACTOR_COLS + ["target", "fwd_return"]).copy()
    # Next-1-day return per (date,symbol) for the equity curve.
    panel["ret_1d"] = panel.groupby("symbol")["close"].transform(lambda c: c.pct_change().shift(-1))
    panel = panel.dropna(subset=["ret_1d"])

    dates = panel.index
    syms = pd.Index(panel["symbol"].to_numpy())

    ml_oof = walk_forward_oof(panel)
    scr = screener_score(panel)

    results = {
        "ml_ranker": {
            "oof_rank_ic": round(rank_ic(ml_oof, panel["fwd_return"], dates), 4),
            **backtest_topk(ml_oof, panel["ret_1d"], dates, syms),
        },
        "screener_baseline": {
            "oof_rank_ic": round(rank_ic(scr, panel["fwd_return"], dates), 4),
            **backtest_topk(scr, panel["ret_1d"], dates, syms),
        },
        "spy_buy_hold": _spy_buyhold(dates.unique()),
    }

    for name, m in results.items():
        if "error" in m:
            print(f"  {name:18s}: {m}")
        else:
            print(f"  {name:18s}: Sharpe {m.get('sharpe')}  CAGR {m.get('cagr_pct')}%  "
                  f"maxDD {m.get('max_drawdown_pct')}%  IC {m.get('oof_rank_ic','-')}")

    ml_sh = results["ml_ranker"].get("sharpe") or -9
    base_sh = results["screener_baseline"].get("sharpe") or -9
    verdict = "BEATS" if ml_sh > base_sh else "DOES NOT beat"
    print(f"\n  Verdict: ML ranker {verdict} the screener baseline on Sharpe "
          f"({ml_sh} vs {base_sh}). Ship only if it beats.")
    results["verdict_ml_beats_screener"] = bool(ml_sh > base_sh)
    return results


def _spy_buyhold(dates) -> dict:
    spy = pd.read_csv(cfg.DATA_DIR / "features" / "SPY_features.csv", index_col="date", parse_dates=True).sort_index()
    r = spy["close"].pct_change().reindex(pd.to_datetime(sorted(dates))).dropna()
    return _metrics(r)


def evaluate_hmm_regimes() -> dict:
    print("\n" + "=" * 60)
    print("2. HMM Regime — forward returns by regime")
    print("=" * 60)
    features_dir = cfg.DATA_DIR / "features"
    spy = pd.read_csv(features_dir / "SPY_regimes.csv", index_col="date", parse_dates=True)
    spy["fwd_5d"] = spy["close"].shift(-5) / spy["close"] - 1
    spy = spy.dropna()
    stats = spy.groupby("regime")["fwd_5d"].agg(["mean", "std", "count"])
    stats["mean"] *= 100
    stats["std"] *= 100
    print(stats.round(2).to_string())

    regime_stats = {
        r: {"mean_fwd_return_pct": float(stats.loc[r, "mean"]),
            "volatility_pct": float(stats.loc[r, "std"]),
            "days_in_regime": int(stats.loc[r, "count"])}
        for r in ["Bull", "Neutral", "Caution", "Defensive"] if r in stats.index
    }

    tm_list = []
    try:
        with open(cfg.MODELS_DIR / "gmm_pipeline.pkl", "rb") as f:
            hd = pickle.load(f)
        hmm, mapping = hd["pipeline"]["hmm"], hd["mapping"]
        order = [s for r in ["Bull", "Neutral", "Caution", "Defensive"]
                 for s, n in mapping.items() if n == r]
        tm_list = hmm.transmat_[np.ix_(order, order)].tolist()
    except Exception as e:
        print(f"  transition matrix unavailable: {e}")

    return {"forward_returns": regime_stats, "transition_matrix": tm_list}


if __name__ == "__main__":
    payload = {
        "ranker_scoreboard": evaluate_ranker(),
        "hmm_regimes": evaluate_hmm_regimes(),
    }
    out_path = cfg.DATA_DIR / "features" / "training_metrics.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path.name}")
