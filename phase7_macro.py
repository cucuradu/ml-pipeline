#!/usr/bin/env python3
"""
Phase 7 — Macro / Credit Overlay (FRED)
=======================================
Turns the regime model from price-only into macro-aware, and gives the bot a
quantified macro picture it currently re-derives by hand via Gemini every
morning (see Trading Bot/memory/MACRO-FRAMEWORK.md).

Free + keyless via pandas-datareader's FRED endpoint. The four series are the
best free *leading* risk indicators:
  • yield curve  (DGS10 − DGS3MO)   — inversion precedes recessions
  • HY OAS       (BAMLH0A0HYM2)     — credit stress, the cleanest risk-off tell
  • NFCI         (financial cond.)  — system-wide tightness
  • claims       (ICSA)             — labor deterioration

Outputs:
  • data/features/macro.json          — summary block + crash-flag inputs
  • data/features/macro_features.csv  — daily-aligned series for the regime model

Every component is a trailing-window PERCENTILE (1 = worst in ~3y), blended into
a single macro_risk ∈ [0,1]. No look-ahead: percentiles use only past data, and
FRED values are as-of their (lagged) release, forward-filled to trading days.
Degrades to {"available": false} if pandas-datareader or the network is missing,
so the pipeline never hard-fails on a data outage.
"""

import logging
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

import config as cfg

PCT_WINDOW = 756          # ~3y trailing window for percentile ranks
CLAIMS_SMOOTH = 4         # 4-week average for initial claims
# macro_risk weights — HY OAS is the dominant leading credit signal.
RISK_WEIGHTS = {"hy": 0.35, "curve": 0.25, "nfci": 0.25, "claims": 0.15}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOGS_DIR / "phase7.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def _trailing_pct_rank(s: pd.Series, window: int = PCT_WINDOW) -> pd.Series:
    """Percentile rank of each point within the trailing `window` (inclusive).

    1.0 = current value is the highest in the window. Uses only past+present
    data (min_periods small so early history still scores), so it never leaks.
    """
    return s.rolling(window, min_periods=60).apply(
        lambda w: (w <= w[-1]).mean(), raw=True
    )


def _fred_series(code: str) -> pd.Series | None:
    """One FRED series via the keyless CSV endpoint (no API key, no extra deps).

    pandas-datareader is avoided on purpose — it imports `distutils`, removed in
    Python 3.12. The fredgraph CSV endpoint needs no key and is stable.
    """
    import io
    import time
    import urllib.request

    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
        f"&cosd={cfg.START_DATE}&coed={cfg.END_DATE}"
    )
    # FRED intermittently 504s under load; retry with a hard per-attempt timeout
    # so a hung gateway fails fast instead of blocking ~60s.
    df = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                df = pd.read_csv(io.BytesIO(resp.read()))
            break
        except Exception as e:
            if attempt == 3:
                logger.warning(f"  FRED {code} fetch failed after retries: {e}")
                return None
            time.sleep(2 * (attempt + 1))
    if df is None:
        return None
    if df.shape[1] < 2:
        return None
    idx = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    val = pd.to_numeric(df.iloc[:, 1], errors="coerce")  # "." → NaN
    ser = pd.Series(val.to_numpy(), index=idx).dropna()
    return ser if not ser.empty else None


def fetch_fred() -> pd.DataFrame | None:
    """Daily DataFrame of the FRED series, forward-filled. None on failure."""
    frames = {}
    for name, code in cfg.FRED_SERIES.items():
        ser = _fred_series(code)
        if ser is not None:
            frames[name] = ser
            logger.info(f"  FRED {code} ({name}): {len(ser)} obs")

    if "dgs10" not in frames or "dgs3mo" not in frames:
        logger.warning("Core yield series missing; macro overlay skipped.")
        return None

    df = pd.DataFrame(frames).sort_index().ffill()
    # Carry each series onto a business-day grid. Use reindex(method="ffill"),
    # NOT asfreq("B"): ICSA (initial claims) is dated on Saturdays, which
    # asfreq("B") drops outright — silently nuking the whole series.
    bidx = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(bidx, method="ffill")
    return df


def build_macro(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Engineer macro features + the summary/crash dict from raw FRED data."""
    feat = pd.DataFrame(index=df.index)
    feat["curve"] = df["dgs10"] - df["dgs3mo"]
    feat["hy_oas"] = df.get("hy_oas")
    feat["nfci"] = df.get("nfci")
    if "claims" in df:
        feat["claims_4wk"] = df["claims"].rolling(CLAIMS_SMOOTH, min_periods=1).mean()

    # Component risk percentiles (1 = most risk-off in the trailing window).
    feat["curve_risk"] = 1.0 - _trailing_pct_rank(feat["curve"])      # inverted curve → high
    if "hy_oas" in feat:
        feat["hy_risk"] = _trailing_pct_rank(feat["hy_oas"])          # wide spreads → high
    if "nfci" in feat:
        feat["nfci_risk"] = _trailing_pct_rank(feat["nfci"])         # tight conditions → high
    if "claims_4wk" in feat:
        feat["claims_risk"] = _trailing_pct_rank(feat["claims_4wk"])  # rising claims → high

    # Per-row weighted mean over whichever components are non-NaN that day,
    # renormalizing the weights row by row. A plain Series sum would propagate a
    # single missing component into NaN everywhere.
    comp = {"curve": "curve_risk", "hy": "hy_risk", "nfci": "nfci_risk", "claims": "claims_risk"}
    present = {c: RISK_WEIGHTS[k] for k, c in comp.items() if c in feat}
    C = feat[list(present)]
    W = pd.Series(present)
    denom = C.notna().mul(W, axis=1).sum(axis=1).replace(0.0, np.nan)
    macro_risk = C.mul(W, axis=1).sum(axis=1, min_count=1) / denom
    feat["macro_risk"] = macro_risk.clip(0.0, 1.0)

    valid = feat.dropna(subset=["macro_risk"])
    if valid.empty:
        return feat, neutral_summary()
    last = valid.iloc[-1]
    risk = float(last["macro_risk"])
    label = "Risk-On" if risk < 0.33 else ("Risk-Off" if risk > 0.66 else "Neutral")

    # 20-day change in HY OAS — a fast credit-stress impulse for the crash flag.
    hy_spike = None
    if "hy_oas" in feat:
        hy = feat["hy_oas"].dropna()
        if len(hy) > 20:
            hy_spike = float(hy.iloc[-1] - hy.iloc[-21])

    summary = {
        "available": True,
        "as_of": feat.index[-1].strftime("%Y-%m-%d"),
        "macro_regime": label,
        "macro_risk": round(risk, 3),
        "yield_curve_10y_3m": round(float(last["curve"]), 3),
        "curve_inverted": bool(last["curve"] < 0),
        "hy_oas": round(float(last["hy_oas"]), 3) if "hy_oas" in feat and pd.notna(last.get("hy_oas")) else None,
        "hy_oas_pct": round(float(last["hy_risk"]), 3) if "hy_risk" in feat and pd.notna(last.get("hy_risk")) else None,
        "hy_oas_20d_chg": round(hy_spike, 3) if hy_spike is not None else None,
        "nfci": round(float(last["nfci"]), 3) if "nfci" in feat and pd.notna(last.get("nfci")) else None,
    }
    return feat, summary


def neutral_summary() -> dict:
    return {"available": False, "as_of": datetime.utcnow().strftime("%Y-%m-%d"),
            "macro_regime": "Neutral", "macro_risk": None}


def main() -> None:
    logger.info("=" * 50)
    logger.info("PHASE 7 — Macro / Credit Overlay (FRED)")
    logger.info("=" * 50)

    features_dir = cfg.DATA_DIR / "features"
    df = fetch_fred()

    if df is None:
        summary = neutral_summary()
    else:
        feat, summary = build_macro(df)
        feat.to_csv(features_dir / "macro_features.csv")
        logger.info(
            f"Macro: {summary['macro_regime']} (risk={summary['macro_risk']}), "
            f"curve={summary['yield_curve_10y_3m']}, hy_oas={summary['hy_oas']}"
        )

    with open(features_dir / "macro.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Phase 7 complete. ✓")


if __name__ == "__main__":
    main()
