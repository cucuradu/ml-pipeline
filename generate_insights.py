#!/usr/bin/env python3
"""
Phase 4 — Assembly & Automation Hook
====================================
Runs the full pipeline and assembles ml-insights.json — the FROZEN contract the
Claude trading bot consumes (Trading Bot/docs/ml-insights-schema.md).

This rewrite (model_version hmm-garch-v2.0):
  • Emits the exact contract shape the consumer parses: universe_ranking as
    [{symbol, ml_score}], sectors.*.score, volatility 1d+5d. (The experimental
    branch had drifted to a shape the consumer silently drops.)
  • Reports a REAL HMM posterior confidence (over the full sequence, not a single
    saturated row) and a REAL persistence_bars (trailing consecutive run).
  • Sharpe-oriented risk haircut: trims deployment when fragility / crash-risk /
    macro stress is high (the de-risk lever the bot reserved these fields for).
  • Adds forward-compatible blocks the consumer tolerates: `macro`, `crash_risk`,
    extra `volatility.*`, and `forward_return_pct` alongside the legacy keys
    (versioned rename, followup P2 — both emitted for one rotation).
"""

import sys
import json
import logging
import subprocess
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOGS_DIR / "phase4.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

UNIVERSE_RANKING_TOPN = 15   # how many ranked names to surface to the bot


def run_script(script_name: str) -> None:
    script_path = cfg.PROJECT_ROOT / script_name
    logger.info(f"Running {script_path.name}...")
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=str(cfg.PROJECT_ROOT))


def _read_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# Regime confidence + persistence (the fixed versions)
# ============================================================
def regime_confidence(features_dir) -> float:
    """Calibrated confidence for the current regime call.

    The raw HMM posterior is useless here — with separable regimes it saturates
    to ~1.0 on ~90% of days (followup S1/P7; confirmed to persist even with diag
    covariance + min_covar). Instead we blend two honest, non-saturating signals:
      • typicality — percentile of today's Gaussian emission likelihood under the
        assigned state vs. that state's own history (is today a normal day for the
        regime, or an outlier on the edge of a transition?).
      • stickiness — the state's self-transition probability (the model's own
        estimate that the regime holds tomorrow).
    """
    spy = pd.read_csv(features_dir / "SPY_features.csv", index_col="date", parse_dates=True).sort_index()
    macro_path = features_dir / "macro_features.csv"
    if macro_path.exists():
        macro = pd.read_csv(macro_path, index_col=0, parse_dates=True).sort_index()
        for col in ("macro_risk", "hy_oas"):
            if col in macro.columns:
                spy[col] = macro[col].reindex(spy.index, method="ffill")

    with open(cfg.MODELS_DIR / "gmm_pipeline.pkl", "rb") as f:
        hmm_data = pickle.load(f)
    scaler = hmm_data["pipeline"]["scaler"]
    hmm = hmm_data["pipeline"]["hmm"]

    feature_cols = [c for c in spy.columns if c not in ("ticker", "close")]
    shifted = spy.copy()
    shifted[feature_cols] = spy[feature_cols].shift(1)
    shifted = shifted.dropna(subset=feature_cols)

    X = scaler.transform(shifted[feature_cols].values)
    states = hmm.predict(X)
    s = int(states[-1])

    # Per-row, per-state diag-Gaussian emission log-likelihood (no private API).
    covars = hmm.covars_
    var = np.diagonal(covars, axis1=1, axis2=2) if covars.ndim == 3 else covars
    var = np.clip(var, 1e-6, None)
    diff = X[:, None, :] - hmm.means_[None, :, :]
    emis = -0.5 * (np.log(2 * np.pi * var)[None] + diff ** 2 / var[None]).sum(axis=2)

    today_ll = emis[-1, s]
    hist_ll = emis[states == s, s]
    typicality = float((hist_ll <= today_ll).mean()) if hist_ll.size else 0.5
    stickiness = float(hmm.transmat_[s, s])
    return float(np.clip(0.5 * typicality + 0.5 * stickiness, 0.05, 0.99))


def persistence_run(regimes: pd.Series) -> int:
    """Number of trailing trading days the current regime has held."""
    last = regimes.iloc[-1]
    run = 0
    for val in regimes.iloc[::-1]:
        if val == last:
            run += 1
        else:
            break
    return int(run)


# ============================================================
# Sector regime (robust rule matching the consumer's documented target)
# ============================================================
def sector_regime(features_dir, sector: str) -> str:
    df = pd.read_csv(features_dir / f"{sector}_features.csv", index_col="date", parse_dates=True).sort_index()
    sma50_dist = float(df["sma_dist_50"].iloc[-1])
    ret_10d = float(df["close"].iloc[-1] / df["close"].iloc[-11] - 1.0) if len(df) > 11 else 0.0
    if sma50_dist < 0 or ret_10d < -0.04:
        return "Bear"
    if sma50_dist > 0 and ret_10d > 0.02:
        return "Trend"
    return "Choppy"


# ============================================================
# Crash-risk flag (additive)
# ============================================================
def crash_risk(term_structure, vvix_level, fragility, macro, breadth_divergence) -> dict:
    """0-1 crash score + boolean from independent tail-risk tells. Additive field."""
    score = 0.0
    reasons = []
    if term_structure == "backwardation":
        score += 0.30; reasons.append("vix_backwardation")
    if vvix_level is not None and vvix_level > 110:
        score += 0.15; reasons.append("vvix_elevated")
    if fragility is not None and fragility > 0.80:
        score += 0.25; reasons.append("high_fragility")
    if macro.get("available"):
        chg = macro.get("hy_oas_20d_chg")
        if chg is not None and chg > 0.30 and (macro.get("hy_oas_pct") or 0) > 0.7:
            score += 0.20; reasons.append("credit_spread_spike")
        if macro.get("curve_inverted"):
            score += 0.05; reasons.append("curve_inverted")
    if breadth_divergence:
        score += 0.05; reasons.append("breadth_divergence")
    score = round(min(score, 1.0), 3)
    return {"score": score, "elevated": bool(score >= 0.5), "reasons": reasons}


# ============================================================
# Assembly
# ============================================================
def generate_json() -> None:
    features_dir = cfg.DATA_DIR / "features"

    # --- Regime ---
    spy_regimes = pd.read_csv(features_dir / "SPY_regimes.csv", index_col="date", parse_dates=True).sort_index()
    final_regime = spy_regimes["regime"].iloc[-1]
    confidence = regime_confidence(features_dir)
    persistence = persistence_run(spy_regimes["regime"])
    logger.info(f"Regime: {final_regime} | confidence: {confidence:.4f} | persistence: {persistence}d")

    # --- Breadth divergence (SPY cap-weight vs RSP equal-weight) ---
    spy_feat = pd.read_csv(features_dir / "SPY_features.csv", index_col="date", parse_dates=True).sort_index()
    rsp_feat = pd.read_csv(features_dir / "RSP_features.csv", index_col="date", parse_dates=True).sort_index()
    spy_20d = spy_feat["close"].iloc[-1] / spy_feat["close"].iloc[-20] - 1
    rsp_20d = rsp_feat["close"].iloc[-1] / rsp_feat["close"].iloc[-20] - 1
    breadth_divergence = bool((spy_20d > 0) and (rsp_20d < 0))
    if breadth_divergence and final_regime == "Bull":
        logger.warning("Breadth divergence: downgrading Bull → Caution")
        final_regime = "Caution"

    # --- Volatility (annualized %, already correct from Phase 3) ---
    garch = _read_json(features_dir / "garch_forecast.json")
    vix = pd.read_csv(cfg.DATA_DIR / "raw" / "VIX_raw.csv", index_col="date", parse_dates=True)["close"].iloc[-1]
    vix3m = pd.read_csv(cfg.DATA_DIR / "raw" / "VIX3M_raw.csv", index_col="date", parse_dates=True)["close"].iloc[-1]
    term_structure = "contango" if vix3m >= vix else "backwardation"
    vvix = None
    vvix_path = cfg.DATA_DIR / "raw" / "VVIX_raw.csv"
    if vvix_path.exists():
        vvix = float(pd.read_csv(vvix_path, index_col="date", parse_dates=True)["close"].iloc[-1])

    # --- Fragility + macro ---
    fragility = _read_json(features_dir / "systemic_fragility.json")["systemic_fragility"]
    macro = _read_json(features_dir / "macro.json") if (features_dir / "macro.json").exists() else {"available": False}

    crash = crash_risk(term_structure, vvix, fragility, macro, breadth_divergence)

    # --- Ranking (full universe) ---
    rank_data = _read_json(features_dir / "universe_ranking.json")
    ranking_full = rank_data.get("ranking", [])
    universe_ranking = [
        {"symbol": r["symbol"], "ml_score": r["ml_score"], "forward_return_pct": r["forward_return_pct"]}
        for r in ranking_full[:UNIVERSE_RANKING_TOPN]
    ]
    sector_predictions = rank_data.get("sector_predictions", {})

    # --- Deployment with Sharpe-oriented risk haircut ---
    base = cfg.REGIME_DEPLOYMENT_MAP[final_regime]
    deployment = base["deployment_target"]
    trade_slots = base["trade_slots"]
    haircut = 1.0
    if fragility is not None and fragility > 0.80:
        haircut *= 0.80
    if crash["elevated"]:
        haircut *= 0.70
    if macro.get("available") and (macro.get("macro_risk") or 0) > 0.70:
        haircut *= 0.85
    deployment = round(min(deployment * haircut, 0.85), 3)
    if haircut < 1.0:
        logger.info(f"Risk haircut {haircut:.2f} → deployment {base['deployment_target']} → {deployment}")

    # --- universe_weights: inverse-vol over top trade_slots ranked names ---
    universe_weights = {}
    if trade_slots >= 1 and ranking_full:
        top = [r["symbol"] for r in ranking_full[:trade_slots]]
        if len(top) == 1:
            universe_weights = {top[0]: 1.0}     # followup P6: skip the no-op IV step
        else:
            inv = {}
            for sym in top:
                fp = features_dir / f"{sym}_features.csv"
                if fp.exists():
                    rv = pd.read_csv(fp, index_col="date", parse_dates=True)["log_return"].iloc[-21:].std() * np.sqrt(252)
                    inv[sym] = 1.0 / rv if rv and rv > 0 else 0.0
            tot = sum(inv.values()) or 1.0
            universe_weights = {k: round(v / tot, 4) for k, v in inv.items()}

    # --- Sectors block ---
    sectors_dict = {}
    for sector in cfg.SECTOR_TICKERS:
        sectors_dict[sector] = {
            "regime": sector_regime(features_dir, sector),
            "score": round(float(sector_predictions.get(sector, 0.0)) / 100.0, 6),  # back to decimal
            "forward_return_pct": round(float(sector_predictions.get(sector, 0.0)), 4),
        }

    insights = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": cfg.MODEL_VERSION,
        "schema_version": cfg.SCHEMA_VERSION,
        "market": {
            "regime": final_regime,
            "confidence": round(confidence, 4),
            "persistence_bars": persistence,
            "breadth_divergence": breadth_divergence,
            "systemic_fragility": round(float(fragility), 4),
            "deployment_target": deployment,
            "trade_slots": trade_slots,
        },
        "universe_ranking": universe_ranking,
        "universe_weights": universe_weights,
        "volatility": {
            "garch_1d_forecast_pct": garch["garch_1d_forecast_pct"],
            "garch_5d_forecast_pct": garch.get("garch_5d_forecast_pct"),
            "vix_implied_term_structure": term_structure,
            "vix": round(float(vix), 2),
            "vvix": vvix,
        },
        "macro": macro,
        "crash_risk": crash,
        "ranking_quality": {"oof_rank_ic": rank_data.get("oof_rank_ic")},
        "sectors": sectors_dict,
    }

    with open(cfg.INSIGHTS_PATH, "w") as f:
        json.dump(insights, f, indent=2)
    logger.info(f"Generated {cfg.INSIGHTS_PATH.name}")

    _append_history(insights)
    print(json.dumps(insights, indent=2))


def _append_history(insights: dict) -> None:
    """Append today's payload to output/history.jsonl, idempotent by date.

    The consumer reads docs/history.jsonl alongside the live file. Re-running on
    the same date overwrites that date's line rather than duplicating it
    (matches the architecture doc's idempotency requirement).
    """
    hist_path = cfg.OUTPUT_DIR / "history.jsonl"
    today = insights["generated_at"][:10]
    lines = []
    if hist_path.exists():
        for ln in hist_path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                if json.loads(ln).get("generated_at", "")[:10] != today:
                    lines.append(ln)
            except json.JSONDecodeError:
                continue
    lines.append(json.dumps(insights))
    hist_path.write_text("\n".join(lines) + "\n")


def main():
    logger.info("=" * 50)
    logger.info("PHASE 4 — Pipeline Assembly")
    logger.info("=" * 50)

    run_script("phase1_data_pipeline.py")
    run_script("phase7_macro.py")          # macro before regime so features exist
    run_script("phase2_regime_model.py")
    run_script("phase3_volatility_model.py")
    run_script("phase5_ranking_model.py")
    run_script("phase6_gnn_topology.py")

    logger.info("Assembling insights JSON...")
    generate_json()

    logger.info("Evaluating models (scoreboard)...")
    run_script("evaluate_models.py")

    logger.info("All phases complete. ✓")


if __name__ == "__main__":
    main()
