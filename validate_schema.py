#!/usr/bin/env python3
"""
Schema Conformance Validator
============================
Validates output/ml-insights.json against the expected schema contract
BEFORE pushing to any remote. Run this as a gate in run_pipeline.sh.

Usage:
    python validate_schema.py                    # validate default output path
    python validate_schema.py path/to/file.json  # validate specific file

Exit codes:
    0 = valid
    1 = invalid (prints all violations to stderr)
"""

import json
import sys
from pathlib import Path

import config as cfg


def validate(data: dict) -> list[str]:
    """
    Validate ml-insights.json against the schema contract.
    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []

    # --- Top-level keys ---
    required_top = ["generated_at", "model_version", "market", "universe_ranking",
                     "universe_weights", "volatility", "sectors"]
    for key in required_top:
        if key not in data:
            errors.append(f"Missing top-level key: '{key}'")

    if errors:
        # If top-level structure is broken, no point checking deeper
        return errors

    # --- market ---
    market = data["market"]
    market_required = ["regime", "confidence", "persistence_bars", "breadth_divergence",
                       "systemic_fragility", "deployment_target", "trade_slots"]
    for key in market_required:
        if key not in market:
            errors.append(f"Missing market key: '{key}'")

    if "regime" in market:
        valid_regimes = {"Bull", "Neutral", "Caution", "Defensive"}
        if market["regime"] not in valid_regimes:
            errors.append(f"Invalid regime '{market['regime']}', expected one of {valid_regimes}")

    if "confidence" in market:
        conf = market["confidence"]
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            errors.append(f"confidence must be float in [0, 1], got {conf}")

    if "persistence_bars" in market:
        pb = market["persistence_bars"]
        if not isinstance(pb, int) or pb < 1:
            errors.append(f"persistence_bars must be int >= 1, got {pb}")

    if "deployment_target" in market:
        dt = market["deployment_target"]
        if not isinstance(dt, (int, float)) or not (0.0 <= dt <= 1.0):
            errors.append(f"deployment_target must be float in [0, 1], got {dt}")

    if "trade_slots" in market:
        ts = market["trade_slots"]
        if not isinstance(ts, int) or ts not in {0, 1, 2, 3}:
            errors.append(f"trade_slots must be int in {{0, 1, 2, 3}}, got {ts}")

    if "breadth_divergence" in market:
        if not isinstance(market["breadth_divergence"], bool):
            errors.append(f"breadth_divergence must be bool, got {type(market['breadth_divergence'])}")

    if "systemic_fragility" in market:
        sf = market["systemic_fragility"]
        if not isinstance(sf, (int, float)) or not (0.0 <= sf <= 1.0):
            errors.append(f"systemic_fragility must be float in [0, 1], got {sf}")

    # --- volatility ---
    vol = data["volatility"]
    if "garch_1d_forecast_pct" not in vol:
        errors.append("Missing volatility key: 'garch_1d_forecast_pct'")
    elif not isinstance(vol["garch_1d_forecast_pct"], (int, float)) or vol["garch_1d_forecast_pct"] <= 0:
        errors.append(f"garch_1d_forecast_pct must be positive float, got {vol['garch_1d_forecast_pct']}")

    if "garch_5d_forecast_pct" not in vol:
        errors.append("Missing volatility key: 'garch_5d_forecast_pct'")
    elif not isinstance(vol["garch_5d_forecast_pct"], (int, float)) or vol["garch_5d_forecast_pct"] <= 0:
        errors.append(f"garch_5d_forecast_pct must be positive float, got {vol['garch_5d_forecast_pct']}")

    if "vix_implied_term_structure" in vol:
        if vol["vix_implied_term_structure"] not in {"contango", "backwardation"}:
            errors.append(f"vix_implied_term_structure must be 'contango' or 'backwardation', "
                          f"got '{vol['vix_implied_term_structure']}'")

    # --- universe_ranking ---
    ur = data["universe_ranking"]
    if not isinstance(ur, list) or len(ur) < 1:
        errors.append("universe_ranking must be a non-empty list")
    else:
        for idx, item in enumerate(ur):
            if not isinstance(item, dict):
                errors.append(f"universe_ranking[{idx}] must be a dict")
            else:
                if "symbol" not in item:
                    errors.append(f"universe_ranking[{idx}] missing 'symbol'")
                if "ml_score" not in item:
                    errors.append(f"universe_ranking[{idx}] missing 'ml_score'")

    # --- universe_weights ---
    uw = data["universe_weights"]
    if not isinstance(uw, dict) or len(uw) < 1:
        errors.append("universe_weights must be a non-empty dict")

    # --- sectors ---
    sectors = data["sectors"]
    if not isinstance(sectors, dict) or len(sectors) < 1:
        errors.append("sectors must be a non-empty dict")
    else:
        valid_sector_regimes = {"Trend", "Choppy", "Bear"}
        for ticker, sec_data in sectors.items():
            if "regime" not in sec_data:
                errors.append(f"sectors['{ticker}'] missing 'regime'")
            elif sec_data["regime"] not in valid_sector_regimes:
                errors.append(f"sectors['{ticker}'].regime = '{sec_data['regime']}', "
                              f"expected one of {valid_sector_regimes}")
            if "score" not in sec_data:
                errors.append(f"sectors['{ticker}'] missing 'score'")

    return errors


def main() -> None:
    # Determine file path
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        json_path = cfg.INSIGHTS_PATH

    if not json_path.exists():
        print(f"FAIL: {json_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(json_path, "r") as f:
        data = json.load(f)

    errors = validate(data)

    if errors:
        print(f"SCHEMA VALIDATION FAILED ({len(errors)} violations):", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Schema validation passed. ✓ ({json_path.name})")
        sys.exit(0)


if __name__ == "__main__":
    main()
