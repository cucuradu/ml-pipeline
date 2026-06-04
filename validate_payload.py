#!/usr/bin/env python3
"""Pre-push validator for ml-insights.json (producer side).

The producer (this repo) and the consumer (the Claude trading bot) are developed
independently. If we push a payload the consumer rejects, the bot silently falls
back to its rule-based path and our whole pipeline contributes nothing. This
script is the gate: `run_pipeline.sh` runs it BEFORE committing, and refuses to
push on failure.

It is a faithful VENDORED copy of the consumer's validation rules in
`Trading Bot/scripts/ml_insights.py:validate()` and `docs/ml-insights-schema.md`
(rules 1-10), so it works on the headless producer host without the bot repo
checked out. If the consumer repo IS present (set CONSUMER_REPO), `--cross-check`
imports the real validator and asserts both agree — drift insurance.

Beyond the hard contract, it emits PRODUCER WARNINGS for the exact anomalies the
consumer flagged in docs/ml-pipeline-followups.md (confidence pinned at 0/1,
implausible fragility, empty ranking) so a regression is caught before it ships.

Usage:
  python validate_payload.py [PATH]                 # default: output/ml-insights.json
  python validate_payload.py PATH --cross-check      # also run the consumer's validator
  python validate_payload.py PATH --strict-warnings  # exit non-zero on warnings too
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config as cfg

FRESHNESS_HOURS = 24
VALID_MARKET_REGIMES = {"Bull", "Neutral", "Caution", "Defensive"}
VALID_SECTOR_REGIMES = {"Trend", "Choppy", "Bear"}


def _parse_iso_utc(s) -> datetime | None:
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def validate(payload: dict, *, now: datetime | None = None) -> dict:
    """Hard contract check. Returns {ok, reason}. Mirrors the consumer 1:1."""
    now = now or datetime.now(timezone.utc)

    if not isinstance(payload, dict):
        return {"ok": False, "reason": "payload is not a JSON object"}

    gen_dt = _parse_iso_utc(payload.get("generated_at"))
    if gen_dt is None:
        return {"ok": False, "reason": "generated_at missing or unparseable"}
    age = now - gen_dt
    if age > timedelta(hours=FRESHNESS_HOURS):
        return {"ok": False, "reason": f"stale: {age.total_seconds() / 3600:.1f}h old"}
    if age < timedelta(seconds=-300):
        return {"ok": False, "reason": f"future-dated: {-age.total_seconds():.0f}s ahead"}

    market = payload.get("market")
    if not isinstance(market, dict):
        return {"ok": False, "reason": "market block missing or not an object"}

    regime = market.get("regime")
    if regime not in VALID_MARKET_REGIMES:
        return {"ok": False, "reason": f"market.regime {regime!r} invalid"}

    dep = market.get("deployment_target")
    if not isinstance(dep, (int, float)) or isinstance(dep, bool) or not (0.0 <= float(dep) <= 1.0):
        return {"ok": False, "reason": f"market.deployment_target {dep!r} not in [0,1]"}

    slots = market.get("trade_slots")
    if not isinstance(slots, (int, float)) or isinstance(slots, bool):
        return {"ok": False, "reason": "market.trade_slots missing or not numeric"}

    frag = market.get("systemic_fragility")
    if frag is not None:
        if not isinstance(frag, (int, float)) or isinstance(frag, bool) or not (0.0 <= float(frag) <= 1.0):
            return {"ok": False, "reason": f"market.systemic_fragility {frag!r} not in [0,1]"}

    sectors = payload.get("sectors") or {}
    if isinstance(sectors, dict):
        for sym, info in sectors.items():
            if isinstance(info, dict) and info.get("regime") not in VALID_SECTOR_REGIMES:
                return {"ok": False, "reason": f"sectors.{sym}.regime {info.get('regime')!r} invalid"}

    return {"ok": True, "reason": None}


def producer_warnings(payload: dict) -> list[str]:
    """Soft checks for the bugs the consumer flagged. Non-fatal by default."""
    warns: list[str] = []
    market = payload.get("market") or {}

    conf = market.get("confidence")
    if isinstance(conf, (int, float)) and conf in (0.0, 1.0):
        warns.append(f"confidence pinned at {conf} (followup S1/P7 — expect ~0.6-0.95)")

    frag = market.get("systemic_fragility")
    if isinstance(frag, (int, float)) and not (0.30 <= float(frag) <= 0.90):
        warns.append(f"systemic_fragility={frag:.3f} outside plausible 0.30-0.90 band (followup S3/P1)")

    vol = payload.get("volatility") or {}
    v1, v5 = vol.get("garch_1d_forecast_pct"), vol.get("garch_5d_forecast_pct")
    if isinstance(v1, (int, float)) and isinstance(v5, (int, float)) and abs(v1 - v5) < 0.05:
        warns.append(f"garch 1d≈5d ({v1} vs {v5}) — horizon arithmetic suspect (followup S4)")

    ranking = payload.get("universe_ranking") or []
    if not ranking:
        warns.append("universe_ranking empty — stock selection gets no ML signal")
    else:
        unknown = [r.get("symbol") for r in ranking
                   if isinstance(r, dict) and r.get("symbol") not in cfg.TRADEABLE_UNIVERSE]
        if unknown:
            warns.append(f"universe_ranking has non-universe symbols: {unknown[:5]}")

    return warns


def _cross_check(payload: dict) -> str | None:
    """Import the real consumer validator and assert agreement. None = agree."""
    repo = os.environ.get("CONSUMER_REPO", "/Users/raducucu/Developer/trading-bot")
    scripts = Path(repo) / "scripts"
    if not (scripts / "ml_insights.py").exists():
        return f"consumer repo not found at {repo} (set CONSUMER_REPO); skipped"
    sys.path.insert(0, str(scripts))
    try:
        import ml_insights as consumer  # type: ignore
    except Exception as e:  # pragma: no cover
        return f"could not import consumer validator: {e}"
    ours = validate(payload)["ok"]
    theirs = consumer.validate(payload)["ok"]
    if ours != theirs:
        return f"DRIFT: vendored ok={ours} but consumer ok={theirs} ({consumer.validate(payload)['reason']})"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default=str(cfg.INSIGHTS_PATH))
    ap.add_argument("--cross-check", action="store_true")
    ap.add_argument("--strict-warnings", action="store_true")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"FAIL: {path} not found")
        return 1
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"FAIL: malformed JSON: {e}")
        return 1

    result = validate(payload)
    if not result["ok"]:
        print(f"FAIL (contract): {result['reason']}")
        return 1
    print("PASS (contract): payload satisfies the frozen schema")

    if args.cross_check:
        drift = _cross_check(payload)
        if drift and drift.startswith("DRIFT"):
            print(f"FAIL (cross-check): {drift}")
            return 1
        print(f"cross-check: {'agree' if drift is None else drift}")

    warns = producer_warnings(payload)
    for w in warns:
        print(f"WARN: {w}")
    if warns and args.strict_warnings:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
