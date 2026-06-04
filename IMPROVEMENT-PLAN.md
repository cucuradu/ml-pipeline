# ML Pipeline — Future Improvement Plan

Captured 2026-06-04 after a full audit of the pipeline **and** its only consumer,
the Claude trading bot at `/Users/raducucu/Developer/trading-bot`.

## Why this plan exists (context)

The producer (this repo) feeds one consumer: an autonomous Claude bot that
paper-trades **70 individual stocks** on Alpaca (sector-momentum + ATR trailing
stops + risk gates). The bot reads `docs/ml-insights.json` from its own repo; if
that file is missing or >24h old it silently falls back to a rule-based regime
(VIX + SPY-200SMA) and its **own 8-factor screener** (`scripts/screener.py`).

Two findings reorder everything:

1. **Delivery was the real failure, not the models.** The bot's live
   `ml-insights.json` was **227h (9.5 days) stale** — every `MACRO-FRAMEWORK.md`
   / `RESEARCH-LOG.md` entry for two weeks reads `rule_fallback … local PC drift`.
   The bot has effectively **never used this pipeline**. Fixed in **Tier 0** (see
   `run_pipeline.sh` + launchd job) — a mediocre signal delivered daily beats a
   brilliant one that's 99h stale.

2. **Do not compete on stock-picking.** On the returns-based scoreboard
   (`evaluate_models.py`) the bot's screener beats our ML ranker on every metric:

   | strategy | Sharpe | CAGR | maxDD | rank IC |
   |---|---|---|---|---|
   | screener_baseline | **1.12** | 28.6% | -27.3% | 0.009 |
   | spy_buy_hold | 0.89 | 15.0% | -22.1% | — |
   | ml_ranker (ours) | 0.84 | 21.5% | -34.5% | 0.008 |

   **Decision: the screener owns `universe_ranking`.** Our ranker is informational
   only and is not worth tuning unless it beats the screener as a *risk overlay*
   (Tier 2). For a Sharpe objective, our edge is the market-level risk brief the
   screener and Gemini cannot produce — regime, macro/credit, breadth, vol — not
   yet-another momentum ranker.

The objective is **risk-adjusted return (Sharpe)**: the dominant levers are
exposure management (regime → deployment) and risk-based sizing (vol), not
forward-return prediction.

---

## Tier 0 — Reliable delivery (DONE, this change)

- `run_pipeline.sh`: run → validate (contract **+ consumer cross-check**) →
  publish into `/Users/raducucu/Developer/trading-bot/docs/ml-insights.json` with
  a distinct git identity. `DRY_RUN=1` proves the path without writing/pushing.
- `validate_payload.py`: vendored copy of the consumer's validator; `--cross-check`
  imports the bot's real `scripts/ml_insights.py:validate()` and asserts agreement.
- `launchd/com.cucuradu.ml-insights.plist`: daily 06:30 schedule (macOS).
- **Verified:** consumer `cross-check: agree`; payload passes the frozen contract.

**Open follow-up:** the first real publish is an outward push to
`github.com/cucuradu/trading-bot` `main` — left for explicit user approval.

---

## Tier 1 — The quant brief the bot explicitly asked for

Source: the bot's own `memory/BACKLOG.md` ("third regime input beyond VIX+200SMA…
**preferred: have the local PC fetch + include in ml-insights.json**") and the
`docs/ml-insights-schema.md` reserved fields. Ranked by Sharpe impact.

### 1.1 Regime quality + stable labels  *(biggest exposure lever)*
- **Pin the state→label mapping after the first fit.** BACKLOG: "avoid phantom
  regime flips on retrain." Today `phase2_regime_model.py:map_states_to_regimes`
  re-derives Bull/Neutral/Caution/Defensive every run from cluster stats — a
  retrain can silently swap labels and flip the bot's posture. Persist the
  mapping (e.g. `models/regime_label_map.json`) and reuse it; only remap on an
  explicit `--remap`.
- **Validate the regime actually beats the rule.** Backtest deployment driven by
  HMM-regime vs the VIX+200SMA fallback on the scoreboard. Note the audit found
  regimes **non-monotonic on forward returns** (Defensive had the *highest* 5d
  fwd return — a mean-reversion/"buy-fear" effect), so the current
  `REGIME_DEPLOYMENT_MAP` (Defensive→0%) may be fighting the data. Re-examine the
  map for a Sharpe objective; ship only if it beats the rule.
- Calibrated `confidence` is already done (typicality × stickiness; ~0.64, no
  longer pinned at 1.0).

### 1.2 Macro / credit overlay in the payload  *(replaces hand-derived macro)*
- `phase7_macro.py` already computes yield curve, HY OAS, NFCI, claims →
  `macro_risk` + a `crash_risk` flag (keyless FRED CSV, with retries).
- **Surface it in the contract** as additive blocks (`macro`, `crash_risk`) — the
  consumer tolerates unknown fields. This replaces the macro paragraph the bot
  writes by hand via Gemini every morning (`MACRO-FRAMEWORK.md`), saving tokens
  and making it consistent + quantified.
- Wire `crash_risk` / `macro_risk` into the deployment haircut and prove the
  haircut **cuts drawdown** on the scoreboard (the -34%/-27% DDs are too large
  for a Sharpe-first system).

### 1.3 Market breadth + divergence  *(bot tracks this by hand today)*
- Emit `% of universe above 50/200-day SMA` and a breadth-vs-price divergence
  flag. BACKLOG line 182; and the bot is **flashing this right now** (2026-06-03:
  "S&P +13.1% vs breadth −0.055 — dangerous divergence over 60d").
- Feeds `market.breadth_divergence` (already in schema) with a real computation
  instead of the SPY/RSP heuristic.

### 1.4 GARCH vol → vol-aware sizing inputs
- `garch_1d` / `garch_5d` are produced (1d≠5d verified). BACKLOG: "vol-aware
  sizing later in `scripts/sizing.py`"; schema reserves `volatility.*`.
- Provide a clean per-name / portfolio forward-vol the bot's half-Kelly sizing
  can consume to right-size positions (forward vol beats the bot's backward ATR).

---

## Tier 2 — ML ranker, only if it earns it (optional)

The screener wins today. Tune the ML ranker **only** as a *risk overlay* on the
screener's momentum signal (regime-conditioned, vol-adjusted targets, conformal
shrinkage), and ship **only if** it beats `screener_baseline` on the scoreboard
(Sharpe + drawdown, across ≥2 windows). Otherwise leave `universe_ranking` to the
screener and stop spending effort here.

Concrete experiments worth trying before giving up:
- Target = vol-scaled / rank-normalized forward return (not raw 5d return).
- Regime-conditional models (separate fit per market regime).
- Blend ML score with the screener factors and let the backtest pick the weight.
- Longer horizon (the daily-bar 5d target has near-zero IC — too noisy).

---

## Guardrails for all future work

- Every signal must clear `evaluate_models.py` (Sharpe / drawdown / IC vs the
  screener + SPY baselines) **before** it ships. No metric, no merge.
- Schema is frozen (`docs/ml-insights-schema.md`): additive fields only; a
  breaking change needs a `model_version` bump + a backwards-compatible window.
- Validate (contract + consumer cross-check) before every publish.
