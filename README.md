# ML Pipeline — Local Research Engine

Nightly quantitative pipeline that generates `ml-insights.json` for the cloud-based Trading Bot. Runs on a local Ubuntu workstation, pushes JSON artifacts via Git, and shuts down.

## Architecture

| Phase | Script | Purpose |
|-------|--------|---------|
| 1 | `phase1_data_pipeline.py` | Fetch OHLCV data (Yahoo Finance), engineer features (RV, SMA, volume) |
| 2 | `phase2_regime_model.py` | HMM regime classifier → Bull / Neutral / Caution / Defensive |
| 3 | `phase3_volatility_model.py` | GARCH(1,1) → 1-day and 5-day annualized volatility forecast |
| 5 | `phase5_ranking_model.py` | XGBoost sector ranking → top 3 sector ETFs by predicted forward return |
| 6 | `phase6_gnn_topology.py` | Graph-based systemic fragility score (sector correlation density) |
| 4 | `generate_insights.py` | Orchestrator: runs phases 1–6, assembles `output/ml-insights.json` |
| — | `validate_schema.py` | Schema conformance validator (pre-push gate) |
| — | `evaluate_models.py` | Backtesting: Spearman rank correlation, hit rate, regime forward returns |

## Output

`output/ml-insights.json` — the sole deliverable. Treat this file like a **public API contract**. The cloud Trading Bot reads it at repo root each pre-market routine.

## Quick Start

```bash
# Install dependencies (in venv)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline
python generate_insights.py

# Or run the cron-ready version (includes git push + shutdown)
./run_pipeline.sh
```

## Cron Setup

```cron
# Run at 4:30 AM ET (10:30 CET) on weekdays
30 10 * * 1-5 /home/radu/Documents/Antigravity/Trader/run_pipeline.sh
```

## Sudo Requirement

`run_pipeline.sh` calls `sudo shutdown -h now` after a successful push. This requires passwordless sudo for the shutdown command:

```bash
echo "radu ALL=(ALL) NOPASSWD: /sbin/shutdown" | sudo tee /etc/sudoers.d/ml-shutdown
```

Without this, `sudo` will block on a password prompt under cron and the machine stays on.

## Configuration

All tunable parameters live in [`config.py`](config.py) — no magic numbers in scripts. Key settings:

- **Tickers:** `CORE_TICKERS`, `SECTOR_TICKERS`
- **Regime:** `N_REGIMES=4`, `REGIME_PERSISTENCE_BARS=3`
- **GARCH:** `GARCH_P=1`, `GARCH_Q=1`, `GARCH_DIST="t"`
- **Seeds:** `GMM_RANDOM_STATE=42`
