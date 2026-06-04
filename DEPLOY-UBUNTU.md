# Deploying the ML pipeline on the Ubuntu host (RTX 5060 Ti)

The Ubuntu PC is the production producer. It runs this pipeline nightly and
**pushes only `ml-insights.json` to the `trading-bot` repo** (the bot reads it
from GitHub). The pipeline *code* lives in `cucuradu/ml-pipeline` (this repo);
the Ubuntu box pulls it from `main`.

> Two repos, two roles:
> - `cucuradu/ml-pipeline` → the pipeline code (pull `main` to deploy/update).
> - `cucuradu/trading-bot` → receives the daily `docs/ml-insights.json` (push).

## 1. One-time setup

```bash
# a) Pipeline code
git clone https://github.com/cucuradu/ml-pipeline.git ~/ml-pipeline
cd ~/ml-pipeline && git checkout main

# b) Python env + deps
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# c) The consumer repo the payload is published INTO (separate clone)
git clone https://github.com/cucuradu/trading-bot.git ~/trading-bot

# d) Git identity + credentials that can PUSH to trading-bot.
#    Use a PAT (repo scope) or a deploy key with write access to trading-bot.
git config --global credential.helper store   # or use SSH remotes
# first manual push will prompt once and cache the PAT
```

## 2. Configure via env (no hardcoded Mac paths)

`run_pipeline.sh` reads these — put them in the cron entry or a sourced file:

```bash
export ML_BOT_REPO="$HOME/trading-bot"                 # the consumer clone above
export ML_BOT_IDENTITY="ml-pipeline-bot <bot@cucuradu.local>"
export CONSUMER_REPO="$HOME/trading-bot"               # enables validator cross-check
# export ML_SHUTDOWN_AFTER=1                           # power off after a good push (optional)
```

## 3. Test before scheduling (writes nothing)

```bash
cd ~/ml-pipeline
DRY_RUN=1 ML_BOT_REPO="$HOME/trading-bot" CONSUMER_REPO="$HOME/trading-bot" \
  bash run_pipeline.sh
tail -n 20 logs/cron.log    # expect: validated (contract + consumer cross-check); DRY_RUN would publish
```

Then one real publish:

```bash
ML_BOT_REPO="$HOME/trading-bot" CONSUMER_REPO="$HOME/trading-bot" bash run_pipeline.sh
```

## 4. Schedule (cron, 7 days/week)

Markets-closed days simply re-emit the last inference so the bot's 24h freshness
check never trips. A small wrapper self-updates the pipeline code first.

```bash
crontab -e
```

```cron
# 06:30 daily: update pipeline code, run, validate, publish payload to trading-bot
30 6 * * *  cd $HOME/ml-pipeline && git pull --rebase origin main >> logs/cron.log 2>&1; \
            ML_BOT_REPO=$HOME/trading-bot CONSUMER_REPO=$HOME/trading-bot \
            ML_BOT_IDENTITY='ml-pipeline-bot <bot@cucuradu.local>' \
            bash run_pipeline.sh
```

## 5. What a run does (and its safety gates)

1. `generate_insights.py` — full pipeline (chunked yfinance fetch + keyless FRED).
2. `validate_payload.py --cross-check` — contract **and** the consumer's own
   `scripts/ml_insights.py:validate()` must both pass, or it refuses to publish.
3. Publish — `git pull --rebase --autostash` the trading-bot clone, copy
   `output/ml-insights.json` + `history.jsonl` into `docs/`, commit as
   `ML_BOT_IDENTITY`, push to `trading-bot` `main`. Idempotent: skips the commit
   when the payload is unchanged.

If anything fails, it exits non-zero and publishes nothing — the bot just stays
on its rule-based fallback until the next good run.

## 6. Health check

The bot treats the payload as stale after 24h. To confirm the host is healthy:

```bash
# on Ubuntu
git -C ~/trading-bot log -1 --format='%ci %an' -- docs/ml-insights.json
```

A timestamp older than ~26h means the Ubuntu run is failing — check
`~/ml-pipeline/logs/cron.log`.
