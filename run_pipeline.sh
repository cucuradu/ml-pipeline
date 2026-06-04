#!/bin/bash
# ============================================================
# ML Pipeline — run, validate, and PUBLISH to the trading-bot repo.
#
# The consumer reads docs/ml-insights.json from the *trading-bot* repo, NOT from
# this one. The old script only committed here, which is why the bot ran on
# rule_fallback for days (memory/MACRO-FRAMEWORK.md shows 26–99h staleness).
# This script closes that gap: it copies the validated payload into the bot repo
# and pushes there with a distinct identity, per docs/ml-insights-publisher.md.
#
# Schedule it 7 days/week via launchd (markets-closed days re-emit Friday's
# inference; the 24h freshness check otherwise forces a Monday fallback). See
# launchd/com.cucuradu.ml-insights.plist in this repo.
#
# Config via env (no hardcoded machine paths):
#   ML_BOT_REPO       path to the trading-bot clone (default: the Mac path)
#   ML_BOT_IDENTITY   "Name <email>" for the producer commit
#   DRY_RUN           =1 to run + validate + stage the file but NOT commit/push
#   ML_SHUTDOWN_AFTER  =1 to power off after a successful push (Ubuntu only)
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_REPO="${ML_BOT_REPO:-/Users/raducucu/Developer/trading-bot}"
BOT_IDENTITY="${ML_BOT_IDENTITY:-ml-pipeline-bot <bot@cucuradu.local>}"
LOG_FILE="${SCRIPT_DIR}/logs/cron.log"
PY="${SCRIPT_DIR}/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

cd "$SCRIPT_DIR" || { echo "FATAL: cannot cd to $SCRIPT_DIR"; exit 1; }
mkdir -p logs
log "================ pipeline run ================"

# 1. Run the full pipeline.
if ! "$PY" generate_insights.py >> "$LOG_FILE" 2>&1; then
    log "PIPELINE FAILED — not publishing."
    exit 1
fi

# 2. Hard gate: the payload MUST satisfy the consumer's frozen contract. Cross-
#    check against the consumer's OWN validator so schema drift is caught here,
#    not silently in the bot's rule_fallback path.
export CONSUMER_REPO="$BOT_REPO"
if ! "$PY" validate_payload.py "${SCRIPT_DIR}/output/ml-insights.json" --cross-check >> "$LOG_FILE" 2>&1; then
    log "VALIDATION FAILED — refusing to publish a payload the bot would reject."
    exit 1
fi
log "Payload validated (contract + consumer cross-check)."

# 3. Publish into the trading-bot repo.
if [ ! -d "$BOT_REPO/.git" ]; then
    log "WARN: bot repo not found at '$BOT_REPO' (set ML_BOT_REPO). Skipped publish."
    exit 0
fi
cd "$BOT_REPO" || { log "FATAL: cannot cd to bot repo"; exit 1; }

# Dry run: report what WOULD publish, but touch nothing (no pull/write/commit/push).
if [ "${DRY_RUN:-0}" = "1" ]; then
    if [ -f docs/ml-insights.json ] && diff -q docs/ml-insights.json "${SCRIPT_DIR}/output/ml-insights.json" >/dev/null 2>&1; then
        log "DRY_RUN: payload identical to bot repo's docs/ml-insights.json — nothing to publish."
    else
        log "DRY_RUN: would publish a new/updated docs/ml-insights.json to $BOT_REPO (no write/commit/push)."
    fi
    exit 0
fi

git pull --rebase --autostash origin main >> "$LOG_FILE" 2>&1 || { log "bot-repo pull failed"; exit 1; }
mkdir -p docs
cp "${SCRIPT_DIR}/output/ml-insights.json" docs/ml-insights.json
[ -f "${SCRIPT_DIR}/output/history.jsonl" ] && cp "${SCRIPT_DIR}/output/history.jsonl" docs/history.jsonl

# 4. Idempotent commit — skip if nothing changed (followup: no git-log noise).
if git diff --quiet -- docs/ml-insights.json docs/history.jsonl 2>/dev/null; then
    log "No payload change since last publish — skipping commit."
else
    git add docs/ml-insights.json docs/history.jsonl
    git -c user.name="${BOT_IDENTITY% <*}" -c user.email="$(echo "$BOT_IDENTITY" | sed -E 's/.*<(.*)>/\1/')" \
        commit -m "ml-insights update $(date +%F)" >> "$LOG_FILE" 2>&1
    if ! git push origin HEAD:main >> "$LOG_FILE" 2>&1; then
        git pull --rebase --autostash origin main >> "$LOG_FILE" 2>&1 && git push origin HEAD:main >> "$LOG_FILE" 2>&1 \
            || { log "PUSH FAILED — bot will fall back until this is fixed."; exit 1; }
    fi
    log "Published fresh ml-insights.json to the bot repo."
fi

[ "${ML_SHUTDOWN_AFTER:-0}" = "1" ] && { log "Shutting down (ML_SHUTDOWN_AFTER=1)."; sudo shutdown -h now; }
log "Done."
