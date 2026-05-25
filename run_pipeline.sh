#!/bin/bash
# ============================================================
# ML Pipeline Automation Hook
# Runs the full pipeline, validates output, commits JSON to both
# the ml-pipeline repo AND the Trading Bot repo, then shuts down.
#
# REQUIREMENTS:
#   • Passwordless sudo for shutdown:
#       echo "radu ALL=(ALL) NOPASSWD: /sbin/shutdown" | sudo tee /etc/sudoers.d/ml-shutdown
#     Without this, `sudo shutdown -h now` will block on a password
#     prompt under cron and the PC stays on, wasting power overnight.
#
#   • SSH key with push access to both repos:
#       - cucuradu/ml-pipeline (this repo)
#       - cucuradu/trading-bot (the cloud consumer)
#
# CRON EXAMPLE (run at 4:30 AM ET / 10:30 CET on weekdays):
#   30 10 * * 1-5 /home/radu/Documents/Antigravity/Trader/run_pipeline.sh
# ============================================================

set -euo pipefail

# Export PATH so cron can find python, git, ssh, etc.
export PATH="/home/radu/Documents/Antigravity/Trader/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Pipeline directory (absolute)
PIPELINE_DIR="/home/radu/Documents/Antigravity/Trader"
LOG_FILE="${PIPELINE_DIR}/logs/cron.log"

# Trading Bot cross-repo delivery config (mirrors config.py)
TRADING_BOT_REPO="git@github.com:cucuradu/trading-bot.git"
TRADING_BOT_CLONE_DIR="/tmp/trading-bot-delivery"
BOT_NAME="ml-pipeline-bot"
BOT_EMAIL="bot@cucuradu.local"

echo "========================================" >> "$LOG_FILE"
echo "Pipeline run: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# --------------------------------------------------------
# Pre-flight: check sudo is passwordless
# --------------------------------------------------------
if ! sudo -n true 2>/dev/null; then
    echo "[$(date)] WARNING: sudo requires a password. Auto-shutdown will HANG under cron." >> "$LOG_FILE"
    echo "  Fix: echo 'radu ALL=(ALL) NOPASSWD: /sbin/shutdown' | sudo tee /etc/sudoers.d/ml-shutdown" >> "$LOG_FILE"
fi

# 1. Navigate to the pipeline directory
cd "$PIPELINE_DIR" || { echo "FATAL: Cannot cd to $PIPELINE_DIR" >> "$LOG_FILE"; exit 1; }

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Execute the full pipeline (Phases 1-6 + insights assembly)
echo "[$(date)] Starting ML Pipeline..." >> "$LOG_FILE"
if ! python generate_insights.py >> "$LOG_FILE" 2>&1; then
    echo "[$(date)] PIPELINE FAILED. Aborting (no shutdown)." >> "$LOG_FILE"
    exit 1
fi

# 4. Schema validation gate — catch contract drift before pushing
echo "[$(date)] Validating output schema..." >> "$LOG_FILE"
if ! python validate_schema.py >> "$LOG_FILE" 2>&1; then
    echo "[$(date)] SCHEMA VALIDATION FAILED. Aborting (no push, no shutdown)." >> "$LOG_FILE"
    exit 1
fi
echo "[$(date)] Schema validation passed. ✓" >> "$LOG_FILE"

# 5. Commit to ml-pipeline repo (this repo)
echo "[$(date)] Committing to ml-pipeline..." >> "$LOG_FILE"
git add output/ml-insights.json output/history.jsonl
commit_msg="Auto-update ml-insights: $(date +'%Y-%m-%d')"
git commit -m "$commit_msg" >> "$LOG_FILE" 2>&1 || true

# 6. Push ml-pipeline repo
echo "[$(date)] Pushing ml-pipeline to GitHub..." >> "$LOG_FILE"
if ! git push origin main >> "$LOG_FILE" 2>&1; then
    echo "[$(date)] ML-PIPELINE PUSH FAILED. Not shutting down so you can retry." >> "$LOG_FILE"
    exit 1
fi

# 7. Cross-repo delivery: push ml-insights.json to Trading Bot repo
echo "[$(date)] Delivering ml-insights.json to Trading Bot repo..." >> "$LOG_FILE"
rm -rf "$TRADING_BOT_CLONE_DIR"

if git clone --depth 1 "$TRADING_BOT_REPO" "$TRADING_BOT_CLONE_DIR" >> "$LOG_FILE" 2>&1; then
    mkdir -p "${TRADING_BOT_CLONE_DIR}/docs"
    cp "${PIPELINE_DIR}/output/ml-insights.json" "${TRADING_BOT_CLONE_DIR}/docs/ml-insights.json"
    cp "${PIPELINE_DIR}/output/history.jsonl" "${TRADING_BOT_CLONE_DIR}/docs/history.jsonl"
    cd "$TRADING_BOT_CLONE_DIR"

    # Use distinct bot identity (schema contract requirement)
    git config user.name "$BOT_NAME"
    git config user.email "$BOT_EMAIL"

    git add docs/ml-insights.json docs/history.jsonl
    git commit -m "ML update: $(date +'%Y-%m-%d') [automated]" >> "$LOG_FILE" 2>&1 || true

    if ! git push origin main >> "$LOG_FILE" 2>&1; then
        echo "[$(date)] TRADING BOT PUSH FAILED. Not shutting down." >> "$LOG_FILE"
        rm -rf "$TRADING_BOT_CLONE_DIR"
        exit 1
    fi

    cd "$PIPELINE_DIR"
    rm -rf "$TRADING_BOT_CLONE_DIR"
    echo "[$(date)] Trading Bot delivery successful. ✓" >> "$LOG_FILE"
else
    echo "[$(date)] WARNING: Could not clone Trading Bot repo. Skipping cross-repo delivery." >> "$LOG_FILE"
    echo "  Verify TRADING_BOT_REPO URL and SSH key access." >> "$LOG_FILE"
    # Don't exit — the ml-pipeline push already succeeded
fi

echo "[$(date)] Pipeline successful. Shutting down." >> "$LOG_FILE"

# 8. Shut down the PC (only after confirmed push)
sudo shutdown -h now
