#!/bin/bash
# ============================================================
# ML Pipeline Automation Hook
# Runs the full pipeline, commits JSON, pushes to GitHub,
# and shuts down the machine ONLY after a successful push.
# ============================================================

# Export PATH so cron can find python, git, ssh, etc.
export PATH="/home/radu/Documents/Antigravity/Trader/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Pipeline directory (absolute)
PIPELINE_DIR="/home/radu/Documents/Antigravity/Trader"
LOG_FILE="${PIPELINE_DIR}/logs/cron.log"

echo "========================================" >> "$LOG_FILE"
echo "Pipeline run: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# 1. Navigate to the pipeline directory
cd "$PIPELINE_DIR" || { echo "FATAL: Cannot cd to $PIPELINE_DIR" >> "$LOG_FILE"; exit 1; }

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Execute the full pipeline (Phases 1-4)
echo "[$(date)] Starting ML Pipeline..." >> "$LOG_FILE"
if ! python generate_insights.py >> "$LOG_FILE" 2>&1; then
    echo "[$(date)] PIPELINE FAILED. Aborting (no shutdown)." >> "$LOG_FILE"
    exit 1
fi

# 4. Stage and commit
echo "[$(date)] Committing..." >> "$LOG_FILE"
git add output/ml-insights.json
commit_msg="Auto-update ml-insights: $(date +'%Y-%m-%d')"
git commit -m "$commit_msg" >> "$LOG_FILE" 2>&1 || true

# 5. Push — only proceed to shutdown if this succeeds
echo "[$(date)] Pushing to GitHub..." >> "$LOG_FILE"
if ! git push origin main >> "$LOG_FILE" 2>&1; then
    echo "[$(date)] PUSH FAILED. Not shutting down so you can retry." >> "$LOG_FILE"
    exit 1
fi

echo "[$(date)] Pipeline successful. Shutting down." >> "$LOG_FILE"

# 6. Shut down the PC (only after confirmed push)
sudo shutdown -h now
