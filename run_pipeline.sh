#!/bin/bash
# ============================================================
# Phase 4 Automation Hook
# Runs the full ML pipeline, commits the new JSON deliverable,
# pushes to GitHub, and safely shuts down the machine.
# ============================================================

set -e # Exit immediately if a command exits with a non-zero status

# 1. Navigate to the pipeline directory
cd /home/radu/Documents/Antigravity/Trader

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Execute the full pipeline (Phases 1-4)
echo "Starting ML Pipeline..."
python generate_insights.py

# 4. Stage the generated JSON
echo "Staging ml-insights.json..."
git add output/ml-insights.json

# 5. Commit with today's date
# We use || true so the script doesn't fail if there are no changes
commit_msg="Auto-update ml-insights: $(date +'%Y-%m-%d')"
git commit -m "$commit_msg" || true

# 6. Push to the isolated research repository
echo "Pushing to GitHub..."
git push origin main

echo "Pipeline successful. Shutting down system."

# 7. Shut down the PC
sudo shutdown -h now
