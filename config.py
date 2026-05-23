"""
Central configuration for the ML pipeline.
All tunable parameters live here — no magic numbers buried in scripts.
"""

from pathlib import Path
from datetime import datetime, timedelta

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for d in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Data Fetching
# ============================================================
# Core index, volatility gauges, equal-weight proxy, and macro yields
CORE_TICKERS = ["SPY", "^VIX", "^VIX3M", "RSP", "^TNX", "^IRX"]

# Mapping: yfinance symbol → clean display name (used in outputs/JSON)
TICKER_DISPLAY_NAMES = {
    "^VIX": "VIX", 
    "^VIX3M": "VIX3M",
    "^TNX": "TNX",
    "^IRX": "IRX"
}

# Sector ETFs (SPDR Select Sector)
SECTOR_TICKERS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLV",   # Health Care
    "XLE",   # Energy
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "XLC",   # Communication Services
]

ALL_TICKERS = CORE_TICKERS + SECTOR_TICKERS

# Lookback period: 5 years of daily bars
LOOKBACK_YEARS = 5
END_DATE = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=LOOKBACK_YEARS * 365)).strftime("%Y-%m-%d")

# ============================================================
# Feature Engineering
# ============================================================
# Realized volatility rolling windows (trading days)
RV_WINDOWS = [5, 10, 21, 63]          # 1wk, 2wk, 1mo, 3mo

# Simple Moving Average periods for distance features
SMA_PERIODS = [20, 50, 200]

# Volume trend rolling window
VOLUME_TREND_WINDOW = 20

# ============================================================
# Regime Classifier (Phase 2)
# ============================================================
N_REGIMES = 4                          # Bull, Neutral, Caution, Defensive
REGIME_PERSISTENCE_BARS = 3            # Days before declaring regime change
GMM_RANDOM_STATE = 42
RUN_OPTUNA = True  # Set to True to dynamically optimize XGBoost hyperparameters
HMM_N_ITER = 200

from typing import Literal

# ============================================================
# GARCH Volatility Model (Phase 3)
# ============================================================
GARCH_P = 1
GARCH_Q = 1
GARCH_DIST: Literal["normal", "t"] = "t"  # Changed to "t" for fat tails

# ============================================================
# Output — The Handshake Contract
# ============================================================
INSIGHTS_FILENAME = "ml-insights.json"
INSIGHTS_PATH = OUTPUT_DIR / INSIGHTS_FILENAME
MODEL_VERSION = "hmm-garch-v1.0"

# ============================================================
# Regime → Deployment Mapping (hardcoded contract)
# ============================================================
REGIME_DEPLOYMENT_MAP = {
    "Bull":      {"deployment_target": 0.85, "trade_slots": 3},
    "Neutral":   {"deployment_target": 0.75, "trade_slots": 2},
    "Caution":   {"deployment_target": 0.50, "trade_slots": 1},
    "Defensive": {"deployment_target": 0.00, "trade_slots": 0},
}

# ============================================================
# Git Automation (Phase 4)
# ============================================================
GIT_REMOTE = "origin"
GIT_BRANCH = "main"
