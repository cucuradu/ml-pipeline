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
# Context tickers — NOT tradeable, used only as features for regime/macro/vol.
# (^VVIX = vol-of-vol; sharpens the crash flag and regime confidence.)
CORE_TICKERS = ["SPY", "^VIX", "^VIX3M", "^VVIX", "RSP", "^TNX", "^IRX"]

# Mapping: yfinance symbol → clean display name (used in outputs/JSON)
TICKER_DISPLAY_NAMES = {
    "^VIX": "VIX",
    "^VIX3M": "VIX3M",
    "^VVIX": "VVIX",
    "^TNX": "TNX",
    "^IRX": "IRX",
}

# Sector ETFs (SPDR Select Sector) — the 11 the regime/fragility models run on.
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

# Broad-index ETFs (cross-sector plays; excluded from sector-attribution math).
BROAD_ETFS = ["SPY", "QQQ", "IWM", "DIA"]

# Individual megacaps + thematic ETFs the bot may trade. MUST stay in lockstep
# with `Trading Bot/scripts/universe.py` — the consumer rejects any symbol it
# does not recognize. Universe changes are a strategy decision in BOTH repos.
STOCK_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "TSLA",
    "AMD", "MU", "ORCL", "CRM", "ADBE", "NOW", "INTU",
    # Consumer / Finance
    "JPM", "V", "MA", "COST", "WMT", "HD", "PG", "KO",
    "BAC", "GS", "MS", "SPGI", "BLK", "NKE", "MCD", "SBUX", "LOW",
    # Health / Industrial
    "UNH", "JNJ", "LLY", "CAT", "BA",
    "ABBV", "MRK", "TMO", "ABT", "DHR", "AMGN",
    "GE", "RTX", "LMT", "HON", "UNP", "DE",
    # Energy / Comms
    "XOM", "CVX", "NFLX", "DIS",
    # Thematic ETFs (treated as sector proxies)
    "SMH", "XBI",
]

# The full tradeable universe (70 names) — what the cross-sectional ranker scores.
TRADEABLE_UNIVERSE = sorted(set(SECTOR_TICKERS) | set(BROAD_ETFS) | set(STOCK_TICKERS))

# Everything to fetch from yfinance: tradeable names + non-tradeable context.
ALL_TICKERS = sorted(set(CORE_TICKERS) | set(TRADEABLE_UNIVERSE))

# Symbol → SPDR sector ETF (for relative-strength + sector-Bear gating).
# "BROAD" = index ETF with no single sector. Mirrors universe.py:SECTOR_OF.
SECTOR_OF = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK",
    "AMD": "XLK", "MU": "XLK", "ORCL": "XLK", "CRM": "XLK",
    "ADBE": "XLK", "NOW": "XLK", "INTU": "XLK", "SMH": "XLK",
    "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY",
    "NKE": "XLY", "MCD": "XLY", "SBUX": "XLY", "LOW": "XLY",
    "WMT": "XLP", "COST": "XLP", "PG": "XLP", "KO": "XLP",
    "JPM": "XLF", "V": "XLF", "MA": "XLF",
    "BAC": "XLF", "GS": "XLF", "MS": "XLF", "SPGI": "XLF", "BLK": "XLF",
    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV",
    "ABBV": "XLV", "MRK": "XLV", "TMO": "XLV", "ABT": "XLV",
    "DHR": "XLV", "AMGN": "XLV", "XBI": "XLV",
    "CAT": "XLI", "BA": "XLI",
    "GE": "XLI", "RTX": "XLI", "LMT": "XLI", "HON": "XLI", "UNP": "XLI", "DE": "XLI",
    "XOM": "XLE", "CVX": "XLE",
    # Sector ETFs map to themselves; broad ETFs to BROAD.
    **{etf: etf for etf in SECTOR_TICKERS},
    **{etf: "BROAD" for etf in BROAD_ETFS},
}

# FRED macro/credit series (keyless via pandas-datareader). Best free *leading*
# risk indicators — feed the regime model and the macro overlay block.
FRED_SERIES = {
    "dgs10": "DGS10",          # 10y Treasury yield
    "dgs3mo": "DGS3MO",        # 3m Treasury yield (curve = dgs10 - dgs3mo)
    "hy_oas": "BAMLH0A0HYM2",  # ICE BofA US High Yield OAS — credit stress
    "nfci": "NFCI",            # Chicago Fed National Financial Conditions Index
    "claims": "ICSA",          # Initial jobless claims (weekly)
}

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
# Bumped from hmm-garch-v1.0: real HMM posteriors, PCA fragility, 70-name ranker,
# multi-horizon GARCH, macro overlay. Rename score→forward_return_pct ships here.
MODEL_VERSION = "hmm-garch-v2.0"
# Additive schema version for the consumer's forward-compat validator (followup C4).
SCHEMA_VERSION = "1.1"

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
