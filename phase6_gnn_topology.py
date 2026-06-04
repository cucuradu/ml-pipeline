#!/usr/bin/env python3
"""
Phase 6 — Systemic Fragility (PCA Absorption Ratio)
===================================================
Replaces the earlier "graph density" heuristic, which was mislabeled a GNN and
produced an implausible 0.018 (the consumer's #1 flagged anomaly, followup S3/P1).

The Absorption Ratio (Kritzman, Li, Page & Rigobon 2010) is the share of total
cross-sector variance explained by the first principal component:

    AR = λ_max / Σ λ_i

over the correlation matrix of recent sector returns. Because trace(corr) = n,
this is λ_max / n. A high AR means one factor dominates — sectors have collapsed
into a single fragile bloc where a shock cascades everywhere. Low AR means
healthy, independent rotation.

Empirical bands (matches what the consumer documented):
  • ~0.40–0.65  normal, diversified market
  • > 0.80      extreme structural correlation — fragile

Output: `systemic_fragility` ∈ [0,1] in data/features/systemic_fragility.json.
The window is 60 trading days (followup S3 suspected the old 30d window was too
short to estimate an 11×11 correlation matrix stably).
"""

import logging
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

ABSORPTION_WINDOW = 60  # trading days; ≥ assets² stability for an 11×11 corr matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOGS_DIR / "phase6.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def build_returns_matrix(features_dir: Path, window: int) -> pd.DataFrame:
    """Daily log returns for the 11 sectors over the last `window` days, aligned.

    The old implementation tail-sliced each series independently, which could
    misalign dates and silently produce a near-identity correlation matrix
    (the likely root cause of the 0.018 bug). We align on the shared DatetimeIndex
    BEFORE slicing so every column covers the same trading days.
    """
    series = {}
    for sector in cfg.SECTOR_TICKERS:
        csv_path = features_dir / f"{sector}_features.csv"
        if not csv_path.exists():
            logger.warning(f"  missing {csv_path.name}; excluded from fragility")
            continue
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True).sort_index()
        series[sector] = df["log_return"]

    matrix = pd.DataFrame(series).dropna(how="any")
    return matrix.tail(window)


def absorption_ratio(returns_matrix: pd.DataFrame) -> float:
    """λ_max / Σλ of the return correlation matrix. ∈ (0,1]."""
    corr = returns_matrix.corr().to_numpy()
    # Symmetric PSD matrix → real non-negative eigenvalues.
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(eigvals.sum())
    if total <= 0:
        return float("nan")
    ar = float(eigvals.max() / total)
    logger.info(
        f"Absorption Ratio: {ar:.4f} (λ_max={eigvals.max():.2f}, n={len(eigvals)}, "
        f"window={len(returns_matrix)}d)"
    )
    return ar


def main():
    logger.info("=" * 50)
    logger.info("PHASE 6 — Systemic Fragility (PCA Absorption Ratio)")
    logger.info("=" * 50)

    features_dir = cfg.DATA_DIR / "features"
    returns_matrix = build_returns_matrix(features_dir, window=ABSORPTION_WINDOW)

    if returns_matrix.shape[0] < 20 or returns_matrix.shape[1] < 3:
        logger.error(
            f"Insufficient aligned sector data ({returns_matrix.shape}); "
            "cannot compute a stable absorption ratio."
        )
        sys.exit(1)

    fragility = absorption_ratio(returns_matrix)
    if not np.isfinite(fragility):
        logger.error("Degenerate correlation matrix; fragility undefined.")
        sys.exit(1)

    out_path = features_dir / "systemic_fragility.json"
    with open(out_path, "w") as f:
        json.dump({"systemic_fragility": round(fragility, 4)}, f)

    logger.info(f"Saved fragility score to {out_path.name}")
    logger.info("Phase 6 complete. ✓")


if __name__ == "__main__":
    main()
