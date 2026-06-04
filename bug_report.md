# ML Pipeline — Bug & Error Report

Audit of every source file in `~/ml-pipeline`. Findings ordered by severity.

---

## 🔴 Critical — Will cause incorrect results or runtime crashes

### C1. `dist_from_52w_high` formula is inverted
**File:** [phase5_ranking_model.py](file:///home/radu/ml-pipeline/phase5_ranking_model.py#L128)

```python
f["dist_from_52w_high"] = (1.0 - (high252 - close) / close).clip(0.0, 1.0)
```

This divides by `close` instead of `high252`. When `close` is far below the 52-week high, `(high252 - close) / close` can exceed 1.0, making the result negative (then clipped to 0). The correct formula is:

```python
f["dist_from_52w_high"] = (close / high252).clip(0.0, 1.0)
```

This matches the feature's documented intent ("how close to the 52-week high") and also simplifies the expression. Currently the model is trained on a distorted feature.

---

### C2. `run_pipeline.sh` missing `set -e` — failures silently ignored
**File:** [run_pipeline.sh](file:///home/radu/ml-pipeline/run_pipeline.sh#L21)

```bash
set -uo pipefail
```

The script has `set -u` (undefined vars) and `pipefail`, but **not** `set -e` (exit on error). While key steps have explicit `if ! ... ; then exit 1; fi` guards, any un-guarded command that fails mid-script (e.g. `mkdir -p docs`, `cp`, `git add`) will silently continue. Should be `set -euo pipefail`.

---

### C3. `evaluate_models.py` — FutureWarning → imminent breakage on `groupby().apply()`
**File:** [evaluate_models.py](file:///home/radu/ml-pipeline/evaluate_models.py#L91)

```python
daily = df.groupby("d").apply(lambda g: spearmanr(g["s"], g["f"]).correlation ...)
```

Same pattern at [phase5_ranking_model.py L178](file:///home/radu/ml-pipeline/phase5_ranking_model.py#L178). In pandas ≥ 2.2, `groupby().apply()` returning a scalar per group triggers a `FutureWarning` about `include_groups` and will change behavior in a future release. This will break silently when pandas is upgraded to the next major version. Should use `.apply(..., include_groups=False)` explicitly, or return the scalar from an aggregation path.

---

### C4. `generate_insights.py` — `vvix` value emitted with float32 noise
**File:** [generate_insights.py](file:///home/radu/ml-pipeline/generate_insights.py#L190) → [ml-insights.json L100](file:///home/radu/ml-pipeline/output/ml-insights.json#L100)

The output has:
```json
"vvix": 89.80000305175781
```

This is because the CSV value was read as float32 (from yfinance) and passed directly to `json.dump`. Should `round()` like every other numeric field:

```python
vvix = round(float(...), 2)
```

This is cosmetic but violates the "clean contract" principle — the consumer parses this.

---

### C5. `phase7_macro.py` — `_trailing_pct_rank` counts `<=` not `<`, inflating percentiles
**File:** [phase7_macro.py](file:///home/radu/ml-pipeline/phase7_macro.py#L59-L61)

```python
lambda w: (w <= w[-1]).mean()
```

This counts the current observation *itself* in the percentile, so the minimum rank of any observation is `1/window` ≈ 0.0013, never 0.0. More importantly, ties are all counted as "below", which **inflates** percentile ranks for flat series (NFCI and claims can be unchanged for weeks). The standard formulation uses `< w[-1]` (exclusive) and/or `rank(pct=True)`. The current implementation biases `macro_risk` upward (more risk-off).

> [!NOTE]
> This is an edge case but can materially affect the regime when multiple component percentiles are inflated by the same ~1-3% bias.

---

## 🟡 Medium — Correctness / robustness issues

### M1. `phase2_regime_model.py` — persistence filter uses `raw_vals` not `filtered_vals`
**File:** [phase2_regime_model.py](file:///home/radu/ml-pipeline/phase2_regime_model.py#L167-L178)

```python
raw_vals = raw_regimes.values.copy()
filtered_vals = np.empty_like(raw_vals)
current_regime = raw_vals[0]

for i in range(len(raw_vals)):
    if i >= persistence_bars:
        window = raw_vals[i - persistence_bars + 1 : i + 1]  # ← uses raw, not filtered
        if len(set(window)) == 1:
            current_regime = window[0]
    filtered_vals[i] = current_regime
```

The window reads from `raw_vals`, which means the filter checks if the *raw* HMM predictions have been consistent for N bars. This is technically a design choice but it means **the filter can switch to a new regime that the raw signal only held for 3 bars, even if the filtered output was stuck in the previous regime for 100 bars.** A more robust persistence filter would read from `filtered_vals` (the previous filtered output), preventing cascading rapid switches.

---

### M2. `generate_insights.py` — `sector_regime()` reads unshifted features (look-ahead)
**File:** [generate_insights.py](file:///home/radu/ml-pipeline/generate_insights.py#L123-L131)

```python
def sector_regime(features_dir, sector: str) -> str:
    df = pd.read_csv(features_dir / f"{sector}_features.csv", ...)
    sma50_dist = float(df["sma_dist_50"].iloc[-1])       # today's SMA distance
    ret_10d = float(df["close"].iloc[-1] / df["close"].iloc[-11] - 1.0)
```

The `SPY_features.csv` data is shifted by 1 day in Phase 2 for the regime model, but the sector features CSVs are **not** shifted. The `sector_regime()` function reads the unshifted latest row, which represents *today's* data — but `sma_dist_50` is computed from today's close which is only final at market close. If the pipeline runs intraday, this is a mild look-ahead issue. In a daily 06:30 cron run, it uses yesterday's close (the last bar), so it's safe in the current schedule.

---

### M3. `generate_insights.py` — `universe_weights` div-by-zero on zero-vol symbol
**File:** [generate_insights.py](file:///home/radu/ml-pipeline/generate_insights.py#L233-L234)

```python
rv = pd.read_csv(fp, ...)[\"log_return\"].iloc[-21:].std() * np.sqrt(252)
inv[sym] = 1.0 / rv if rv and rv > 0 else 0.0
```

`rv and rv > 0` relies on the truthiness of a numpy scalar. If `rv` is `np.nan`, `rv and ...` evaluates `rv > 0` (since `np.nan` is truthy), which returns `False`, so it falls to `0.0` — OK. But if `rv` is exactly `0.0` (e.g., a stock with identical prices for 21 days), it would also be caught — OK. However, a more defensive check would be `if np.isfinite(rv) and rv > 0` for clarity and to handle `np.inf`.

---

### M4. `phase5_ranking_model.py` — `walk_forward_ic` uses positional TimeSeriesSplit on a panel
**File:** [phase5_ranking_model.py](file:///home/radu/ml-pipeline/phase5_ranking_model.py#L169-L182)

`TimeSeriesSplit` splits by **row index position**, but the panel has multiple rows per date (one per symbol). This means a single date's symbols can be split across train and test sets — the model sees some symbols' data from date T in training and predicts other symbols from the same date T in test. This isn't look-ahead per se (features are per-symbol), but it violates the temporal integrity of the cross-validation because the cross-sectional target (demeaned forward return) is computed from all symbols on that date.

The `evaluate_models.py` version (`walk_forward_oof`) correctly uses date-based splits. This function should too.

---

### M5. `requirements.txt` — `networkx` listed but never imported
**File:** [requirements.txt](file:///home/radu/ml-pipeline/requirements.txt#L38)

```
networkx>=3.0               # Sector-correlation graph density / fragility
```

Phase 6 was rewritten from graph density to PCA absorption ratio and no longer imports `networkx`. This is a dead dependency. Similarly, `ta>=0.11` is listed but never imported by any phase.

---

### M6. `config.py` — `from typing import Literal` buried mid-file, after module-level code
**File:** [config.py](file:///home/radu/ml-pipeline/config.py#L141)

```python
from typing import Literal
```

This import is placed at line 141, after ~140 lines of module-level variable definitions. While functionally harmless (it's still module-scope), it violates PEP 8 convention and can confuse tools. The `Literal` import is only used for `GARCH_DIST` (line 148) and should be at the top with the other imports.

---

### M7. `phase1_data_pipeline.py` — single-ticker chunk misattributes data
**File:** [phase1_data_pipeline.py](file:///home/radu/ml-pipeline/phase1_data_pipeline.py#L86-L97)

```python
multi = isinstance(data.columns, pd.MultiIndex)
for ticker in chunk:
    ...
    if multi:
        df = data[ticker].dropna(how="all")
    else:
        df = data.dropna(how="all")  # single-ticker chunk
```

If a chunk has multiple tickers but `yf.download` returns a non-MultiIndex (e.g. all but one failed), **every** ticker in the chunk loop gets the same single DataFrame. This is unlikely but not impossible with flaky Yahoo responses. The `data.columns` check should be per-ticker, or the loop should break after the first non-MultiIndex assignment.

---

### M8. `generate_insights.py` — no `try/except` around file reads during assembly
**File:** [generate_insights.py](file:///home/radu/ml-pipeline/generate_insights.py#L162-L199)

`generate_json()` does bare `pd.read_csv()` and `_read_json()` calls for `SPY_regimes.csv`, `SPY_features.csv`, `RSP_features.csv`, `VIX_raw.csv`, `VIX3M_raw.csv`, `garch_forecast.json`, `systemic_fragility.json`, `universe_ranking.json`. If any upstream phase produced an empty or corrupt file, the entire assembly crashes with an unhandled exception. Given the `config.py` rule about graceful degradation, each read should be wrapped or at least checked before indexing with `.iloc[-1]`.

---

## 🟢 Low — Style, maintenance, minor issues

### L1. `visualize_regimes.py` title says "GMM" but model is HMM

**File:** [visualize_regimes.py](file:///home/radu/ml-pipeline/visualize_regimes.py#L39)

```python
plt.title("SPY Price with GMM Market Regimes (Last 5 Years)", ...)
```

The regime model was migrated from GMM to HMM. The plot title (and the docstring on line 4) still say "GMM." Similarly, the saved model file is still named `gmm_pipeline.pkl` for compatibility (line 239 of phase2), which is a known conscious decision but adds confusion.

---

### L2. `run_pipeline.sh` — hardcoded Mac default path
**File:** [run_pipeline.sh](file:///home/radu/ml-pipeline/run_pipeline.sh#L24)

```bash
BOT_REPO="${ML_BOT_REPO:-/Users/raducucu/Developer/trading-bot}"
```

The default falls back to a macOS path. Since this is now deployed on Ubuntu (`~/trading-bot`), the default should match the deploy doc or at least be `$HOME/trading-bot`.

---

### L3. `validate_payload.py` — hardcoded Mac path in cross-check
**File:** [validate_payload.py](file:///home/radu/ml-pipeline/validate_payload.py#L131)

```python
repo = os.environ.get("CONSUMER_REPO", "/Users/raducucu/Developer/trading-bot")
```

Same issue — the fallback path is Mac-specific.

---

### L4. `config.py` — `RUN_OPTUNA = True` declared but never referenced
**File:** [config.py](file:///home/radu/ml-pipeline/config.py#L138)

```python
RUN_OPTUNA = True  # Set to True to dynamically optimize XGBoost hyperparameters
```

This flag is defined but `grep` shows it's never imported or checked in any phase script. It's a dead config parameter.

---

### L5. `launchd/` plist is macOS-specific, useless on Ubuntu
**File:** `launchd/com.cucuradu.ml-insights.plist`

The deployment doc (`DEPLOY-UBUNTU.md`) correctly uses cron. The launchd plist is a macOS artifact that will never work on the Ubuntu host. Not harmful but adds confusion.

---

## Summary Table

| ID | Severity | File | Issue |
|----|----------|------|-------|
| C1 | 🔴 Critical | phase5 | `dist_from_52w_high` formula divides by `close` instead of `high252` |
| C2 | 🔴 Critical | run_pipeline.sh | Missing `set -e` — un-guarded failures silently continue |
| C3 | 🔴 Critical | evaluate_models / phase5 | `groupby().apply()` scalar return → pandas FutureWarning / breakage |
| C4 | 🔴 Critical | generate_insights | VVIX emitted with float32 noise (89.80000305175781) |
| C5 | 🔴 Critical | phase7 | `_trailing_pct_rank` uses `<=` (inclusive), inflating percentiles |
| M1 | 🟡 Medium | phase2 | Persistence filter reads from `raw_vals` not `filtered_vals` |
| M2 | 🟡 Medium | generate_insights | `sector_regime()` reads unshifted features |
| M3 | 🟡 Medium | generate_insights | `universe_weights` div check uses numpy truthiness |
| M4 | 🟡 Medium | phase5 | `walk_forward_ic` splits panel by row-position, not date |
| M5 | 🟡 Medium | requirements.txt | `networkx` and `ta` are dead dependencies |
| M6 | 🟡 Medium | config.py | `from typing import Literal` mid-file |
| M7 | 🟡 Medium | phase1 | Single-ticker fallback misattributes data across chunk |
| M8 | 🟡 Medium | generate_insights | No error handling around file reads in assembly |
| L1 | 🟢 Low | visualize_regimes | Title says "GMM" but model is HMM |
| L2 | 🟢 Low | run_pipeline.sh | Default `BOT_REPO` path is Mac-specific |
| L3 | 🟢 Low | validate_payload | Default `CONSUMER_REPO` path is Mac-specific |
| L4 | 🟢 Low | config.py | `RUN_OPTUNA` declared but never used |
| L5 | 🟢 Low | launchd/ | macOS plist unused on Ubuntu |
