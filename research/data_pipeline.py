import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf


# Add the project root to sys.path so we can import config
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import SECTOR_TICKERS, CORE_TICKERS, START_DATE, END_DATE, DATA_DIR

def build_features_dataset():
    print("=========================================")
    print("Institutional Data Pipeline (Research)")
    print("=========================================")
    
    # 1. Fetch Equities Data
    tickers = ["SPY"] + SECTOR_TICKERS
    print(f"Fetching Equities Data for {len(tickers)} tickers from {START_DATE} to {END_DATE}...")
    
    # yfinance downloads data where columns are multi-index (Price, Ticker)
    prices_df = yf.download(tickers, start=START_DATE, end=END_DATE)['Close']
    
    # Ensure SPY is handled
    spy_df = pd.DataFrame({'SPY_Close': prices_df['SPY']})
    
    # Calculate 5-day forward return for every sector
    # forward_5d_return(t) = (Price(t+5) - Price(t)) / Price(t)
    # We shift by -5 so the target for today is the return over the NEXT 5 days
    returns_5d = prices_df[SECTOR_TICKERS].pct_change(periods=5).shift(-5)
    
    # 2. Fetch Macroeconomic Data (FRED API/Direct CSV)
    print("Fetching Macroeconomic Data from FRED...")
    macro_series = {
        'T10Y2Y': 'Yield_Curve_10Y2Y',
        'BAMLH0A0HYM2': 'High_Yield_Spread'
    }
    
    macro_dfs = []
    for series_id, col_name in macro_series.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        print(f"Downloading {series_id}...")
        df_series = pd.read_csv(url, parse_dates=['observation_date'], index_col='observation_date', na_values='.')
        df_series.rename(columns={series_id: col_name}, inplace=True)
        # Convert index to timezone-naive datetime to match yfinance output
        df_series.index = df_series.index.tz_localize(None)
        macro_dfs.append(df_series)
        
    macro_df = pd.concat(macro_dfs, axis=1)
    
    # 3. Data Engineering & Merging
    print("Merging datasets and handling missing values...")
    
    # Stack the sector prices into a long-format dataframe so we have one target column `y`
    # This aligns perfectly with the XGBoost Ranker which needs to rank sectors per day.
    long_sectors = []
    
    for sector in SECTOR_TICKERS:
        df_sector = pd.DataFrame({
            'date': prices_df.index,
            'ticker': sector,
            'sector_close': prices_df[sector].values,
            'forward_5d_return': returns_5d[sector].values
        }).set_index('date')
        long_sectors.append(df_sector)
        
    final_long_df = pd.concat(long_sectors)
    
    # Merge SPY and Macro onto the long dataframe based on Date index
    final_merged = final_long_df.join(spy_df, how='left').join(macro_df, how='left')
    
    # Handle Holiday mismatches between Equities and Bond Markets using forward-fill
    # Then drop NaNs (which mainly removes the last 5 days where forward_5d_return is NaN,
    # plus any initial dates where macro data might be missing)
    final_cleaned = final_merged.ffill().dropna()
    
    # Ensure date is a column (reset index)
    final_cleaned = final_cleaned.reset_index()
    
    # 4. Save perfectly clean DataFrame
    output_dir = os.path.join(DATA_DIR, "features")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "SPY_features.csv")
    
    # Convert categorical tickers to numerical encoding for XGBoost
    # XGBoost can't natively handle string columns easily without category dtypes
    final_cleaned['ticker_id'] = final_cleaned['ticker'].astype('category').cat.codes
    
    final_cleaned.to_csv(output_path, index=False)
    
    print(f"Data Pipeline Complete!")
    print(f"Final shape: {final_cleaned.shape}")
    print(f"Cleaned dataset saved to: {output_path}")

if __name__ == "__main__":
    build_features_dataset()
