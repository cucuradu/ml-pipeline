#!/usr/bin/env python3
"""
Visualization script to plot SPY price colored by GMM Regimes.
Saves the output to output/regime_plot.png
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import config as cfg

def main():
    csv_path = cfg.DATA_DIR / "features" / "SPY_regimes.csv"
    if not csv_path.exists():
        print("Data not found. Please run the pipeline first.")
        return

    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    
    # Define colors for the regimes
    color_map = {
        "Bull": "#2ca02c",       # Green
        "Neutral": "#1f77b4",    # Blue
        "Caution": "#ff7f0e",    # Orange
        "Defensive": "#d62728"   # Red
    }

    plt.figure(figsize=(15, 8))
    
    # Plot the base price line lightly
    plt.plot(df.index, df["close"], color='black', alpha=0.3, linewidth=1)
    
    # Scatter plot on top with regime colors
    for regime, color in color_map.items():
        mask = df["regime"] == regime
        plt.scatter(df.index[mask], df["close"][mask], 
                    color=color, label=regime, s=15, alpha=0.8)

    plt.title("SPY Price with GMM Market Regimes (Last 5 Years)", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("SPY Close Price ($)", fontsize=12)
    plt.legend(title="Regime", loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = cfg.OUTPUT_DIR / "regime_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to {output_path}")

if __name__ == "__main__":
    main()
