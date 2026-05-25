import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import pickle
import sys
from pathlib import Path
import numpy as np

# Add parent directory to sys.path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    import config as cfg
except ImportError:
    st.error("Could not import config.py. Make sure app.py is in the dashboard/ directory of the Trader repo.")
    st.stop()

# Set page config
st.set_page_config(page_title="Research Department: Control Center", layout="wide", page_icon="🔬")

# ==========================================
# Data Loading
# ==========================================
@st.cache_data
def load_data():
    # 1. Load ML Insights JSON
    insights_path = cfg.OUTPUT_DIR / cfg.INSIGHTS_FILENAME
    insights_data = {}
    if insights_path.exists():
        with open(insights_path, "r") as f:
            insights_data = json.load(f)

    # 2. Load SPY Features & Regimes
    features_path = cfg.DATA_DIR / "features" / "SPY_features.csv"
    regimes_path = cfg.DATA_DIR / "features" / "SPY_regimes.csv"
    
    df_merged = pd.DataFrame()
    if features_path.exists() and regimes_path.exists():
        df_feat = pd.read_csv(features_path, parse_dates=['date'])
        df_reg = pd.read_csv(regimes_path, parse_dates=['date'])
        
        # Merge on date, only taking 'regime' from df_reg to avoid duplicate 'close' columns
        df_merged = pd.merge(df_feat, df_reg[['date', 'regime']], on='date', how='inner')
        df_merged.set_index('date', inplace=True)
        df_merged.sort_index(inplace=True)

    # 3. Load GARCH Model
    garch_vol = None
    garch_path = cfg.MODELS_DIR / "garch_model.pkl"
    if garch_path.exists():
        try:
            with open(garch_path, "rb") as f:
                res = pickle.load(f)
            # Extract conditional volatility, unscale (assuming returns were scaled by 100), and annualize
            garch_vol = (res.conditional_volatility / 100.0) * np.sqrt(252)
        except Exception as e:
            # Silently handle failure to load arch model (e.g. if arch is not installed in the environment running streamlit)
            pass

    return insights_data, df_merged, garch_vol

insights, df_spy, garch_vol_series = load_data()

# ==========================================
# UI: Sidebar
# ==========================================
st.sidebar.title("Research Department: Control Center")

if insights:
    st.sidebar.markdown(f"**Generated At:** `{insights.get('generated_at', 'N/A')}`")
    st.sidebar.markdown(f"**Model Version:** `{insights.get('model_version', 'N/A')}`")

if not df_spy.empty:
    min_date = df_spy.index.min().date()
    max_date = df_spy.index.max().date()
    
    start_date, end_date = st.sidebar.slider(
        "Date Range (Historical Charts)",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )
    
    # Filter DataFrame based on date slider
    df_filtered = df_spy.loc[start_date:end_date]
else:
    df_filtered = pd.DataFrame()
    st.sidebar.warning("No historical data found. Have you run Phase 1 & 2?")

# ==========================================
# UI: Top Row (Metrics)
# ==========================================
st.markdown("### Key Pipeline Metrics")
col1, col2, col3, col4 = st.columns(4)

if insights:
    market = insights.get("market", {})
    vol = insights.get("volatility", {})
    
    regime = market.get("regime", "N/A")
    conf = market.get("confidence", 0) * 100
    dep_target = market.get("deployment_target", 0)
    slots = market.get("trade_slots", 0)
    garch_1d = vol.get("garch_1d_forecast_pct", 0)
    vix_struct = vol.get("vix_implied_term_structure", "N/A")

    col1.metric("Current Regime", regime)
    col2.metric("GMM Confidence", f"{conf:.4f}%")
    col3.metric("Deployment Target / Slots", f"{dep_target} / {slots}")
    col4.metric("Volatility Forecast (1D)", f"{garch_1d}%", delta=vix_struct, delta_color="off")
else:
    st.warning("No ML Insights found. Pipeline may not have run successfully.")

st.divider()

# ==========================================
# UI: Middle Row (Regime Classifier Visuals)
# ==========================================
st.markdown("### Regime Classifier Diagnostics (Phase 2)")
col_a, col_b = st.columns(2)

# Global colors for regimes to ensure consistency across charts
regime_colors = {
    "Bull": "green",
    "Neutral": "gray",
    "Caution": "orange",
    "Defensive": "red"
}

if not df_filtered.empty:
    # Visual A: GMM Feature Space
    with col_a:
        st.markdown("**GMM Feature Space**")
        
        # Verify required columns exist
        if "sma_dist_200" in df_filtered.columns and "rv_21d" in df_filtered.columns:
            fig_a = px.scatter(
                df_filtered.reset_index(),
                x="sma_dist_200",
                y="rv_21d",
                color="regime",
                color_discrete_map=regime_colors,
                hover_data=["date", "close", "log_return"],
                title="SMA Dist 200 vs Realized Vol (21d) by Regime",
                template="plotly_dark"
            )
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.info("Missing 'sma_dist_200' or 'rv_21d' features in dataset.")

    # Visual B: SPY Price & Regime Timeline
    with col_b:
        st.markdown("**SPY Price & Regime Timeline**")
        fig_b = go.Figure()
        
        # Add SPY line
        fig_b.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered["close"],
            mode='lines',
            name='SPY Close',
            line=dict(color='white')
        ))
        
        # Find contiguous regimes for background shading
        regime_blocks = []
        current_regime = None
        start_idx = None
        
        for idx, row in df_filtered.iterrows():
            if row['regime'] != current_regime:
                if current_regime is not None:
                    regime_blocks.append((start_idx, idx, current_regime))
                current_regime = row['regime']
                start_idx = idx
                
        if current_regime is not None and not df_filtered.empty:
            regime_blocks.append((start_idx, df_filtered.index[-1], current_regime))
            
        for start, end, reg in regime_blocks:
            fig_b.add_vrect(
                x0=start, x1=end,
                fillcolor=regime_colors.get(reg, "gray"),
                opacity=0.2,
                layer="below",
                line_width=0,
            )
            
        fig_b.update_layout(
            title="SPY Close Price over Time (Regimes Shaded)",
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Price"
        )
        st.plotly_chart(fig_b, use_container_width=True)

st.divider()

# ==========================================
# UI: Bottom Row (Volatility & Sectors)
# ==========================================
st.markdown("### Volatility Tracking & Sector Breadth")
col_c, col_d = st.columns(2)

# Visual C: Volatility Line Chart
with col_c:
    st.markdown("**GARCH Volatility vs Realized Volatility**")
    if not df_filtered.empty:
        if "rv_21d" in df_filtered.columns:
            fig_c = go.Figure()
            
            # Realized Volatility
            fig_c.add_trace(go.Scatter(
                x=df_filtered.index,
                y=df_filtered["rv_21d"],
                mode='lines',
                name='Realized Vol (21d)',
                line=dict(color='cyan')
            ))
            
            # GARCH Volatility
            if garch_vol_series is not None:
                # Filter GARCH to match the exact Date Slider boundaries
                garch_mask = (garch_vol_series.index.date >= start_date) & (garch_vol_series.index.date <= end_date)
                garch_aligned = garch_vol_series.loc[garch_mask]
                
                fig_c.add_trace(go.Scatter(
                    x=garch_aligned.index,
                    y=garch_aligned,
                    mode='lines',
                    name='GARCH Conditional Vol',
                    line=dict(color='magenta', dash='dash')
                ))
                
            fig_c.update_layout(
                title="Volatility Tracking",
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Annualized Volatility"
            )
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.info("Missing 'rv_21d' feature in dataset.")

# Visual D: Sector Breadth
with col_d:
    st.markdown("**Sector Momentum Scores**")
    if insights and "sectors" in insights:
        sectors_data = insights["sectors"]
        # Prepare df for bar chart
        records = []
        for ticker, data in sectors_data.items():
            records.append({
                "Ticker": ticker,
                "Regime": data.get("regime", "N/A"),
                "Score": data.get("score", 0)
            })
        
        if records:
            df_sec = pd.DataFrame(records)
            df_sec.sort_values("Score", ascending=True, inplace=True)
            
            # Colors for sector regimes
            sec_regime_colors = {
                "Trend": "dodgerblue",
                "Choppy": "gold",
                "Bear": "crimson"
            }
            
            fig_d = px.bar(
                df_sec,
                x="Score",
                y="Ticker",
                orientation='h',
                color="Regime",
                color_discrete_map=sec_regime_colors,
                title="Sector Momentum Scores",
                template="plotly_dark",
                text_auto='.4f'
            )
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("No sector data inside 'sectors' object.")
    else:
        st.info("No 'sectors' object available in ML Insights.")
