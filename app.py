import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- MODULES ---
import config
from utils import get_api_key, terminal_chart_layout
import data_engine as de
import quant_engine as qe
import ai_engine as ai
from utils import generate_pdf_report, parse_eco_value, analyze_eco_context, generate_cot_analysis

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="JD Capital Quantitative Investment Terminal", page_icon="static/logo.png")
st.markdown(config.CSS_STYLE, unsafe_allow_html=True)


# --- SESSION STATE INITIALIZATION ---
if 'gemini_calls' not in st.session_state: st.session_state['gemini_calls'] = 0
if 'news_calls' not in st.session_state: st.session_state['news_calls'] = 0
if 'rapid_calls' not in st.session_state: st.session_state['rapid_calls'] = 0
if 'coingecko_calls' not in st.session_state: st.session_state['coingecko_calls'] = 0
if 'fred_calls' not in st.session_state: st.session_state['fred_calls'] = 0
if 'narrative_cache' not in st.session_state: st.session_state['narrative_cache'] = None
if 'thesis_cache' not in st.session_state: st.session_state['thesis_cache'] = None
if 'last_ai_call_time' not in st.session_state: st.session_state['last_ai_call_time'] = 0
if 'gemini_model_name' not in st.session_state: st.session_state['gemini_model_name'] = None

# --- SIDEBAR ---
with st.sidebar:
    st.image("static/logo.png", use_container_width=True)
    st.markdown("<h3 style='color: #00FFFF;'>COMMAND LINE</h3>", unsafe_allow_html=True)
    selected_asset = st.selectbox("SEC / Ticker", list(config.ASSETS.keys()))
    asset_info = config.ASSETS[selected_asset]
    
    # --- ASSET CHANGE CACHE CLEARING ---
    if st.session_state.get('last_asset') != selected_asset:
        st.session_state['narrative_cache'] = None
        st.session_state['thesis_cache'] = None
        st.session_state['last_asset'] = selected_asset

    use_demo_data = st.checkbox("🛠️ USE DEMO DATA (Save Quota)", value=True, help="Use mock data for Calendar to save RapidAPI credits.")
    use_grounding = st.checkbox("🔍 ENABLE SEARCH GROUNDING", value=False, help="Enable real-time Google Search grounding for global macro conditions and data releases.")
    if use_grounding:
        st.markdown("""
        <div style='font-size:0.7em; color:#00FFFF; line-height:1.2; margin-bottom:10px;'>
        • Global macro conditions & data releases<br>
        • Economic data release breakdowns<br>
        • Cross-asset positioning & flow insights<br>
        • Central bank watch & rate path expectations<br>
        • Weekly deep-dive research notes
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("📡 API QUOTA MONITOR", expanded=True):
        st.markdown("<div style='font-size:0.7em; color:#AAAAAA;'>Session Usage vs Hard Limits</div>", unsafe_allow_html=True)
        import time
        last_call = st.session_state.get('last_ai_call_time', 0)
        time_passed = time.time() - last_call
        cooldown = config.THRESHOLDS.get('AI_COOLDOWN', 20)
        ready_in = max(0, int(cooldown - time_passed))
        
        st.write(f"**Gemini AI (Free Tier)**")
        if ready_in > 0:
            st.warning(f"Throttled: Ready in {ready_in}s")
        else:
            st.success("Ready for Synthesis")
        st.progress(min(time_passed / cooldown, 1.0) if ready_in > 0 else 1.0)
        
        st.write(f"**NewsAPI** ({st.session_state['news_calls']} / 100)")
        st.progress(min(st.session_state['news_calls'] / 100, 1.0))
        st.write(f"**RapidAPI** ({st.session_state['rapid_calls']} / 10)")
        st.progress(min(st.session_state['rapid_calls'] / 10, 1.0))
        st.write(f"**FRED** ({st.session_state['fred_calls']} calls)")
        st.write(f"**CoinGecko** ({st.session_state['coingecko_calls']} calls)")
        
        if use_demo_data:
            st.success("🟢 DEMO MODE ACTIVE")
        else:
            st.warning("🔴 LIVE API MODE")
        
        # Dependency Status
        if not ai.HAS_NLP:
            st.warning("NLP Disabled: `textblob` missing")
        if not de.HAS_COT_LIB:
            st.info("Using Direct COT Fetch (No Lib)")
            
    st.markdown("---")
    rapid_key = get_api_key("rapidapi_key")
    news_key = get_api_key("news_api_key")
    gemini_key = get_api_key("gemini_api_key")
    cg_key = get_api_key("coingecko_key") 
    fred_key = get_api_key("fred_api_key")
    finnhub_key = get_api_key("finnhub_api_key")
    
    st.markdown("---")
    with st.expander("🌍 MARKET PULSE", expanded=True):
        watchlist_tickers = [v['ticker'] for v in list(config.ASSETS.values())[:6]] # Only top 6 for speed
        watchlist_df = de.get_watchlist_data(watchlist_tickers)
        if not watchlist_df.empty:
            for _, row in watchlist_df.iterrows():
                col_t, col_p = st.columns([2, 1])
                color = "#00FFFF" if row['change'] >= 0 else "#8080FF"
                col_t.markdown(f"<span style='font-size:0.8em;'>{row['ticker']}</span>", unsafe_allow_html=True)
                col_p.markdown(f"<span style='color:{color}; font-family:monospace; font-size:0.8em;'>{row['change']:+.2f}%</span>", unsafe_allow_html=True)
        else: st.info("Pulse unavailable.")

    if st.button(">> REFRESH DATA"): 
        st.cache_data.clear()
        st.rerun()

# ==============================================================================
# 1. CENTRALIZED DATA FETCHING ENGINE (CORE ONLY)
# ==============================================================================
# Essential Market Data (Required for HUD and base logic)
daily_data = de.get_daily_data(asset_info['ticker'])
curr = daily_data['Close'].iloc[-1] if not daily_data.empty else 0
high = daily_data['High'] if not daily_data.empty else pd.Series([0])
low = daily_data['Low'] if not daily_data.empty else pd.Series([0])
pct = ((daily_data['Close'].iloc[-1] / daily_data['Close'].iloc[-2]) - 1) * 100 if len(daily_data) > 1 else 0

# Advanced Engines (Lazy Loading prioritized in tabs)
_, ml_prob = qe.get_ml_prediction(asset_info['ticker'])
regime_data = qe.get_market_regime(asset_info['ticker'])
hurst = qe.calculate_hurst(daily_data['Close'].values) if not daily_data.empty else 0.5
ml_signal = "BULLISH" if ml_prob > config.THRESHOLDS['ML_BULLISH'] else "BEARISH" if ml_prob < config.THRESHOLDS['ML_BEARISH'] else "NEUTRAL"

# Multilayer Technicals (Initial fetch for HUD)
key_levels = qe.get_key_levels(daily_data)
# ADAPTIVE: Radar now tunes its own parameters based on regime_data
radar_signals = qe.calculate_technical_radar(daily_data, regime=regime_data['regime'] if regime_data else "STABLE")

# State for Asset Change Confirmation
if 'last_analyzed_asset' not in st.session_state: st.session_state['last_analyzed_asset'] = selected_asset
show_regen_warning = st.session_state['last_analyzed_asset'] != selected_asset

# --- INITIALIZE GLOBAL VARIABLES (to avoid NameError with lazy loading) ---
intraday_data = pd.DataFrame()
vol_profile = None
poc_price = 0
vwap_df = pd.DataFrame()
rs_data = pd.DataFrame()
corr_matrix = pd.DataFrame()
ms_df = pd.DataFrame()
ms_trend = "N/A"
active_fvgs = []
smc_data = {"obs": [], "sweeps": []}
# ADAPTIVE: Volatility lookback changes based on market fractal (Hurst)
adaptive_vol_window = qe.get_hurst_adjusted_lookback(hurst)
vol_cone = {}
of_bias = "N/A"
cot_history = pd.DataFrame()
cot_data = None
gex_df = None
current_iv = 0
recession_prob = 0
yc_regime = "N/A"
yc_color = "neutral"
yc_impact = "N/A"
news_sentiment_df = pd.DataFrame()
df_cpi = pd.DataFrame()
df_ff = pd.DataFrame()
df_m2 = pd.DataFrame()
macro_regime = None
pred_dates, pred_paths = None, None
seasonality_stats = None
cot_history = pd.DataFrame()
md_score, md_status, md_reasons = 0, "STABLE", []
macro_risk = None

# FRED Base Data (Used in HUD and AI Context)
df_yield_3m = de.get_fred_series("T10Y3M", fred_key)
df_yield = de.get_fred_series("T10Y2Y", fred_key)
df_ff = de.get_fred_series("FEDFUNDS", fred_key)
df_cpi = de.get_fred_series("CPIAUCSL", fred_key)

if not df_yield_3m.empty:
    recession_prob = qe.calculate_recession_probability(df_yield_3m['value'].iloc[-1])
    yc_regime, yc_color = qe.get_yield_curve_regime(df_yield['value'].iloc[-1] if not df_yield.empty else None, df_yield_3m['value'].iloc[-1])
    yc_impact = qe.get_regime_impact(yc_regime, asset_info['ticker'])

# Populate Macro Context for AI
macro_context_data = {}
if not df_yield.empty: macro_context_data['yield_curve'] = f"{df_yield['value'].iloc[-1]:.2f}%"
if not df_cpi.empty: 
    try: macro_context_data['cpi'] = f"{(df_cpi['value'].pct_change(12).iloc[-1]*100):.2f}%"
    except: pass
if not df_ff.empty: macro_context_data['rates'] = f"{df_ff['value'].iloc[-1]:.2f}%"
if macro_regime: macro_context_data['regime'] = macro_regime['regime']

# --- NEW: ADVANCED RISK CALCULATIONS ---
if not daily_data.empty:
    md_score, md_status, md_reasons = qe.detect_momentum_deterioration(daily_data)
    var_95, cvar_95 = qe.calculate_var_cvar(daily_data)
    macro_context_data['momentum_score'] = md_score
    macro_context_data['momentum_status'] = md_status
    macro_context_data['var_95'] = f"{var_95:.2f}%"
    macro_context_data['cvar_95'] = f"{cvar_95:.2f}%"
    macro_context_data['regime_stability'] = f"{regime_data.get('prob_stay', 1.0)*100:.0f}%"
    macro_context_data['kelly_size'] = f"{qe.calculate_kelly_criterion(ml_prob)*100:.1f}%"
    
    # NEW: Adaptive Risk Levels
    adaptive_risk = qe.calculate_adaptive_risk_levels(daily_data, curr, regime=regime_data['regime'] if regime_data else "STABLE")
    
    # Implied Expected Moves (1D and 1W)
    if current_iv:
        r1d = qe.calculate_implied_range(curr, current_iv, days=1)
        r1w = qe.calculate_implied_range(curr, current_iv, days=7)
        if r1d: macro_context_data['expected_move_1d'] = f"±{r1d['move']:,.2f} ({r1d['lower_1sd']:,.0f} - {r1d['upper_1sd']:,.0f})"
        if r1w: macro_context_data['expected_move_1w'] = f"±{r1w['move']:,.2f} ({r1w['lower_1sd']:,.0f} - {r1w['upper_1sd']:,.0f})"
    
    # NEW: Probabilistic Outcome Logic
    pred_dates, pred_paths = qe.generate_monte_carlo(daily_data)
    outcome_probs = {}
    if pred_paths is not None and key_levels:
        outcome_targets = {
            "R1 (Bullish Target)": key_levels['R1'],
            "Pivot (Neutral Mean)": key_levels['Pivot'],
            "S1 (Bearish Target)": key_levels['S1']
        }
        outcome_probs = qe.get_outcome_probabilities(pred_paths, curr, outcome_targets)
        macro_context_data['outcome_probs'] = outcome_probs
if fred_key:
    macro_risk = qe.calculate_macro_pressure(fred_key)
    if macro_risk:
        macro_context_data['macro_pressure_score'] = macro_risk['score']
        macro_context_data['macro_pressure_status'] = macro_risk['status']

# ==============================================================================
# 2. HEAD-UP DISPLAY (HUD) - Immediate Situational Awareness
# ==============================================================================
st.markdown(f"<h1 style='border-bottom: 2px solid #00FFFF;'>{selected_asset} <span style='font-size:0.5em; color:#AAAAAA;'>QUANTITATIVE INVESTMENT TERMINAL</span></h1>", unsafe_allow_html=True)

# Initialize HUD variables to avoid NameError if daily_data is empty
curr = 0.0
pct = 0.0
ml_bias = "NEUTRAL"
fig = None

if not daily_data.empty:
    close, high, low = daily_data['Close'], daily_data['High'], daily_data['Low']
    curr = close.iloc[-1]
    pct = ((curr - close.iloc[-2]) / close.iloc[-2]) * 100 if len(daily_data) > 1 else 0.0
    
    # HUD Layout
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LAST PX", f"{curr:,.2f}", f"{pct:.2f}%")
    
    # ML Signal (CONSOLIDATED)
    ml_signal = "BULLISH" if ml_prob > config.THRESHOLDS['ML_BULLISH'] else "BEARISH" if ml_prob < config.THRESHOLDS['ML_BEARISH'] else "NEUTRAL"
    ml_conf = abs(ml_prob - 0.5) * 200
    ml_color = "bullish" if ml_signal == "BULLISH" else "bearish" if ml_signal == "BEARISH" else "neutral"
    
    # Kelly Sizing
    kelly_pct = qe.calculate_kelly_criterion(ml_prob)
    
    c2.markdown(f"""
    <div class='terminal-box' style="text-align:center; padding:5px;">
        <div style="font-size:0.8em; color:#00FFFF;">AI PREDICTION</div>
        <span class='{ml_color}'>{ml_signal}</span>
        <div style="font-size:0.8em; margin-top:5px; color:#AAAAAA;">CONF: {ml_conf:.0f}%</div>
        <div style="font-size:0.75em; margin-top:3px; color:#00FFFF; border-top:1px solid #1E252F; padding-top:2px;">KELLY SIZE: {kelly_pct*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Regime
    hurst_type = "TRENDING" if hurst > config.THRESHOLDS['HURST_TRENDING'] else "MEAN REVERT" if hurst < config.THRESHOLDS['HURST_MEAN_REVERT'] else "RANDOM WALK"
    h_color = "#00FFFF" if hurst > config.THRESHOLDS['HURST_TRENDING'] else "#8080FF" if hurst < config.THRESHOLDS['HURST_MEAN_REVERT'] else "gray"
    
    if regime_data:
        regime_val = regime_data['regime']
        regime_col = regime_data['color']
        if "COMPRESSION" in regime_val.upper():
            regime_col = "neutral" # Force neutral color but add a custom border
            box_style = "border: 1px solid #FFA500; background: rgba(255, 165, 0, 0.05);"
        else:
            box_style = ""
            
        c3.markdown(f"""
        <div class='terminal-box' style="padding:10px; {box_style}">
            <div style="font-size:0.8em; color:#00FFFF;">QUANT REGIME</div>
            <div style="font-size:1.1em; font-weight:bold;" class='{regime_col}'>{regime_val}</div>
            <div style="font-size:0.7em; display:flex; justify-content:space-between; margin-top:5px;">
                <span>FRACTAL:</span>
                <span style="color:{h_color}">{hurst_type}</span>
            </div>
            <div style="font-size:0.7em; display:flex; justify-content:space-between; margin-top:2px;">
                <span>STABILITY:</span>
                <span style="color:#00FFFF;">{regime_data.get('prob_stay', 1.0)*100:.0f}%</span>
            </div>
            <div style="font-size:0.6em; margin-top:5px; color:#00FFFF; border-top:1px solid #1E252F; padding-top:2px; text-align:center;">
                SYSTEM ADAPTIVE: ACTIVE
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        c3.info("Calculating Regime...")
    
    c4.metric("HIGH/LOW", f"{high.max():,.2f} / {low.min():,.2f}")
    
    # RADAR SIGNALS SCOREBOARD
    if radar_signals:
        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
        r_cols = st.columns(3)
        for i, (key, sig) in enumerate(radar_signals.items()):
            with r_cols[i % 3]:
                st.markdown(f"""
                <div class='terminal-box' style="text-align:center; padding:5px; border: 1px solid #1E252F;">
                    <div style="font-size:0.7em; color:#AAAAAA; text-transform:uppercase;">{key}</div>
                    <div class='{sig['col']}' style='font-size:0.9em;'>{sig['bias']}</div>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# 2.5 RISK ADVISORY: MOMENTUM DETERIORATION & MACRO PRESSURE
# ==============================================================================
st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
risk_c1, risk_c2 = st.columns(2)

with risk_c1:
    md_color = "#FF4B4B" if md_score >= 70 else "#FFA500" if md_score >= 30 else "#00FFFF"
    st.markdown(f"""
    <div class='terminal-box' style='border-left: 5px solid {md_color};'>
        <div style='color:#AAAAAA; font-size:0.8em; text-transform:uppercase;'>Momentum Deterioration Index</div>
        <div style='display:flex; align-items:baseline; gap:10px;'>
            <span style='font-size:2em; font-weight:bold; color:{md_color};'>{md_score}</span>
            <span style='font-size:1.1em; color:{md_color};'>{md_status}</span>
        </div>
        <div style='margin-top:5px;'>
            {"".join([f"<div style='font-size:0.75em; color:#AAAAAA;'>• {r}</div>" for r in md_reasons]) if md_reasons else "<div style='font-size:0.75em; color:gray;'>No deterioration detected.</div>"}
        </div>
    </div>
    """, unsafe_allow_html=True)

with risk_c2:
    if macro_risk:
        mr_color = "#FF4B4B" if macro_risk['score'] >= 60 else "#FFA500" if macro_risk['score'] >= 30 else "#00FFFF"
        st.markdown(f"""
        <div class='terminal-box' style='border-left: 5px solid {mr_color};'>
            <div style='color:#AAAAAA; font-size:0.8em; text-transform:uppercase;'>Macro Pressure Score</div>
            <div style='display:flex; align-items:baseline; gap:10px;'>
                <span style='font-size:2em; font-weight:bold; color:{mr_color};'>{macro_risk['score']}</span>
                <span style='font-size:1.1em; color:{mr_color};'>{macro_risk['status']}</span>
            </div>
            <div style='margin-top:5px;'>
                {"".join([f"<div style='font-size:0.75em; color:#AAAAAA;'>• {f}</div>" for f in macro_risk['factors']]) if macro_risk['factors'] else "<div style='font-size:0.75em; color:gray;'>Macro environment supportive.</div>"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Macro Risk analysis requires valid FRED API Key.")

# --- NEW: PROBABILISTIC RISK SECTION ---
st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
var_c1, var_c2 = st.columns(2)
with var_c1:
    if not daily_data.empty:
        st.markdown(f"""
        <div class='terminal-box' style='border-left: 5px solid #8080FF;'>
            <div style='color:#AAAAAA; font-size:0.8em; text-transform:uppercase;'>Value at Risk (VaR 95%)</div>
            <div style='display:flex; align-items:baseline; gap:10px;'>
                <span style='font-size:2em; font-weight:bold; color:#8080FF;'>{var_95:+.2f}%</span>
            </div>
            <div style='font-size:0.7em; color:gray;'>Worst-case historical loss over 1 day (95% confidence).</div>
        </div>
        """, unsafe_allow_html=True)

with var_c2:
    if not daily_data.empty:
        st.markdown(f"""
        <div class='terminal-box' style='border-left: 5px solid #FF4B4B;'>
            <div style='color:#AAAAAA; font-size:0.8em; text-transform:uppercase;'>Expected Shortfall (CVaR)</div>
            <div style='display:flex; align-items:baseline; gap:10px;'>
                <span style='font-size:2em; font-weight:bold; color:#FF4B4B;'>{cvar_95:+.2f}%</span>
            </div>
            <div style='font-size:0.7em; color:gray;'>Avg. loss if the VaR threshold is breached (Tail Risk).</div>
        </div>
        """, unsafe_allow_html=True)

# --- NEW: ADAPTIVE EXECUTION LEVELS ---
if adaptive_risk:
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    ar_c1, ar_c2 = st.columns(2)
    with ar_c1:
        st.markdown(f"""
        <div class='terminal-box' style='border-left: 5px solid #8080FF;'>
            <div style='color:#AAAAAA; font-size:0.8em; text-transform:uppercase;'>Adaptive Stop Loss (ATR-Based)</div>
            <div style='font-size:1.5em; font-weight:bold; color:#8080FF;'>{adaptive_risk['stop_loss']:,.2f}</div>
            <div style='font-size:0.7em; color:gray;'>Volatility Multiplier: {adaptive_risk['mult']}x ATR</div>
        </div>
        """, unsafe_allow_html=True)
    with ar_c2:
        st.markdown(f"""
        <div class='terminal-box' style='border-left: 5px solid #00FFFF;'>
            <div style='color:#AAAAAA; font-size:0.8em; text-transform:uppercase;'>Adaptive Take Profit</div>
            <div style='font-size:1.5em; font-weight:bold; color:#00FFFF;'>{adaptive_risk['take_profit']:,.2f}</div>
            <div style='font-size:0.7em; color:gray;'>Target: 1.5x Risk Units</div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 3. MACRO & SENTIMENT CONTEXT (THE "WEATHER")
# ==============================================================================
st.markdown("---")
st.markdown("### 🌎 PHASE 1: MACRO & SENTIMENT CONTEXT")

macro_tab1, macro_tab2, macro_tab3 = st.tabs(["🇺🇸 MACRO DASHBOARD", "📰 NEWS & CALENDAR", "🕒 RECESSION CLOCK"])

with macro_tab3:
    if fred_key and not df_yield_3m.empty:
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"""
            <div class='terminal-box' style='text-align:center;'>
                <div style='color:#AAAAAA; font-size:0.8em;'>NY FED MODEL</div>
                <div style='font-size:2em; color:#00FFFF; font-weight:bold;'>{recession_prob:.1f}%</div>
                <div style='font-size:0.8em; color:gray;'>Prob. of Recession (12M)</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='terminal-box' style='text-align:center; margin-top:10px;'>
                 <div style='color:#AAAAAA; font-size:0.8em;'>CURVE REGIME</div>
                 <div class='{yc_color}' style='font-size:1.1em;'>{yc_regime}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='terminal-box' style='border-left: 3px solid #00FFFF;'>
                <div style='font-size:0.75em; color:#AAAAAA; margin-bottom:5px;'>💡 STRATEGIC IMPACT: {selected_asset.upper()}</div>
                <div style='font-size:0.85em; line-height:1.4;'>{yc_impact}</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            try:
                fig_rc = go.Figure()
                fig_rc.add_trace(go.Scatter(x=df_yield_3m.index, y=df_yield_3m['value'], name="10Y-3M Spread", fill='tozeroy', fillcolor='rgba(0, 255, 255, 0.1)', line=dict(color='#00FFFF')))
                fig_rc.add_hline(y=0, line_dash="dash", line_color="white")
                fig_rc = terminal_chart_layout(fig_rc, title="YIELD CURVE (10Y-3M) HISTORY", height=250)
                st.plotly_chart(fig_rc, use_container_width=True)
            except Exception as e:
                st.warning("Recession chart unavailable")
    else:
        st.info("Recession data requires FRED API Key.")

with macro_tab1:
    if fred_key:
        with st.spinner("Fetching Macro Data..."):
            df_ff = de.get_fred_series("FEDFUNDS", fred_key)
            df_cpi = de.get_fred_series("CPIAUCSL", fred_key)
            df_m2 = de.get_fred_series("M2SL", fred_key)
            macro_regime = qe.get_macro_ml_regime(df_cpi, df_ff)
            correlations = qe.get_correlations(asset_info['ticker'], fred_key)
            
        macro_col_main, macro_col_ml = st.columns([2, 1]) # Consolidated for mobile
        with macro_col_main:
            # RESTORED: Inner tabs to fit 4 charts properly
            mt_inner_1, mt_inner_2 = st.tabs(["YIELD CURVE & RATES", "INFLATION & LIQUIDITY"])
            
            with mt_inner_1:
                c_m1, c_m2 = st.columns(2)
                with c_m1:
                    # Yield Curve
                    if not df_yield.empty:
                        curr_yield = df_yield['value'].iloc[-1]
                        yield_color = "#8080FF" if curr_yield < 0 else "#00FFFF"
                        try:
                            fig_yc = go.Figure()
                            fig_yc.add_trace(go.Scatter(x=df_yield.index, y=df_yield['value'], fill='tozeroy', fillcolor='rgba(102, 204, 255, 0.2)', line=dict(color=yield_color)))
                            fig_yc.add_hline(y=0, line_dash="dash", line_color="white")
                            fig_yc = terminal_chart_layout(fig_yc, title=f"10Y-2Y SPREAD: {curr_yield:.2f}%", height=200)
                            st.plotly_chart(fig_yc, use_container_width=True)
                        except: st.warning("Yield chart error")
                        st.caption("CONTEXT: " + ("⚠️ RECESSION SIGNAL" if curr_yield < 0 else "NORMAL GROWTH"))
                with c_m2:
                     # RESTORED: Fed Funds Chart
                    if not df_ff.empty:
                        try:
                            fig_ff = go.Figure()
                            fig_ff.add_trace(go.Scatter(x=df_ff.index, y=df_ff['value'], line=dict(color="#40E0FF")))
                            fig_ff = terminal_chart_layout(fig_ff, title=f"FED FUNDS: {df_ff['value'].iloc[-1]:.2f}%", height=200)
                            st.plotly_chart(fig_ff, use_container_width=True)
                        except: st.warning("Rates chart error")
                        st.caption("CONTEXT: BASELINE RISK-FREE RATE")

            with mt_inner_2:
                c_m3, c_m4 = st.columns(2)
                with c_m3:
                    # CPI
                    if not df_cpi.empty:
                        df_cpi['YoY'] = df_cpi['value'].pct_change(12) * 100
                        try:
                            fig_cpi = go.Figure()
                            fig_cpi.add_trace(go.Bar(x=df_cpi.index, y=df_cpi['YoY'], marker_color='#00FFFF'))
                            fig_cpi = terminal_chart_layout(fig_cpi, title=f"CPI (YoY): {df_cpi['YoY'].iloc[-1]:.2f}%", height=200)
                            st.plotly_chart(fig_cpi, use_container_width=True)
                        except: st.warning("CPI chart error")
                        st.caption("TARGET: 2.0%")
                with c_m4:
                     # M2
                    if not df_m2.empty:
                        try:
                            fig_m2 = go.Figure()
                            fig_m2.add_trace(go.Scatter(x=df_m2.index, y=df_m2['value'], line=dict(color="#00FFFF")))
                            fig_m2 = terminal_chart_layout(fig_m2, title="M2 LIQUIDITY", height=200)
                            st.plotly_chart(fig_m2, use_container_width=True)
                        except: st.warning("M2 chart error")
                        st.caption("CONTEXT: MONEY SUPPLY")

        with macro_col_ml:
            st.markdown("**MACRO ML ENGINE**")
            if macro_regime:
                st.markdown(f"""
                <div class='terminal-box'>
                    <div style='color:#AAAAAA; font-size:0.8em;'>ECONOMIC REGIME</div>
                    <div style='color:#00FFFF; font-size:1.1em; font-weight:bold;'>{macro_regime['regime']}</div>
                    <hr>
                    <div style='font-size:0.8em;'>RATES: {macro_regime['rate']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # DXY COMPARISON
            if not correlations.empty and 'Dollar' in correlations:
                dxy_corr = correlations['Dollar']
                corr_color = "#00FFFF" if dxy_corr > 0.5 else "#8080FF" if dxy_corr < -0.5 else "white"
                st.markdown(f"""
                <div class='terminal-box' style='margin-top:10px;'>
                    <div style='color:#AAAAAA; font-size:0.8em;'>DXY CORRELATION</div>
                    <div style='color:{corr_color}; font-size:1.5em;'>{dxy_corr:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Add `fred_api_key` to view Fed Macro Data.")

with macro_tab2:
    with st.spinner("Loading Intel..."):
        news_general = de.get_financial_news_general(news_key, query=asset_info.get('news_query', 'Finance'))
        news_ff = de.get_forex_factory_news(rapid_key, 'breaking')
        news_finnhub = de.get_finnhub_news(finnhub_key, category='general')
        eco_events = de.get_economic_calendar(rapid_key, use_demo=use_demo_data)
        news_sentiment_df = ai.calculate_news_sentiment(news_general[:5] + news_ff[:5] + news_finnhub[:5])

    col_eco, col_news = st.columns([1, 1])
    
    # (Helper functions moved to utils.py)

    with col_eco:
        st.markdown("**📅 HIGH IMPACT EVENTS (USD)**")
        if eco_events:
            cal_data = []
            for event in eco_events:
                context, bias = analyze_eco_context(event.get('actual', ''), event.get('forecast', ''), event.get('previous', ''))
                cal_data.append({"TIME": event.get('time', 'N/A'), "EVENT": event.get('event_name', 'Unknown'), "CONTEXT": context, "BIAS": bias})
            df_cal = pd.DataFrame(cal_data)
            def color_bias(val):
                color = '#CCCCCC'
                if 'Bullish' in val: color = '#00FFFF' 
                elif 'Bearish' in val: color = '#8080FF'
                return f'color: {color}'
            if not df_cal.empty: 
                st.dataframe(df_cal.style.map(color_bias, subset=['BIAS']), use_container_width=True, hide_index=True)
        else: st.info("No high impact events.")

    with col_news:
        st.markdown(f"**📰 {asset_info.get('news_query', 'LATEST')} WIRE**")
        if ai.HAS_NLP and not news_sentiment_df.empty:
             try:
                 fig_sent = go.Figure()
                 fig_sent.add_trace(go.Scatter(
                     x=news_sentiment_df.index, y=news_sentiment_df['cumulative'],
                     mode='lines+markers', line=dict(color='#00FFFF', width=2), name="Sentiment"))
                 fig_sent = terminal_chart_layout(fig_sent, title="SENTIMENT VELOCITY", height=150)
                 fig_sent.update_layout(xaxis=dict(showgrid=False, visible=False))
                 st.plotly_chart(fig_sent, use_container_width=True)
             except: st.warning("Sentiment chart error")
        
        # RESTORED: Tabs for General vs Forex Factory vs Finnhub
        tab_gen, tab_ff, tab_fh = st.tabs(["📰 GENERAL", "⚡ FOREX FACTORY", "📈 FINNHUB"])
        
        def render_news(items):
            if items:
                for news in items:
                    st.markdown(f"""
                    <div style="border-bottom:1px solid #333; padding-bottom:5px; margin-bottom:5px;">
                        <a class='news-link' href='{news['url']}' target='_blank'>▶ {news['title']}</a><br>
                        <span style='font-size:0.7em; color:#AAAAAA;'>{news['time']} | {news['source']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else: st.markdown("<div style='color:gray;'>No data.</div>", unsafe_allow_html=True)
        
        with tab_gen: render_news(news_general)
        with tab_ff: render_news(news_ff)
        with tab_fh: render_news(news_finnhub)

# ==============================================================================
# 4. STRATEGIC ANALYSIS (TREND & POSITIONING)
# ==============================================================================
st.markdown("---")
st.markdown("### 🔭 PHASE 2: STRATEGIC ANALYSIS (Trend & Positioning)")

strat_main_tab, strat_corr_tab = st.tabs(["🔭 STRATEGIC ANALYSIS", "🗺️ INTER-MARKET CORRELATIONS"])

with strat_corr_tab:
    with st.spinner("Calculating Correlations..."):
        corr_tickers = list(set([asset_info['ticker'], "GC=F", "^GSPC", "EURUSD=X", "BTC-USD", "^TNX"]))
        corr_returns = de.get_correlation_data(corr_tickers)
        corr_matrix = qe.calculate_correlation_matrix(corr_returns)
        
    if not corr_matrix.empty:
        try:
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale=[[0, '#8080FF'], [0.5, '#12161F'], [1, '#00FFFF']],
                zmin=-1, zmax=1,
                text=np.around(corr_matrix.values, decimals=2),
                texttemplate="%{text}",
                showscale=True
            ))
            fig_corr = terminal_chart_layout(fig_corr, title="180D ASSET CORRELATION MATRIX", height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("Context: 1.0 = Perfect Positive, -1.0 = Perfect Inverse. High absolute values indicate strong relationships.")
        except: st.warning("Correlation Matrix error")
    else:
        st.info("Correlation data unavailable.")

with strat_main_tab:
    with st.spinner("Analyzing Market Structure..."):
        intraday_data = de.get_intraday_data(asset_info['ticker'])
        dxy_data = de.get_dxy_data(fred_key)
        ms_df, ms_trend, ms_last_sh, ms_last_sl = qe.detect_market_structure(daily_data)
        active_fvgs = qe.detect_fair_value_gaps(daily_data)
        smc_data = qe.detect_smc_patterns(daily_data)
        vol_profile, poc_price = qe.calculate_volume_profile(intraday_data)
        val, vah = qe.calculate_value_area(vol_profile)
        
    strat_col1, strat_col2 = st.columns([2, 1])

    with strat_col1:
        # --- CHART: MULTILAYER LIQUIDITY MAP ---
        if not daily_data.empty:
            fig = go.Figure()
            
            # Trace 1: Candles
            fig.add_trace(go.Candlestick(
                x=daily_data.index, open=daily_data['Open'], high=high, low=low, close=close, name="Price",
                increasing_line_color="#00FFFF", decreasing_line_color="#405060"
            ))
            
            # Trace 2: FVG
            for fvg in active_fvgs:
                color = "rgba(0, 255, 255, 0.15)" if "Bullish" in fvg['type'] else "rgba(128, 128, 255, 0.15)"
                fig.add_shape(type="rect", x0=fvg['date'], x1=daily_data.index[-1], y0=fvg['bottom'], y1=fvg['top'], fillcolor=color, line_width=0)
    
            # Trace 3: Swing Points
            sh_mask = ms_df['Structure'] == 'SH'
            sl_mask = ms_df['Structure'] == 'SL'
            fig.add_trace(go.Scatter(x=ms_df[sh_mask].index, y=ms_df[sh_mask]['High'], mode='markers', marker=dict(symbol='triangle-down', size=8, color='#8080FF'), name='Swing High'))
            fig.add_trace(go.Scatter(x=ms_df[sl_mask].index, y=ms_df[sl_mask]['Low'], mode='markers', marker=dict(symbol='triangle-up', size=8, color='#00FFFF'), name='Swing Low'))
    
            # SMC OVERLAYS: Order Blocks
            if smc_data:
                for ob in smc_data['obs']:
                    ob_color = "rgba(0, 255, 255, 0.2)" if "Bullish" in ob['type'] else "rgba(128, 128, 255, 0.2)"
                    fig.add_shape(type="rect", x0=ob['date'], x1=daily_data.index[-1], y0=ob['bottom'], y1=ob['top'], fillcolor=ob_color, line_width=0, layer="below")
    
            # VOLUME PROFILE OVERLAYS
            if poc_price: 
                fig.add_hline(y=poc_price, line_dash="dash", line_color="#FFFFFF", annotation_text="POC", annotation_position="bottom right")
            if val and vah:
                fig.add_hrect(y0=val, y1=vah, fillcolor="rgba(255, 255, 255, 0.05)", line_width=0, layer="below")
    
            # Trace 4: DXY Overlay
            if not dxy_data.empty:
                dxy_aligned = dxy_data['Close'].reindex(daily_data.index, method='ffill')
                fig.add_trace(go.Scatter(x=dxy_aligned.index, y=dxy_aligned.values, name="DXY", line=dict(color='#8080FF', width=2), opacity=0.5, yaxis="y2"))
    
            try:
                fig = terminal_chart_layout(fig, height=500)
                fig.update_layout(
                    yaxis=dict(title="Price"),
                    yaxis2=dict(title="DXY", overlaying="y", side="right", showgrid=False, tickfont=dict(color="#8080FF")),
                    legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)")
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Main Liquidity Chart unavailable")
    
    with strat_col2:
        # --- MARKET STRUCTURE DETAILS ---
        trend_color = "bullish" if "BULLISH" in ms_trend else "bearish" if "BEARISH" in ms_trend else "neutral"
        st.markdown(f"""
        <div class='terminal-box'>
            <div style='color:#AAAAAA; font-size:0.8em;'>MARKET STRUCTURE (Trend)</div>
            <div style='font-size:1.2em; font-weight:bold;'>{ms_trend}</div>
            <hr style='margin:5px 0;'>
            <div style='font-size:0.8em;'>Last High: <span style='color:#8080FF'>{ms_last_sh:,.2f}</span></div>
            <div style='font-size:0.8em;'>Last Low: <span style='color:#00FFFF'>{ms_last_sl:,.2f}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- MONTE CARLO ---
        st.markdown("**🎲 PROBABILITY CONE (Distribution)**")
        if pred_dates is not None and pred_paths is not None:
            try:
                # Calculate Percentiles
                median_path = np.percentile(pred_paths, 50, axis=1)
                upper_95 = np.percentile(pred_paths, 95, axis=1)
                lower_5 = np.percentile(pred_paths, 5, axis=1)
                upper_80 = np.percentile(pred_paths, 80, axis=1)
                lower_20 = np.percentile(pred_paths, 20, axis=1)
                
                fig_pred = go.Figure()
                
                # 90% Confidence Interval
                fig_pred.add_trace(go.Scatter(x=pred_dates.tolist() + pred_dates.tolist()[::-1], 
                                           y=upper_95.tolist() + lower_5.tolist()[::-1],
                                           fill='toself', fillcolor='rgba(0, 255, 255, 0.1)',
                                           line=dict(color='rgba(255,255,255,0)'), name='90% Range'))
                
                # 60% Confidence Interval
                fig_pred.add_trace(go.Scatter(x=pred_dates.tolist() + pred_dates.tolist()[::-1], 
                                           y=upper_80.tolist() + lower_20.tolist()[::-1],
                                           fill='toself', fillcolor='rgba(0, 255, 255, 0.2)',
                                           line=dict(color='rgba(255,255,255,0)'), name='60% Range'))
                
                # Median Path
                fig_pred.add_trace(go.Scatter(x=pred_dates, y=median_path, name='Median', 
                                           line=dict(color='#00FFFF', width=2)))
                
                fig_pred = terminal_chart_layout(fig_pred, title="MC DISTRIBUTION (126D)", height=250)
                fig_pred.update_layout(showlegend=False)
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Probability of Profit (Simplified)
                final_prices = pred_paths[-1]
                prob_up = (final_prices > curr).mean() * 100
                st.markdown(f"<div style='font-size:0.75em; color:#AAAAAA; text-align:center;'>PROBABILITY OF UPSIDE: <span style='color:#00FFFF;'>{prob_up:.1f}%</span></div>", unsafe_allow_html=True)
                
            except Exception as e: 
                st.warning(f"Forecast error: {e}")
            
        # --- RESTORED: SEASONALITY TABS ---
        if seasonality_stats:
            st.markdown("**⏳ SEASONAL TENDENCIES**")
            tab_hour, tab_day, tab_week = st.tabs(["HOUR (NY)", "DAY", "WEEK"])
            
            with tab_hour:
                if 'hourly_perf' in seasonality_stats and seasonality_stats['hourly_perf'] is not None:
                    try:
                        hp = seasonality_stats['hourly_perf']
                        fig_h = go.Figure()
                        colors = ['#00FFFF' if v > 0 else '#8080FF' for v in hp.values]
                        fig_h.add_trace(go.Bar(x=[f"{h:02d}:00" for h in hp.index], y=hp.values, marker_color=colors))
                        fig_h = terminal_chart_layout(fig_h, title="AVG RETURN BY HOUR", height=200)
                        st.plotly_chart(fig_h, use_container_width=True)
                    except: st.warning("Hourly chart error")
            
            with tab_day:
                if 'day_high' in seasonality_stats:
                    fig_d = go.Figure()
                    fig_d.add_trace(go.Bar(x=seasonality_stats['day_high'].index, y=seasonality_stats['day_high'].values, marker_color='#00FFFF'))
                    fig_d = terminal_chart_layout(fig_d, title="PROB OF HIGH OF WEEK", height=200)
                    st.plotly_chart(fig_d, use_container_width=True)
    
            with tab_week:
                if 'week_returns' in seasonality_stats:
                     wr = seasonality_stats['week_returns']
                     fig_w = go.Figure()
                     colors = ['#00FFFF' if v > 0 else '#8080FF' for v in wr.values]
                     fig_w.add_trace(go.Bar(x=["Wk 1", "Wk 2", "Wk 3", "Wk 4", "Wk 5"], y=wr.values, marker_color=colors))
                     fig_w = terminal_chart_layout(fig_w, title="MONTHLY SEASONALITY", height=200)
                     st.plotly_chart(fig_w, use_container_width=True)

# --- 4B. COT & FUNDAMENTALS (RESTORED FULL DETAIL) ---
with st.expander("🏛️ INSTITUTIONAL POSITIONING (COT) & FUNDAMENTALS", expanded=True):
    with st.spinner("Fetching Institutional Data..."):
        cot_history = de.fetch_cot_history(selected_asset, start_year=2024)
        cg_id = asset_info.get('cg_id')
        cg_data = de.get_coingecko_stats(cg_id, cg_key) if cg_id else None
        
    # (Helper functions moved to utils.py)

    cot_col, fund_col = st.columns([2, 1])
    
    with cot_col:
        st.markdown("**COT FUTURES POSITIONING**")
        # 1. Fetch COT
        cot_history = de.fetch_cot_history(selected_asset, start_year=2024)
        if cot_history is not None and not cot_history.empty:
            cot_config = config.COT_MAPPING[selected_asset]
            spec_label, hedge_label = cot_config['labels']
            
            # Calculations
            cot_history['Net Speculator'] = cot_history['spec_long'] - cot_history['spec_short']
            cot_history['Net Hedger'] = cot_history['hedge_long'] - cot_history['hedge_short']
            cot_history['Spec Z-Score'] = qe.calculate_z_score(cot_history['Net Speculator'])
            latest_cot = cot_history.iloc[-1]
            prev_cot = cot_history.iloc[-2] if len(cot_history) > 1 else latest_cot
            
            # Capture for AI
            cot_data = {"sentiment": "BULLISH" if latest_cot['Net Speculator'] > 0 else "BEARISH", "net_spec": latest_cot['Net Speculator'], "z_score": latest_cot['Spec Z-Score']}
            
            # RESTORED: METRICS ROW
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(f"{spec_label} (Net)", f"{int(latest_cot['Net Speculator']):,}", f"{int(latest_cot['Net Speculator'] - prev_cot['Net Speculator']):,}")
            mc2.metric(f"{hedge_label} (Net)", f"{int(latest_cot['Net Hedger']):,}", f"{int(latest_cot['Net Hedger'] - prev_cot['Net Hedger']):,}", delta_color="inverse")
            mc3.metric("Z-Score", f"{latest_cot['Spec Z-Score']:.2f}σ", "Extreme" if abs(latest_cot['Spec Z-Score']) > 2 else "Neutral", delta_color="off")
            
            # RESTORED: Interpretation Box
            cot_txt = generate_cot_analysis(latest_cot['Net Speculator'], latest_cot['Net Hedger'], spec_label, hedge_label)
            st.info(cot_txt)

            # RESTORED: TABS FOR CHARTS
            tab_trend, tab_struct, tab_osc = st.tabs(["📈 NET TREND", "🦋 STRUCTURE", "📊 Z-SCORE"])
            
            with tab_trend:
                try:
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(x=cot_history['date'], y=cot_history['Net Speculator'], name=spec_label, line=dict(color='#00FFFF', width=2)))
                    fig_trend.add_trace(go.Scatter(x=cot_history['date'], y=cot_history['Net Hedger'], name=hedge_label, line=dict(color='#8080FF', width=2)))
                    fig_trend.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_trend = terminal_chart_layout(fig_trend, title="NET POSITIONING HISTORY", height=300)
                    st.plotly_chart(fig_trend, use_container_width=True)
                except: st.warning("COT Trend chart error")

            with tab_struct:
                try:
                    fig_struct = go.Figure()
                    fig_struct.add_trace(go.Bar(x=cot_history['date'], y=cot_history['spec_long'], name=f"{spec_label} Longs", marker_color='#00FFFF'))
                    fig_struct.add_trace(go.Bar(x=cot_history['date'], y=-cot_history['spec_short'], name=f"{spec_label} Shorts", marker_color='#8080FF'))
                    fig_struct.update_layout(barmode='overlay')
                    fig_struct = terminal_chart_layout(fig_struct, title="BUTTERFLY CHART (Long vs Short)", height=300)
                    st.plotly_chart(fig_struct, use_container_width=True)
                except: st.warning("COT Structure chart error")

            with tab_osc:
                try:
                    fig_z = go.Figure()
                    colors = ['#8080FF' if val > 2 or val < -2 else '#333' for val in cot_history['Spec Z-Score']]
                    fig_z.add_trace(go.Bar(x=cot_history['date'], y=cot_history['Spec Z-Score'], marker_color=colors))
                    fig_z.add_hline(y=2, line_dash="dot", line_color="#8080FF")
                    fig_z.add_hline(y=-2, line_dash="dot", line_color="#8080FF")
                    fig_z = terminal_chart_layout(fig_z, title="OSCILLATOR (Z-Score)", height=300)
                    st.plotly_chart(fig_z, use_container_width=True)
                except: st.warning("COT Oscillator chart error")

        else:
            st.info("COT Data unavailable for this asset.")

    with fund_col:
        cg_id = asset_info.get('cg_id')
        if cg_id and cg_key:
            st.markdown("**🦎 COINGECKO FUNDAMENTALS**")
            cg_data = de.get_coingecko_stats(cg_id, cg_key)
            if cg_data:
                st.metric("Market Rank", f"#{cg_data['rank']}")
                ath_color = "#8080FF" if cg_data['ath_change'] < -20 else "#00FFFF"
                st.markdown(f"**ATH Drawdown:** <span style='color:{ath_color}'>{cg_data['ath_change']:.2f}%</span>", unsafe_allow_html=True)
                sentiment_val = int(cg_data.get('sentiment', 50) or 50)
                st.progress(max(0, min(100, sentiment_val)))
                st.caption(f"Sentiment: {cg_data.get('sentiment', 50)}% Bullish")
                
                # RESTORED: Algo & Description
                st.markdown(f"**Algorithm:** `{cg_data['hashing']}`")
                with st.expander("Asset Description"):
                    st.write(cg_data['desc'])

# ==============================================================================
# 4C. FINNHUB INSIDER ACTIVITY (US COMPANIES)
# ==============================================================================
if finnhub_key:
    # Only try if we have an exact ticker (US stock) or attempt to use opt_ticker if it's SPY/QQQ
    _symbol = asset_info.get('ticker')
    if _symbol and not _symbol.startswith('^') and not '=' in _symbol and not '-' in _symbol:
        pass # It's likely a normal stock symbol
    elif asset_info.get('opt_ticker') and not '^' in asset_info.get('opt_ticker'):
        _symbol = asset_info.get('opt_ticker')
    else:
        _symbol = None
        
    if _symbol:
        with st.expander("🏢 FINNHUB INSIDER ACTIVITY & SENTIMENT (US COMPANIES)", expanded=False):
            with st.spinner("Fetching Finnhub Insider Data..."):
                to_date_str = datetime.now().strftime('%Y-%m-%d')
                from_date_str = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                
                insider_sent = de.get_finnhub_insider_sentiment(_symbol, from_date_str, to_date_str, finnhub_key)
                insider_trans = de.get_finnhub_insider_transactions(_symbol, finnhub_key)
                
            col_sent, col_trans = st.columns([1, 1])
            
            with col_sent:
                st.markdown(f"**INSIDER SENTIMENT ({_symbol})**")
                if not insider_sent.empty:
                    try:
                        insider_sent['date'] = pd.to_datetime(insider_sent['year'].astype(str) + '-' + insider_sent['month'].astype(str))
                        fig_isent = go.Figure()
                        colors = ['#00FFFF' if m > 0 else '#8080FF' for m in insider_sent['mspr']]
                        fig_isent.add_trace(go.Bar(x=insider_sent['date'], y=insider_sent['mspr'], marker_color=colors))
                        fig_isent = terminal_chart_layout(fig_isent, title="MONTHLY SHARE PURCHASE RATIO (MSPR)", height=250)
                        st.plotly_chart(fig_isent, use_container_width=True)
                        st.caption("MSPR ranges from -100 (Negative) to 100 (Positive). Signals price changes 30-90 days out.")
                    except:
                        st.warning("Could not render insider sentiment chart.")
                else:
                    st.info(f"No insider sentiment data available for {_symbol}.")
            
            with col_trans:
                st.markdown(f"**LATEST TRANSACTIONS ({_symbol})**")
                if not insider_trans.empty:
                    # Clean up transaction data for display
                    display_trans = []
                    for _, row in insider_trans.head(10).iterrows():
                        color = "#00FFFF" if row.get('change', 0) > 0 else "#8080FF"
                        t_type = "BUY" if row.get('change', 0) > 0 else "SELL"
                        display_trans.append({"Date": row.get('transactionDate', ''), "Name": row.get('name', ''), "Type": t_type, "Shares": row.get('change', 0), "Price": row.get('transactionPrice', 0)})
                    
                    df_disp = pd.DataFrame(display_trans)
                    st.dataframe(df_disp, hide_index=True, use_container_width=True)
                else:
                    st.info(f"No recent transactions found for {_symbol}.")

# ==============================================================================
# 5. MARKET DYNAMICS (VOLATILITY & FLOW)
# ==============================================================================
# --- PHASE 3: MARKET DYNAMICS ---
st.markdown("---")
st.markdown("### ⚙️ PHASE 3: MARKET DYNAMICS (Vol & Flow)")

with st.spinner("Calculating Volatility Surface..."):
    gex_df, gex_date, gex_spot, current_iv = qe.get_gex_profile(asset_info['opt_ticker'])
    # ADAPTIVE: Volatility cone window is now dynamic
    vol_cone = qe.get_volatility_cone(daily_data, windows=[int(adaptive_vol_window)])
    of_df, of_bias = qe.calculate_order_flow_proxy(daily_data)

dyn_col1, dyn_col2, dyn_col3 = st.columns(3)

with dyn_col1:
    # GEX
    if gex_df is not None:
        total_gex = gex_df['gex'].sum() / 1_000_000
        sent_color = "bullish" if total_gex > 0 else "bearish"
        st.markdown(f"""
        <div class='terminal-box'>
            <div style='color:#00FFFF;'>NET GAMMA EXPOSURE</div>
            <div style='font-size:1.5em; color:white;'>${total_gex:.1f}M</div>
            <span class='{sent_color}'>{"STICKY (Low Vol)" if total_gex > 0 else "SLIPPERY (High Vol)"}</span>
        </div>
        """, unsafe_allow_html=True)
        
        center_strike = gex_spot 
        gex_zoom = gex_df[(gex_df['strike'] > center_strike * 0.9) & (gex_df['strike'] < center_strike * 1.1)]
        try:
            fig_gex = go.Figure()
            colors = ['#00FFFF' if x > 0 else '#8080FF' for x in gex_zoom['gex']]
            fig_gex.add_trace(go.Bar(x=gex_zoom['strike'], y=gex_zoom['gex'], marker_color=colors))
            fig_gex.add_vline(x=center_strike, line_dash="dot", line_color="white")
            fig_gex = terminal_chart_layout(fig_gex, title="GAMMA PROFILE", height=200)
            st.plotly_chart(fig_gex, use_container_width=True)
        except: st.warning("Gamma Profile error")
    else: st.info("No Options Data")

with dyn_col2:
    # VOL CONE & IV
    if not daily_data.empty:
        hv_current = daily_data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
    else: hv_current = 0
    iv_display = current_iv if current_iv else 0
    vol_premium = iv_display - hv_current
    
    vol_regime = vol_cone.get('regime', 'N/A')
    vol_rank = vol_cone.get('rank', 0.5)
    
    st.markdown(f"""
    <div class='terminal-box'>
        <div style='color:#AAAAAA; font-size:0.8em;'>VOLATILITY SURFACE</div>
        <div style='font-size:1.1em; font-weight:bold;'>{vol_regime}</div>
        <div style='font-size:0.8em;'>Rank: {vol_rank*100:.0f}%</div>
        <hr style='margin:5px 0;'>
        <div style='font-size:0.8em; display:flex; justify-content:space-between;'>
            <span>IV (Expected):</span>
            <span style='color:#00FFFF;'>{iv_display:.1f}%</span>
        </div>
        <div style='font-size:0.8em; display:flex; justify-content:space-between;'>
            <span>HV (Realized):</span>
            <span style='color:#8080FF;'>{hv_current:.1f}%</span>
        </div>
        <div style='color:{"#8080FF" if vol_premium > 0 else "#00FFFF"}; font-size:0.85em; font-weight:bold; margin-top:2px;'>Prem/Disc: {vol_premium:+.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Expected Moves (Probabilistic Ranges)
    if iv_display > 0:
        range_1d = qe.calculate_implied_range(curr, iv_display, days=1)
        range_1w = qe.calculate_implied_range(curr, iv_display, days=7)
        
        st.markdown(f"""
<div class='terminal-box' style='margin-top:10px;'>
    <div style='color:#AAAAAA; font-size:0.75em; text-transform:uppercase;'>Implied Expected Moves</div>
    <div style='font-size:0.9em; margin-top:5px; color:#00FFFF;'>
        <b>1-DAY:</b> ±{range_1d['move']:,.2f} 
    </div>
    <div style='font-size:0.75em; color:gray;'>[{range_1d['lower_1sd']:,.0f} - {range_1d['upper_1sd']:,.0f}]</div>
    <div style='font-size:0.9em; margin-top:8px; color:#00FFFF;'>
        <b>1-WEEK:</b> ±{range_1w['move']:,.2f}
    </div>
    <div style='font-size:0.75em; color:gray;'>[{range_1w['lower_1sd']:,.0f} - {range_1w['upper_1sd']:,.0f}]</div>
    <div style='font-size:0.65em; color:gray; margin-top:8px; border-top:1px solid #1E252F; padding-top:4px;'>*68% Probability (1-Sigma)</div>
</div>
""", unsafe_allow_html=True)

with dyn_col3:
    # ORDER FLOW PROXY
    of_color = "bullish" if "Buying" in of_bias else "bearish"
    st.markdown(f"""
    <div class='terminal-box'>
        <div style='color:#AAAAAA; font-size:0.8em;'>ORDER FLOW PRESSURE</div>
        <div style='font-size:1.2em; font-weight:bold;' class='{of_color}'>{of_bias}</div>
        <div style='font-size:0.8em; color:gray;'>Volume-Weighted Impulse</div>
    </div>
    """, unsafe_allow_html=True)
    if vol_profile is not None:
         try:
             fig_vp = go.Figure()
             colors = ['#00FFFF' if x == poc_price else '#333' for x in vol_profile['PriceLevel']]
             fig_vp.add_trace(go.Bar(y=vol_profile['PriceLevel'], x=vol_profile['Volume'], orientation='h', marker_color='#40E0FF', opacity=0.4))
             fig_vp.add_hline(y=poc_price, line_dash="dash", line_color="#FFFFFF", annotation_text="POC")
             fig_vp = terminal_chart_layout(fig_vp, title="INTRADAY VOLUME PROFILE", height=200)
             st.plotly_chart(fig_vp, use_container_width=True)
         except: st.warning("Volume Profile error")

# ==============================================================================
# 6. TACTICAL EXECUTION (ENTRY/EXIT)
# ==============================================================================
st.markdown("---")
st.markdown("### 🎯 PHASE 4: TACTICAL EXECUTION")

exe_col1, exe_col2 = st.columns([2, 1])

with exe_col1:
    st.markdown("**SESSION VWAP + KEY LEVELS**")
    with st.spinner("Calculating VWAP..."):
        vwap_df = qe.calculate_vwap_bands(intraday_data)
        
    if not vwap_df.empty:
        try:
            fig_vwap = go.Figure()
            fig_vwap.add_trace(go.Candlestick(x=vwap_df.index, open=vwap_df['Open'], high=vwap_df['High'], low=vwap_df['Low'], close=vwap_df['Close'], name="Price", increasing_line_color="#00FFFF", decreasing_line_color="#405060"))
            fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['VWAP'], name="Session VWAP", line=dict(color='#FFFFFF', width=2)))
            fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Upper_Band_1'], name="+1 STD", line=dict(color='gray', width=1), opacity=0.3))
            fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Lower_Band_1'], name="-1 STD", line=dict(color='gray', width=1), opacity=0.3))
            
            if key_levels:
                fig_vwap.add_hline(y=key_levels['PDH'], line_dash="dot", line_color="#8080FF", annotation_text="PDH")
                fig_vwap.add_hline(y=key_levels['PDL'], line_dash="dot", line_color="#00FFFF", annotation_text="PDL")
                fig_vwap.add_hline(y=key_levels['Pivot'], line_width=1, line_color="#40E0FF", annotation_text="DAILY PIVOT")
                
            fig_vwap = terminal_chart_layout(fig_vwap, height=450)
            st.plotly_chart(fig_vwap, use_container_width=True)
        except: st.warning("VWAP chart error")

        # RESTORED: Key Levels Text List
        if key_levels:
            st.markdown("#### 🔑 ALGO LEVELS")
            cur_price = intraday_data['Close'].iloc[-1] if not intraday_data.empty else 0
            levels_list = [("R1 (Resist)", key_levels['R1']), ("PDH (High)", key_levels['PDH']), ("PIVOT", key_levels['Pivot']), ("PDL (Low)", key_levels['PDL']), ("S1 (Support)", key_levels['S1'])]
            
            c_lvl_cols = st.columns(3)
            for i, (name, price) in enumerate(levels_list):
                 dist = abs(price - cur_price) / cur_price
                 color = "#FFFF00" if dist < config.THRESHOLDS['LEVEL_PROXIMITY'] else "#8080FF" if price > cur_price else "#00FFFF"
                 with c_lvl_cols[i % 3]:
                     st.markdown(f"<div style='font-size:0.8em; color:gray;'>{name}</div><div style='color:{color}; font-family:monospace;'>{price:,.2f}</div>", unsafe_allow_html=True)

with exe_col2:
    st.markdown("**INTRADAY ALPHA (vs SPY)**")
    if not rs_data.empty:
        curr_rs = rs_data['RS_Score'].iloc[-1]
        rs_color = "#00FFFF" if curr_rs > 0 else "#8080FF"
        rs_text = "OUTPERFORMING" if curr_rs > 0 else "UNDERPERFORMING"
        st.markdown(f"<span style='color:{rs_color}; font-weight:bold;'>{rs_text}</span>", unsafe_allow_html=True)
        
        try:
            fig_rs = go.Figure()
            fig_rs.add_hline(y=0, line_color="#333", line_dash="dash")
            fig_rs.add_trace(go.Scatter(x=rs_data.index, y=rs_data['RS_Score'], mode='lines', line=dict(color=rs_color, width=2), fill='tozeroy', fillcolor='rgba(102, 204, 255, 0.2)'))
            fig_rs = terminal_chart_layout(fig_rs, height=150)
            st.plotly_chart(fig_rs, use_container_width=True)
        except: st.warning("RS chart error")
    
    st.markdown("**RISK / BACKTEST**")
    strat_perf = qe.run_strategy_backtest(asset_info['ticker'])
    if strat_perf:
        sig_color = "#00FFFF" if "LONG" in strat_perf['signal'] else "#8080FF"
        st.markdown(f"Signal: <span style='color:{sig_color}; font-weight:bold;'>{strat_perf['signal']}</span>", unsafe_allow_html=True)
        st.metric("Sharpe", f"{strat_perf['sharpe']:.2f}")
        
        # RESTORED: Equity Curve Chart
        try:
            ec_df = pd.DataFrame({"Strategy": strat_perf['equity_curve'], "Buy & Hold": strat_perf['df']['Cum_BnH']})
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(x=ec_df.index, y=ec_df['Buy & Hold'], name="Buy & Hold", line=dict(color='#8080FF', dash='dot')))
            fig_perf.add_trace(go.Scatter(x=ec_df.index, y=ec_df['Strategy'], name="Active Strat", line=dict(color='#00FFFF', width=2), fill='tozeroy', fillcolor='rgba(102, 204, 255, 0.1)'))
            fig_perf = terminal_chart_layout(fig_perf, title="STRATEGY EDGE", height=200)
            st.plotly_chart(fig_perf, use_container_width=True)
        except: st.warning("Backtest chart error")

# --- PHASE 5: UNCERTAINTY QUANTIFICATION & EXECUTION INTELLIGENCE ---
st.markdown("---")
st.markdown("### 🧠 PHASE 5: UNCERTAINTY QUANTIFICATION & EXECUTION INTELLIGENCE")

# NEW: PROBABILISTIC OUTCOME MATRIX
if outcome_probs:
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    st.markdown("**📊 PROBABILISTIC OUTCOME MATRIX (Empirical Probability)**")
    p_cols = st.columns(len(outcome_probs))
    for i, (target, prob) in enumerate(outcome_probs.items()):
        with p_cols[i]:
            p_color = "#00FFFF" if prob > 0.6 else "#8080FF" if prob < 0.4 else "white"
            st.markdown(f"""
            <div class='terminal-box' style='text-align:center; border-top: 3px solid {p_color};'>
                <div style='color:#AAAAAA; font-size:0.7em;'>{target.upper()}</div>
                <div style='font-size:1.5em; font-weight:bold; color:{p_color};'>{prob*100:.1f}%</div>
                <div style='font-size:0.7em; color:gray;'>Sizing Edge: {qe.calculate_kelly_criterion(prob)*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

if show_regen_warning:
    st.info(f"🔄 Asset changed to **{selected_asset}**. AI synthesis below may be stale. Regenerate to update.")

# Prepare Data for LLM
gex_summary = gex_df if gex_df is not None else None
combined_news_for_llm = news_general[:5] + news_ff[:5] if 'news_general' in locals() else []
news_text_summary = "\n".join([f"- {n['title']} ({n['source']})" for n in combined_news_for_llm])

if gemini_key:
    c_ai1, c_ai2 = st.columns([1, 4])
    with c_ai1:
        import time
        current_time = time.time()
        cooldown = config.THRESHOLDS['AI_COOLDOWN']
        time_since_last = current_time - st.session_state['last_ai_call_time']
        can_call = time_since_last > cooldown
        
        if st.button("📝 QUANTIFY UNCERTAINTY (Brief)", disabled=not can_call):
            if can_call:
                with st.spinner("Synthesizing..."):
                    st.session_state['last_ai_call_time'] = current_time
                    st.session_state['last_analyzed_asset'] = selected_asset # Update analyzed asset
                    narrative = ai.get_technical_narrative(
                        ticker=selected_asset, price=curr, daily_pct=pct, regime=regime_data,
                        ml_signal=ml_signal, gex_data=gex_summary, cot_data=cot_data,
                        levels=key_levels, macro_data=macro_context_data, api_key=gemini_key, 
                        model_name=config.GEMINI_MODEL_NAME, use_grounding=use_grounding,
                        ms_trend=ms_trend
                    )
                    st.session_state['narrative_cache'] = narrative
                    st.rerun()
            else:
                # LIVE COUNTDOWN UX
                countdown_placeholder = st.empty()
                for i in range(int(cooldown - time_since_last), 0, -1):
                    countdown_placeholder.warning(f"Throttling: Auto-unlock in {i}s...")
                    time.sleep(1)
                st.rerun()
                
        if st.button("🔎 GENERATE EXECUTION THESIS", disabled=not can_call):
             if can_call:
                with st.spinner("Writing Thesis..."):
                    st.session_state['last_ai_call_time'] = current_time
                    st.session_state['last_analyzed_asset'] = selected_asset
                    thesis_text = ai.generate_deep_dive_thesis(
                        ticker=selected_asset, price=curr, change=pct, regime=regime_data,
                        ml_signal=ml_signal, gex_data=gex_summary, cot_data=cot_data,
                        levels=key_levels, news_summary=news_text_summary, macro_data=macro_context_data, 
                        api_key=gemini_key, model_name=config.GEMINI_MODEL_NAME, use_grounding=use_grounding,
                        ms_trend=ms_trend
                    )
                    st.session_state['thesis_cache'] = thesis_text
                    st.rerun()
             else:
                # Same live countdown logic
                countdown_placeholder = st.empty()
                for i in range(int(cooldown - time_since_last), 0, -1):
                    countdown_placeholder.warning(f"Throttling: Auto-unlock in {i}s...")
                    time.sleep(1)
                st.rerun()

        # PDF EXPORT
        st.markdown("---")
        if st.session_state['narrative_cache'] or st.session_state['thesis_cache']:
            # --- CAPTURE CHART FOR PDF (Robust Fallback) ---
            chart_bytes = None
            if fig is not None:
                try:
                    import plotly.io as pio
                    # Use a specific template for PDF export to ensure high contrast
                    pdf_fig = go.Figure(fig) # Copy the current figure
                    pdf_fig.update_layout(template="plotly_dark", plot_bgcolor='black', paper_bgcolor='black')
                    chart_bytes = pio.to_image(pdf_fig, format="png", engine="kaleido", width=1000, height=500)
                except Exception as e:
                    st.warning(f"Chart capture failed: {str(e)[:100]}. Proceeding with text-only brief.")
                    chart_bytes = None

            pdf_data = {
                "ticker": selected_asset, "price": curr, "pct": pct,
                "ml_signal": ml_signal, "regime": regime_data['regime'] if regime_data else "N/A",
                "ms_trend": ms_trend,
                "narrative": st.session_state['narrative_cache'],
                "thesis": st.session_state['thesis_cache'],
                "levels": key_levels,
                "chart_image": chart_bytes
            }
            pdf_bytes = generate_pdf_report(pdf_data)
            st.download_button(
                label="📥 DOWNLOAD INSTITUTIONAL BRIEF",
                data=pdf_bytes,
                file_name=f"Terminal_Brief_{selected_asset}_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

    with c_ai2:
        if st.session_state['narrative_cache']:
             st.markdown(f"""
            <div class='terminal-box' style='border-left: 4px solid #00FFFF; margin-bottom:10px;'>
                <div style='font-family: monospace; white-space: pre-wrap;'>{st.session_state['narrative_cache']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        if st.session_state['thesis_cache']:
            st.markdown(f"""
            <div class='terminal-box' style='padding: 20px; font-family: Georgia, serif; line-height: 1.7;'>
                <div style='white-space: pre-wrap; font-size: 0.9em;'>{st.session_state['thesis_cache']}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("Add GEMINI_API_KEY to enable AI Synthesis.")
