import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from datetime import datetime
from utils import safe_yf_download
from data_engine import get_fred_series

# --- 1. ADVANCED VOLATILITY MODELING ---
def calculate_garman_klass_vol(df, window=20, trading_periods=252):
    """
    Garman-Klass Volatility: More efficient than Close-to-Close volatility 
    as it considers High, Low, Open, and Close.
    """
    try:
        log_hl = np.log(df['High'] / df['Low']) ** 2
        log_co = np.log(df['Close'] / df['Open']) ** 2
        var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return np.sqrt(var.rolling(window=window).mean() * trading_periods)
    except: return pd.Series(dtype=float)

def get_volatility_cone(df, windows=[5, 20, 60]):
    """Calculates volatility percentiles to identify if current vol is cheap/expensive."""
    if df.empty: return {}
    res = {}
    current_vol = calculate_garman_klass_vol(df).iloc[-1]
    
    # Calculate historical quantile cones
    for w in windows:
        hist_vol = calculate_garman_klass_vol(df, window=w)
        res[f'min_{w}'] = hist_vol.min()
        res[f'max_{w}'] = hist_vol.max()
        res[f'med_{w}'] = hist_vol.median()
        res[f'curr_{w}'] = hist_vol.iloc[-1]
    
    # Statistical Regime
    rank = (current_vol - res['min_20']) / (res['max_20'] - res['min_20'])
    if rank < 0.2:
        regime = "COMPRESSION (Volatility Squeeze)"
    elif rank > 0.8:
        regime = "EXPANDED (Volatility Mean-Reversion)"
    else:
        regime = "STABLE (Trend Environment)"
    return {"data": res, "regime": regime, "rank": rank}

# --- 2. MARKET STRUCTURE & LIQUIDITY ---
def detect_market_structure(df, window=5):
    """
    Identifies Swing Highs (SH) and Swing Lows (SL) to map structure.
    Returns the dataframe with 'Structure' column.
    """
    df = df.copy()
    df['Swing_High'] = df['High'].rolling(window=window*2+1, center=True).max() == df['High']
    df['Swing_Low'] = df['Low'].rolling(window=window*2+1, center=True).min() == df['Low']
    
    structure = []
    last_high = df['High'].iloc[0]
    last_low = df['Low'].iloc[0]
    
    for i in range(len(df)):
        if df['Swing_High'].iloc[i]:
            structure.append("SH") # Swing High
            last_high = df['High'].iloc[i]
        elif df['Swing_Low'].iloc[i]:
            structure.append("SL") # Swing Low
            last_low = df['Low'].iloc[i]
        else:
            structure.append(np.nan)
            
    df['Structure'] = structure
    
    # Determine Trend
    # If Price > Last Swing High -> Bullish BOS (Break of Structure)
    # If Price < Last Swing Low -> Bearish BOS
    current_close = df['Close'].iloc[-1]
    
    # Find last confirmed points
    try:
        last_sh = df[df['Swing_High']]['High'].iloc[-1]
        last_sl = df[df['Swing_Low']]['Low'].iloc[-1]
        
        trend = "NEUTRAL"
        if current_close > last_sh: 
            trend = "BULLISH (BOS)"
        elif current_close < last_sl: 
            trend = "BEARISH (BOS)"
        else: 
            # In an internal range, check Accumulation vs Distribution
            range_height = last_sh - last_sl
            if range_height > 0:
                pos_in_range = (current_close - last_sl) / range_height
                
                # Use recent volume trend to confirm bias
                recent_vol = df['Volume'].tail(5).mean()
                avg_vol = df['Volume'].tail(20).mean()
                vol_increasing = recent_vol > avg_vol
                
                bias = "Neutral Chop"
                if pos_in_range > 0.6 and vol_increasing:
                    bias = "Accumulation (Testing Highs)"
                elif pos_in_range < 0.4 and vol_increasing:
                    bias = "Distribution (Testing Lows)"
                    
                trend = f"RANGE [{last_sl:,.2f} - {last_sh:,.2f}] ({bias})"
            else:
                trend = "CONSOLIDATION (Internal Range)"
        
        return df, trend, last_sh, last_sl
    except:
        return df, "UNCERTAIN", 0, 0

def detect_fair_value_gaps(df):
    """
    Identifies Fair Value Gaps (FVG) / Imbalances.
    Bullish FVG: Candle 1 High < Candle 3 Low
    Bearish FVG: Candle 1 Low > Candle 3 High
    """
    fvgs = []
    # Loop through last 50 candles
    lookback = 50
    subset = df.iloc[-lookback:]
    
    for i in range(len(subset) - 2):
        # Bullish FVG
        if subset['Low'].iloc[i+2] > subset['High'].iloc[i]:
            fvgs.append({
                "type": "Bullish FVG",
                "top": subset['Low'].iloc[i+2],
                "bottom": subset['High'].iloc[i],
                "date": subset.index[i+1]
            })
        # Bearish FVG
        elif subset['High'].iloc[i+2] < subset['Low'].iloc[i]:
             fvgs.append({
                "type": "Bearish FVG",
                "top": subset['Low'].iloc[i],
                "bottom": subset['High'].iloc[i+2],
                "date": subset.index[i+1]
            })
            
    # Filter only unmitigated (active) FVGs could be complex, 
    # returning last 3 significant ones for display
    return fvgs[-3:] if fvgs else []

def detect_smc_patterns(df, window=5):
    """
    Detects SMC (Smart Money Concepts) patterns:
    - Order Blocks (OB): High-probability reversal zones.
    - Liquidity Sweeps: Stop hunts at key highs/lows.
    """
    if len(df) < 20: return {"obs": [], "sweeps": []}
    
    obs = []
    # Simplified OB Detection: Last counter-trend candle before a change in structure
    for i in range(5, len(df) - 5):
        # Bullish OB: Bearish candle before a strong impulsive up move
        if df['Close'].iloc[i] < df['Open'].iloc[i]:
            if df['Close'].iloc[i+1] > df['High'].iloc[i] and df['Close'].iloc[i+3] > df['High'].iloc[i+1]:
                obs.append({
                    "type": "Bullish OB",
                    "top": df['High'].iloc[i],
                    "bottom": df['Low'].iloc[i],
                    "date": df.index[i]
                })
        # Bearish OB: Bullish candle before a strong impulsive down move
        elif df['Close'].iloc[i] > df['Open'].iloc[i]:
            if df['Close'].iloc[i+1] < df['Low'].iloc[i] and df['Close'].iloc[i+3] < df['Low'].iloc[i+1]:
                obs.append({
                    "type": "Bearish OB",
                    "top": df['High'].iloc[i],
                    "bottom": df['Low'].iloc[i],
                    "date": df.index[i]
                })

    sweeps = []
    # Simplified Sweep Detection: Price wick goes beyond recent SH/SL but body stays within
    for i in range(10, len(df)):
        lookback = df.iloc[i-10:i]
        prev_high = lookback['High'].max()
        prev_low = lookback['Low'].min()
        
        # High Sweep
        if df['High'].iloc[i] > prev_high and df['Close'].iloc[i] < prev_high:
            sweeps.append({"type": "High Sweep", "price": df['High'].iloc[i], "date": df.index[i]})
        # Low Sweep
        if df['Low'].iloc[i] < prev_low and df['Close'].iloc[i] > prev_low:
            sweeps.append({"type": "Low Sweep", "price": df['Low'].iloc[i], "date": df.index[i]})

    return {"obs": obs[-5:], "sweeps": sweeps[-3:]}

def calculate_value_area(vol_profile, pct=0.7):
    """Calculates the Value Area High (VAH) and Low (VAL) for a volume profile."""
    if vol_profile is None or vol_profile.empty: return None, None
    
    total_vol = vol_profile['Volume'].sum()
    target_vol = total_vol * pct
    
    poc_idx = vol_profile['Volume'].idxmax()
    poc_vol = vol_profile.loc[poc_idx, 'Volume']
    
    current_vol = poc_vol
    low_idx = poc_idx
    high_idx = poc_idx
    
    while current_vol < target_vol:
        prev_low_vol = vol_profile.loc[low_idx-1, 'Volume'] if low_idx > 0 else 0
        next_high_vol = vol_profile.loc[high_idx+1, 'Volume'] if high_idx < len(vol_profile)-1 else 0
        
        if prev_low_vol > next_high_vol:
            current_vol += prev_low_vol
            low_idx -= 1
        else:
            current_vol += next_high_vol
            high_idx += 1
            
        if low_idx == 0 and high_idx == len(vol_profile)-1: break
            
    val = vol_profile.loc[low_idx, 'PriceLevel']
    vah = vol_profile.loc[high_idx, 'PriceLevel']
    return val, vah

# --- 3. ORDER FLOW & STATS ---
def calculate_order_flow_proxy(df):
    """
    Approximates Buying vs Selling pressure using High-Low position relative to Close.
    (Money Flow approximation).
    """
    df['MF_Multiplier'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    df['MF_Vol'] = df['MF_Multiplier'] * df['Volume']
    
    # Cumulative Volume Delta (Proxy)
    df['CVD'] = df['MF_Vol'].cumsum()
    
    # Flow Sentiment (Last 5 periods)
    recent_flow = df['MF_Vol'].tail(5).sum()
    bias = "Aggressive Buying" if recent_flow > 0 else "Aggressive Selling"
    
    return df, bias

# --- 4. ADVANCED MOMENTUM & MACRO RISK ---
def detect_momentum_deterioration(df):
    """
    Identifies if momentum is fading even if price is still rising.
    Returns a score (0-100) and a description.
    """
    if len(df) < 30: return 0, "Insufficient Data"
    
    score = 0
    reasons = []
    
    # 1. RSI Divergence (Simplified)
    # Price makes higher high, RSI makes lower high over last 20 periods
    recent = df.tail(20)
    p_high_1 = recent['High'].iloc[:10].max()
    p_high_2 = recent['High'].iloc[10:].max()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    r_high_1 = df['RSI'].tail(20).iloc[:10].max()
    r_high_2 = df['RSI'].tail(20).iloc[10:].max()
    
    if p_high_2 > p_high_1 and r_high_2 < r_high_1:
        score += 40
        reasons.append("Bearish RSI Divergence")
        
    # 2. Momentum Slope (ROC of ROC)
    df['Mom'] = df['Close'].pct_change(10)
    mom_slope = df['Mom'].tail(5).diff().mean()
    if df['Mom'].iloc[-1] > 0 and mom_slope < 0:
        score += 30
        reasons.append("Momentum Deceleration (ROC Slope < 0)")
        
    # 3. Volume-Price Divergence (Churning check)
    vol_sma = df['Volume'].rolling(20).mean()
    price_move = abs(df['Close'].iloc[-1] - df['Close'].iloc[-5])
    vol_impulse = df['Volume'].tail(5).mean() / vol_sma.iloc[-1]
    
    if price_move < (df['Close'].iloc[-1] * 0.005) and vol_impulse > 1.5:
        score += 25
        reasons.append("Institutional Churning (High Vol, Low Price Progress)")

    # 4. EMA Slope Decay
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    df['Gap'] = (df['EMA20'] - df['EMA50']) / df['EMA50']
    gap_slope = df['Gap'].tail(5).diff().mean()
    
    if df['Gap'].iloc[-1] > 0 and gap_slope < 0:
        score += 25
        reasons.append("EMA Compression (Trend Convergence)")

    status = "STABLE"
    if score >= 60: status = "CRITICAL DETERIORATION"
    elif score >= 30: status = "MODERATE WEAKNESS"
    
    return score, status, reasons

@st.cache_data(ttl=3600)
def calculate_macro_pressure(api_key):
    """
    Aggregates macro-economic stress factors: Real Rates, Credit Spreads, and DXY.
    """
    if not api_key: return None
    
    try:
        # Fetch Key Macro Stress Indicators
        # 1. 10Y Real Interest Rate (TIPS) - Rising real rates pressure risk assets
        real_rate_df = get_fred_series("DFII10", api_key)
        # 2. High Yield Credit Spread - Widening spreads = financial stress
        hy_spread_df = get_fred_series("BAMLH0A0HYM2", api_key)
        # 3. Dollar Index Proxy
        dxy_df = get_fred_series("DTWEXAFEGS", api_key)
        
        if real_rate_df.empty or hy_spread_df.empty or dxy_df.empty:
            return None
            
        curr_real_rate = real_rate_df['value'].iloc[-1]
        prev_real_rate = real_rate_df['value'].iloc[-5] # 1 week ago
        
        curr_hy_spread = hy_spread_df['value'].iloc[-1]
        prev_hy_spread = hy_spread_df['value'].iloc[-5]
        
        curr_dxy = dxy_df['value'].iloc[-1]
        prev_dxy = dxy_df['value'].iloc[-5]
        
        # Scoring logic (Higher = More Pressure)
        pressure_score = 0
        factors = []
        
        # Real Rate Pressure
        if curr_real_rate > 2.0: 
            pressure_score += 30
            factors.append(f"Restrictive Real Rates ({curr_real_rate:.2f}%)")
        elif curr_real_rate > prev_real_rate:
            pressure_score += 15
            factors.append("Rising Real Rates (Liquidity Tightening)")
            
        # Credit Stress
        if curr_hy_spread > 4.5:
            pressure_score += 40
            factors.append(f"High Credit Stress (Spread: {curr_hy_spread:.2f})")
        elif curr_hy_spread > prev_hy_spread:
            pressure_score += 20
            factors.append("Widening Credit Spreads (Deleveraging Risk)")
            
        # Dollar Pressure
        if curr_dxy > prev_dxy:
            pressure_score += 15
            factors.append("Strong Dollar (USD Liquidity Drain)")
            
        status = "LOW"
        if pressure_score >= 60: status = "HIGH (Institutional De-risking)"
        elif pressure_score >= 30: status = "MODERATE (Macro Headwinds)"
        
        return {
            "score": pressure_score,
            "status": status,
            "factors": factors,
            "real_rate": curr_real_rate,
            "hy_spread": curr_hy_spread,
            "dxy": curr_dxy
        }
    except:
        return None

# --- RETAINING ORIGINAL FEATURES (DO NOT REMOVE) ---

# --- MATH ---
def calculate_hurst(series, lags=range(2, 20)):
    try:
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except: return 0.5

def calculate_z_score(series, window=52):
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    return (series - roll_mean) / roll_std

# --- ML REGIMES ---
@st.cache_data(ttl=3600)
def get_market_regime(ticker):
    try:
        df = safe_yf_download(ticker, period="5y", interval="1d")
        if df.empty or len(df) < 50: 
             # Fallback: Basic Volatility Regime
             return {"regime": "NEUTRAL (Establishing)", "color": "neutral", "confidence": 0.5}
             
        data = df.copy()
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(20).std()
        data = data.dropna()
        
        try:
            X = data[['Returns', 'Volatility']].values
            gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
            gmm.fit(X)
            current_state = gmm.predict(X[[-1]])[0]
            probs = gmm.predict_proba(X[[-1]])[0]
            means = gmm.means_
            state_order = np.argsort(means[:, 1]) 
            regime_map = {state_order[0]: "LOW VOL", state_order[1]: "NEUTRAL (Chop)", state_order[2]: "HIGH VOL (Crisis)"}
            
            # Refine LOW VOL state
            regime_desc = regime_map.get(current_state, "NEUTRAL (Transitional)")
            
            # Secondary check: Is it Trending or Compressing?
            if regime_desc == "LOW VOL":
                returns_vol = data['Returns'].std()
                # If vol is exceptionally low, it's a squeeze/compression
                if data['Volatility'].iloc[-1] < returns_vol * 0.6:
                    regime_desc = "COMPRESSION (Volatility Squeeze)"
                else:
                    # Check for price progress to confirm trend
                    price_progress = abs(data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
                    if price_progress > returns_vol * 2: # Significant progress relative to vol
                        regime_desc = "LOW VOL (Trend)"
                    else:
                        regime_desc = "LOW VOL (Range-Bound)"
                 
            color = "bullish" if "Trend" in regime_desc or "LOW VOL" in regime_desc else "bearish" if "HIGH VOL" in regime_desc else "neutral"
            return {"regime": regime_desc, "color": color, "confidence": max(probs)}
        except:
            # Fallback
            current_vol = data['Volatility'].iloc[-1]
            avg_vol = data['Volatility'].mean()
            if current_vol > avg_vol * 1.5:
                return {"regime": "HIGH VOL (Risk-Off)", "color": "bearish", "confidence": 0.7}
            elif current_vol < avg_vol * 0.6:
                return {"regime": "COMPRESSION (Range)", "color": "neutral", "confidence": 0.7}
            else:
                return {"regime": "STABLE (Chop)", "color": "neutral", "confidence": 0.6}
    except: 
        return {"regime": "STABLE (Neutral)", "color": "neutral", "confidence": 0.5}

@st.cache_data(ttl=86400)
def get_macro_ml_regime(cpi_df, rate_df):
    if cpi_df.empty or rate_df.empty: return None
    try:
        df = pd.merge(cpi_df, rate_df, left_index=True, right_index=True, how='inner')
        df.columns = ['CPI', 'Rates']
        df['CPI_YoY'] = df['CPI'].pct_change(12) * 100
        df = df.dropna()
        X = df[['CPI_YoY', 'Rates']].values
        if len(X) < 12: return None
        gmm = GaussianMixture(n_components=4, random_state=42)
        gmm.fit(X)
        curr_cpi = df['CPI_YoY'].iloc[-1]
        curr_rate = df['Rates'].iloc[-1]
        regime_name = "Neutral"
        if curr_cpi > 4 and curr_rate < curr_cpi: regime_name = "INFLATIONARY (Neg Real Rates)"
        elif curr_cpi > 4 and curr_rate > curr_cpi: regime_name = "TIGHTENING (Pos Real Rates)"
        elif curr_cpi < 2: regime_name = "DEFLATIONARY / RISK OFF"
        else: regime_name = "GOLDILOCKS / STABLE"
        return {"regime": regime_name, "cpi": curr_cpi, "rate": curr_rate}
    except: return None

@st.cache_data(ttl=3600)
def get_ml_prediction(ticker):
    try:
        df = safe_yf_download(ticker, period="2y", interval="1d") 
        if df.empty: return None, 0.5
        data = df.copy()
        data['Returns'] = data['Close'].pct_change()
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        data['Vol_5d'] = data['Returns'].rolling(5).std()
        data['Mom_5d'] = data['Close'].pct_change(5)
        data = data.dropna()
        if len(data) < 50: return None, 0.5
        X = data[['Vol_5d', 'Mom_5d']]
        y = data['Target']
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X, y)
        prob_up = model.predict_proba(X.iloc[[-1]])[0][1]
        return model, prob_up
    except: return None, 0.5

# --- GAMMA EXPOSURE ---
def calculate_black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

@st.cache_data(ttl=3600)
def get_gex_profile(opt_ticker):
    if opt_ticker is None: return None, None, None, None
    try:
        tk = yf.Ticker(opt_ticker)
        try:
            hist = tk.history(period="1d")
            spot_price = hist['Close'].iloc[-1] if not hist.empty else tk.fast_info.last_price
        except: return None, None, None, None
        
        if spot_price is None: return None, None, None, None
        exps = tk.options
        if not exps: return None, None, None, None
        
        target_exp = exps[1] if len(exps) > 1 else exps[0]
        chain = tk.option_chain(target_exp)
        calls, puts = chain.calls, chain.puts
        
        if calls.empty or puts.empty: return None, None, None, None
        
        # Calculate ATM IV
        atm_mask = (calls['strike'] > spot_price * 0.95) & (calls['strike'] < spot_price * 1.05)
        atm_calls = calls[atm_mask]
        avg_iv = atm_calls['impliedVolatility'].mean() * 100 if not atm_calls.empty else 0
        
        exp_date = datetime.strptime(target_exp, "%Y-%m-%d")
        days_to_exp = (exp_date - datetime.now()).days
        T = 0.001 if days_to_exp <= 0 else days_to_exp / 365.0
        gex_data = []
        strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
        for K in strikes:
            if K < spot_price * 0.75 or K > spot_price * 1.25: continue
            c_row = calls[calls['strike'] == K]
            p_row = puts[puts['strike'] == K]
            c_oi = c_row['openInterest'].iloc[0] if not c_row.empty else 0
            p_oi = p_row['openInterest'].iloc[0] if not p_row.empty else 0
            c_iv = c_row['impliedVolatility'].iloc[0] if not c_row.empty and 'impliedVolatility' in c_row.columns else 0.2
            p_iv = p_row['impliedVolatility'].iloc[0] if not p_row.empty and 'impliedVolatility' in p_row.columns else 0.2
            c_gamma = calculate_black_scholes_gamma(spot_price, K, T, 0.045, c_iv)
            p_gamma = calculate_black_scholes_gamma(spot_price, K, T, 0.045, p_iv)
            net_gex = (c_gamma * c_oi - p_gamma * p_oi) * spot_price * 100
            gex_data.append({"strike": K, "gex": net_gex})
            
        df = pd.DataFrame(gex_data, columns=['strike', 'gex'])
        return df, target_exp, spot_price, avg_iv
    except: return None, None, None, None

def calculate_volume_profile(df, bins=50):
    if df.empty: return None, None
    price_range = df['High'].max() - df['Low'].min()
    if price_range == 0: return None, None
    bin_size = price_range / bins
    df['PriceBin'] = ((df['Close'] - df['Low'].min()) // bin_size).astype(int)
    vol_profile = df.groupby('PriceBin')['Volume'].sum().reset_index()
    vol_profile['PriceLevel'] = df['Low'].min() + (vol_profile['PriceBin'] * bin_size)
    poc_idx = vol_profile['Volume'].idxmax()
    return vol_profile, vol_profile.loc[poc_idx, 'PriceLevel']

@st.cache_data(ttl=3600)
def get_seasonality_stats(daily_data, ticker_name):
    stats = {}
    try:
        df = daily_data.copy()
        df['Week_Num'] = df.index.to_period('W')
        high_days = df.groupby('Week_Num')['High'].idxmax().apply(lambda x: df.loc[x].name.day_name())
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        stats['day_high'] = high_days.value_counts().reindex(days_order, fill_value=0) / len(high_days) * 100
        df['Day'] = df.index.day
        df['Month_Week'] = np.ceil(df['Day'] / 7).astype(int)
        df['Returns'] = df['Close'].pct_change()
        stats['week_returns'] = df.groupby('Month_Week')['Returns'].mean() * 100
        try:
            intra = safe_yf_download(ticker_name, period="60d", interval="1h")
            if not intra.empty:
                if intra.index.tz is None: intra.index = intra.index.tz_localize('UTC')
                intra.index = intra.index.tz_convert('America/New_York')
                intra['Hour'] = intra.index.hour
                intra['Return'] = intra['Close'].pct_change()
                target_hours = [2,3,4,5,6, 8,9,10,11, 14,15,16,17,18, 20,21,22,23]
                stats['hourly_perf'] = intra[intra['Hour'].isin(target_hours)].groupby('Hour')['Return'].mean() * 100
        except: stats['hourly_perf'] = None
        return stats
    except: return None

@st.cache_data(ttl=3600)
def generate_monte_carlo(stock_data, days=126, simulations=1000):
    if stock_data is None or stock_data.empty or len(stock_data) < 2: return None, None
    try:
        close = stock_data['Close']
        log_returns = np.log(1 + close.pct_change())
        u, var = log_returns.mean(), log_returns.var()
        drift = u - (0.5 * var)
        stdev = log_returns.std()
        price_paths = np.zeros((days + 1, simulations))
        price_paths[0] = close.iloc[-1]
        daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (days, simulations)))
        for t in range(1, days + 1): price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]
        return pd.date_range(start=close.index[-1], periods=days + 1, freq='B'), price_paths
    except: return None, None

def calculate_technical_radar(df):
    if df.empty or len(df) < 30: return None
    data = df.copy()
    close = data['Close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    k = close.ewm(span=12, adjust=False, min_periods=12).mean()
    d = close.ewm(span=26, adjust=False, min_periods=26).mean()
    data['MACD'] = k - d
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()
    data['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    data['EMA_50'] = close.ewm(span=50, adjust=False).mean()
    last = data.iloc[-1]
    signals = {}
    
    if last['RSI'] < 30: signals['RSI'] = {"val": f"{last['RSI']:.0f}", "bias": "OVERSOLD (Bull)", "col": "bullish"}
    elif last['RSI'] > 70: signals['RSI'] = {"val": f"{last['RSI']:.0f}", "bias": "OVERBOUGHT (Bear)", "col": "bearish"}
    else: signals['RSI'] = {"val": f"{last['RSI']:.0f}", "bias": "NEUTRAL", "col": "neutral"}
    
    macd_hist = last['MACD'] - last['MACD_Signal']
    if macd_hist > 0 and last['MACD'] > 0: signals['MACD'] = {"val": f"{macd_hist:.2f}", "bias": "BULLISH", "col": "bullish"}
    elif macd_hist < 0 and last['MACD'] < 0: signals['MACD'] = {"val": f"{macd_hist:.2f}", "bias": "BEARISH", "col": "bearish"}
    else: signals['MACD'] = {"val": f"{macd_hist:.2f}", "bias": "NEUTRAL", "col": "neutral"}
    
    ema_gap = abs(last['EMA_20'] - last['EMA_50']) / last['EMA_50']
    
    if ema_gap < 0.005: # Less than 0.5% gap
        signals['Trend'] = {"val": "Compression", "bias": "TIGHTENING (Squeeze)", "col": "neutral"}
    elif last['Close'] > last['EMA_20'] and last['EMA_20'] > last['EMA_50']:
        signals['Trend'] = {"val": "Uptrend", "bias": "STRONG BULL", "col": "bullish"}
    elif last['Close'] < last['EMA_20'] and last['EMA_20'] < last['EMA_50']:
        signals['Trend'] = {"val": "Downtrend", "bias": "STRONG BEAR", "col": "bearish"}
    else: 
        signals['Trend'] = {"val": "Chop", "bias": "WEAK/MIXED", "col": "neutral"}
    return signals

@st.cache_data(ttl=3600)
def get_correlations(base_ticker, api_key):
    try:
        if not api_key: return pd.Series()
        
        tickers = {"Base": base_ticker, "VIX": "^VIX", "10Y Yield": "^TNX", "Gold": "GC=F"}
        unique_tickers = list(set(tickers.values()))
        yf_data = safe_yf_download(unique_tickers, period="6mo", interval="1d")
        
        fred_data = get_fred_series("DTWEXAFEGS", api_key) 
        
        if yf_data.empty: return pd.Series()
        
        if isinstance(yf_data.columns, pd.MultiIndex): 
            if 'Close' in yf_data.columns.get_level_values(0):
                 yf_df = yf_data.xs('Close', axis=1, level=0)
            else:
                 yf_df = yf_data['Close'].copy()
        elif 'Close' in yf_data.columns: 
            yf_df = yf_data['Close'].copy()
        else: 
            yf_df = yf_data.copy()

        if isinstance(yf_df, pd.Series): yf_df = yf_df.to_frame(name=unique_tickers[0])

        for label, ticker in tickers.items():
            if ticker in yf_df.columns:
                yf_df.rename(columns={ticker: label}, inplace=True)
        
        combined = yf_df
        if not fred_data.empty:
            fred_data = fred_data.rename(columns={'value': 'Dollar'})
            if yf_df.index.tz is not None: yf_df.index = yf_df.index.tz_localize(None)
            if fred_data.index.tz is not None: fred_data.index = fred_data.index.tz_localize(None)
            combined = pd.concat([yf_df, fred_data], axis=1).ffill().dropna()
            
        if combined.empty or 'Base' not in combined.columns: return pd.Series()
        
        corrs = combined.pct_change().rolling(20).corr(combined['Base'].pct_change()).iloc[-1]
        return corrs.drop('Base', errors='ignore') 
    except Exception as e: 
        return pd.Series()

@st.cache_data(ttl=300)
def run_strategy_backtest(ticker):
    try:
        df = safe_yf_download(ticker, period="2y", interval="1d")
        if df.empty: return None
        df['Returns'] = df['Close'].pct_change()
        df['Range'] = df['High'] - df['Low']
        df['TR'] = pd.concat([df['Range'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
        df['Log_TR'] = np.log(df['TR'] / df['Close'])
        df['Vol_Forecast'] = df['Log_TR'].ewm(span=10).mean()
        df['Vol_Baseline'] = df['Log_TR'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['Signal'] = np.where((df['Vol_Forecast'] > df['Vol_Baseline']) & (df['Close'] > df['SMA_50']), 1, 0)
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        df['Cum_BnH'] = (1 + df['Returns']).cumprod()
        df['Cum_Strat'] = (1 + df['Strategy_Returns']).cumprod()
        total_return = df['Cum_Strat'].iloc[-1] - 1
        sharpe = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252) if df['Strategy_Returns'].std() != 0 else 0
        current_signal = "LONG" if df['Signal'].iloc[-1] == 1 else "CASH/NEUTRAL"
        return {"df": df, "signal": current_signal, "return": total_return, "sharpe": sharpe, "equity_curve": df['Cum_Strat']}
    except: return None

def calculate_vwap_bands(df):
    if df.empty: return df
    df = df.copy()
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['TP'] * df['Volume']
    df['Date'] = df.index.date
    df['Cum_VP'] = df.groupby('Date')['VP'].cumsum()
    df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
    df['VWAP'] = df['Cum_VP'] / df['Cum_Vol']
    df['Sq_Dist'] = df['Volume'] * (df['TP'] - df['VWAP'])**2
    df['Cum_Sq_Dist'] = df.groupby('Date')['Sq_Dist'].cumsum()
    df['Std_Dev'] = np.sqrt(df['Cum_Sq_Dist'] / df['Cum_Vol'])
    df['Upper_Band_1'] = df['VWAP'] + df['Std_Dev']
    df['Lower_Band_1'] = df['VWAP'] - df['Std_Dev']
    return df

@st.cache_data(ttl=300) 
def get_relative_strength(asset_ticker, benchmark_ticker="SPY"):
    try:
        asset = safe_yf_download(asset_ticker, period="5d", interval="15m")
        bench = safe_yf_download(benchmark_ticker, period="5d", interval="15m")
        if asset.empty or bench.empty: return pd.DataFrame()
        df = pd.DataFrame(index=asset.index)
        df['Asset_Close'] = asset['Close']
        df['Bench_Close'] = bench['Close']
        df = df.dropna()
        current_date = df.index[-1].date()
        session_data = df[df.index.date == current_date].copy()
        if session_data.empty: return pd.DataFrame()
        session_data['Asset_Pct'] = (session_data['Asset_Close'] / session_data['Asset_Close'].iloc[0]) - 1
        session_data['Bench_Pct'] = (session_data['Bench_Close'] / session_data['Bench_Close'].iloc[0]) - 1
        session_data['RS_Score'] = session_data['Asset_Pct'] - session_data['Bench_Pct']
        return session_data
    except: return pd.DataFrame()

def get_key_levels(daily_df):
    if daily_df.empty: return {}
    try: last_complete_day = daily_df.iloc[-2]
    except: return {}
    high, low, close = last_complete_day['High'], last_complete_day['Low'], last_complete_day['Close']
    pivot = (high + low + close) / 3
    return {
        "PDH": high, "PDL": low, "PDC": close, "Pivot": pivot, 
        "R1": (2 * pivot) - low, "S1": (2 * pivot) - high
    }

def calculate_correlation_matrix(returns_df):
    """Computes the correlation matrix for the given returns DataFrame."""
    if returns_df.empty: return pd.DataFrame()
    return returns_df.corr()

def calculate_recession_probability(spread_10y3m):
    """
    NY Fed simplified model: Probability = Norm.CDF(-0.533 + (-0.633 * Spread))
    This is a simplified linear approximation of the probit model.
    """
    if spread_10y3m is None: return 0.0
    # Probit-like approximation
    # Note: Traditional spread is 10Y - 3M. If spread is negative, prob increases.
    z_score = -0.533 - 0.633 * (spread_10y3m) 
    prob = norm.cdf(z_score) * 100
    return prob

def get_yield_curve_regime(spread_10y2y, spread_10y3m):
    """
    Determines the Yield Curve regime based on spread levels and trends.
    """
    if spread_10y2y is None or spread_10y3m is None: return "Unknown", "neutral"
    
    # Logic for Clock
    if spread_10y2y < 0 and spread_10y3m < 0:
        return "INVERTED (High Recession Risk)", "bearish"
    elif spread_10y2y < 0 or spread_10y3m < 0:
        return "FLATTENING (Early Warning)", "neutral"
    elif spread_10y2y > 0 and spread_10y2y < 0.5:
        return "NORMAL (Low Growth)", "neutral"
    elif spread_10y2y >= 0.5:
        return "STEEPENING (Growth/Inflation)", "bullish"
    
    return "Neutral", "neutral"

def get_regime_impact(regime_name, ticker):
    """
    Returns an institutional explanation of how the current curve regime 
    typically impacts a specific asset class.
    """
    asset_class = "Generic"
    if "GSPC" in ticker or "IXIC" in ticker: asset_class = "Equity"
    elif "-USD" in ticker: asset_class = "Crypto"
    elif "GC=F" in ticker: asset_class = "Gold"
    elif "EURUSD" in ticker: asset_class = "Forex"

    impacts = {
        "INVERTED": {
            "Equity": "High recession risk signals margin compression. Institutional capital typically rotates out of Growth and into Defensive sectors.",
            "Crypto": "Risk-off regime. Tightening liquidity and rising recession fears are historically negative for speculative 'digital gold' assets.",
            "Gold": "Bullish. Yield curve inversion often precedes falling real rates and economic uncertainty, driving flight-to-safety demand.",
            "Forex": "Supportive for USD due to volatility and safe-haven flows, but can indicate late-cycle peak for the Greenback.",
            "Generic": "Recessionary signal. Liquidity is constrained, favoring defensive positioning and high cash levels."
        },
        "FLATTENING": {
            "Equity": "Transitionary. Late-cycle dynamic. Market starts pricing in peak earnings and slower future growth. Range-bound behavior common.",
            "Crypto": "Neutral to Bearish. Speculative momentum typically stalls as the 'easy money' phase of the cycle concludes.",
            "Gold": "Neutral. Market waits for clear directional signals from real rates and inflation expectations.",
            "Forex": "USD strength often peaks here as other central banks begin catching up to higher rates.",
            "Generic": "Yield curve flattening suggests a slowing economy. Growth expectations are lowering relative to current rates."
        },
        "NORMAL": {
            "Equity": "Healthy expansion. Broad-based growth is supported by stable interest rate expectations and credit availability.",
            "Crypto": "Bullish. Stable growth and moderate inflation provide the ideal environment for risk-on adoption and upward price action.",
            "Gold": "Stable to Bearish. Opportunity cost of holding non-yielding assets increases as the economy grows without excessive inflation.",
            "Forex": "Reflects standard economic growth metrics. Capital follows growth differentials and productivity.",
            "Generic": "The benchmark state. Economy is in a sustainable growth phase with low systemic risk."
        },
        "STEEPENING": {
            "Equity": "Bullish if growth-driven; Negative if inflation-driven. Watch if higher long-term rates start hurting valuations.",
            "Crypto": "Positive. Steepening curves often correlate with reflation trades. High correlation with rising liquidity and inflation sentiment.",
            "Gold": "Strongly Bullish. In a Bear Steepener (inflation-led), Gold acts as the primary hedge against currency debasement.",
            "Forex": "Volatile. Reflects deep shifts in future rate expectations. Typically leads to significant trend reversals.",
            "Generic": "Macro re-acceleration. Reflects expectations of future growth or rising inflation/risk premiums."
        }
    }
    
    # Normalize regime name key
    regime_key = "NORMAL"
    if "INVERTED" in regime_name.upper(): regime_key = "INVERTED"
    elif "FLATTENING" in regime_name.upper(): regime_key = "FLATTENING"
    elif "STEEPENING" in regime_name.upper(): regime_key = "STEEPENING"
    
    return impacts.get(regime_key, {}).get(asset_class, impacts.get(regime_key, {}).get("Generic", "No specific context available."))
