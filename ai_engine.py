import streamlit as st
import pandas as pd
import google.generativeai as genai
import config

# --- SAFE IMPORT SYSTEM ---
try:
    from textblob import TextBlob
    HAS_NLP = True
except ImportError:
    HAS_NLP = False

def get_gemini_model(api_key, model_name=None):
    """
    Helper to get the model using the constant from config.py or an override.
    """
    try:
        genai.configure(api_key=api_key)
        m_name = model_name if model_name else config.GEMINI_MODEL_NAME
        return genai.GenerativeModel(m_name)
    except Exception as e:
        # Cascade Fallback System
        for fallback in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
            try:
                return genai.GenerativeModel(fallback)
            except:
                continue
        return None

def get_technical_narrative(ticker, price, daily_pct, regime, ml_signal, gex_data, cot_data, levels, macro_data, api_key, model_name=None):
    if not api_key: return "AI Analyst unavailable (No Key)."
    if 'gemini_calls' in st.session_state: st.session_state['gemini_calls'] += 1
    
    gex_text = "N/A"
    if gex_data is not None:
        total_gex = gex_data['gex'].sum()
        gex_text = f"Net Gamma: ${total_gex/1_000_000:.1f}M ({'Long/Sticky' if total_gex>0 else 'Short/Volatile'})"
    lvl_text = "N/A"
    if levels:
        lvl_text = f"Pivot: {levels['Pivot']:.2f}, R1: {levels['R1']:.2f}, S1: {levels['S1']:.2f}"
    
    macro_str = "N/A"
    if macro_data:
        macro_str = f"YieldCurve: {macro_data.get('yield_curve', 'N/A')}, Inflation(CPI): {macro_data.get('cpi', 'N/A')}%, Rates: {macro_data.get('rates', 'N/A')}%, MacroRegime: {macro_data.get('regime', 'N/A')}, MacroPressureScore: {macro_data.get('macro_pressure_score', 'N/A')}/100 ({macro_data.get('macro_pressure_status', 'N/A')})"
    
    cot_str = "N/A"
    if cot_data and 'sentiment' in cot_data:
        cot_str = cot_data['sentiment']

    prompt = f"""
    You are a Senior Portfolio Manager. Analyze data for {ticker} and write a 3-bullet executive summary.
    DATA: Price: {price:,.2f} ({daily_pct:.2f}%), Regime: {regime['regime'] if regime else 'Unknown'}, 
    ML: {ml_signal}, GEX: {gex_text}, COT: {cot_str}, Levels: {lvl_text},
    MOMENTUM DETERIORATION: {macro_data.get('momentum_status', 'Stable')} (Score: {macro_data.get('momentum_score', 0)})
    MACRO CONTEXT: {macro_str}
    TASK:
    1. PRIORITIZE PRICE ACTION: Use Market Structure and Levels as the primary signal.
    2. CONTEXTUALIZE POSITIONING: Treat COT and GEX as secondary confirmation or potential headwinds, NOT as the primary driver.
    3. Synthesize Technicals + Macro.
    4. Identify key trigger level.
    5. Final Execution bias ("Buy Dips", "Fade", etc).
    JD Capital Institutional style. Keep it concise.
    DO NOT use markdown symbols like ** or ##. Use plain text.
    """
    try:
        model = get_gemini_model(api_key, model_name)
        if not model: return "Error: No valid models found. Check API Key."
            
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e): return "⚠️ QUOTA EXCEEDED (Free Tier: 5 RPM). Please wait 60s."
        return f"AI Analyst unavailable: {str(e)}"

def generate_deep_dive_thesis(ticker, price, change, regime, ml_signal, gex_data, cot_data, levels, news_summary, macro_data, api_key, model_name=None):
    if not api_key: return "Deep Dive unavailable (No Key)."
    if 'gemini_calls' in st.session_state: st.session_state['gemini_calls'] += 1
    
    model = get_gemini_model(api_key, model_name)
    gex_text = "N/A"
    if gex_data is not None:
        total_gex = gex_data['gex'].sum()
        gex_text = f"Net Gamma: ${total_gex/1_000_000:.1f}M"
    macro_str = "N/A"
    if macro_data:
        macro_str = f"YieldCurve: {macro_data.get('yield_curve', 'N/A')}, CPI: {macro_data.get('cpi', 'N/A')}%, Rates: {macro_data.get('rates', 'N/A')}%, Regime: {macro_data.get('regime', 'N/A')}, MacroPressure: {macro_data.get('macro_pressure_score', 'N/A')}/100"
    
    cot_str = "N/A"
    if cot_data and 'sentiment' in cot_data:
        cot_str = cot_data['sentiment']

    prompt = f"""
    Write a detailed Investment Thesis for {ticker}.
    DATA: Price: {price:,.2f} ({change:.2f}%), Regime: {regime['regime'] if regime else 'Unknown'}, 
    ML: {ml_signal}, GEX: {gex_text}, COT: {cot_str}
    MACRO: {macro_str}
    NEWS: {news_summary}
    OUTPUT FORMAT:
    Use plain text. DO NOT use markdown characters like "##", "###", or "**".
    1. PRICE ACTION & MARKET STRUCTURE (Primary Focus)
    2. INSTITUTIONAL POSITIONING & FLOW (Confirmation/Headwinds)
    3. THE MACRO CROSSROADS
    4. CORE THESIS & EXECUTION BIAS
    5. KEY LEVELS & INVALIDATION
    """
    try:
        model = get_gemini_model(api_key)
        if not model: return "Error: No valid models found."
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e): return "⚠️ QUOTA EXCEEDED (Free Tier: 5 RPM). Please wait 60s."
        return f"Thesis Generation Failed: {str(e)}"

def calculate_news_sentiment(news_items):
    if not HAS_NLP or not news_items: return pd.DataFrame()
    
    scores = []
    for news in news_items:
        try:
            # FIX: Use title + description instead of repeating title (which doubles polarity artificially)
            text_to_analyze = f"{news['title']} {news.get('description', '')}"
            blob = TextBlob(text_to_analyze) 
            score = blob.sentiment.polarity
            scores.append({
                "title": news['title'],
                "score": score,
                "time": news['time']
            })
        except: continue
        
    df = pd.DataFrame(scores)
    if df.empty: return pd.DataFrame()
    
    df = df.iloc[::-1].reset_index(drop=True) 
    df['cumulative'] = df['score'].cumsum()
    return df
