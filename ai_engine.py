import streamlit as st
import pandas as pd
import os
import config

# --- GOOGLE GENAI SDK (NEW) ---
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# --- OLD GOOGLE GENERATIVEAI SDK (FALLBACK) ---
try:
    import google.generativeai as old_genai
    HAS_OLD_GENAI = True
except ImportError:
    HAS_OLD_GENAI = False

# --- NLP SAFE IMPORT ---
try:
    from textblob import TextBlob
    HAS_NLP = True
except ImportError:
    HAS_NLP = False

def get_genai_client(api_key):
    """Initializes the new google-genai client."""
    if not HAS_GENAI: return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"GenAI Client Init Error: {e}")
        return None

def get_technical_narrative(ticker, price, daily_pct, regime, ml_signal, gex_data, cot_data, levels, macro_data, api_key, model_name=None, use_grounding=False, ms_trend="N/A"):
    if not api_key: return "AI Analyst unavailable (No Key)."
    if 'gemini_calls' in st.session_state: st.session_state['gemini_calls'] += 1
    
    # Process Regime & Trend
    regime_val = regime['regime'] if regime else 'Unknown'
    
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
    You are a Senior Portfolio Manager. Analyze data for {ticker} and write a detailed, comprehensive executive summary.
    DATA: Price: {price:,.2f} ({daily_pct:.2f}%), Regime: {regime_val}, MarketStructure: {ms_trend},
    ML: {ml_signal}, GEX: {gex_text}, COT: {cot_str}, Levels: {lvl_text},
    MOMENTUM DETERIORATION: {macro_data.get('momentum_status', 'Stable')} (Score: {macro_data.get('momentum_score', 0)})
    MACRO CONTEXT: {macro_str}
    TASK:
    1. REGIME ADHERENCE: If Regime is 'COMPRESSION', 'RANGE-BOUND', or MarketStructure indicates 'RANGE', DO NOT label it a trend. State the explicit range boundaries.
    2. STRICT NEUTRALITY PRE-BREAKOUT: A 'Boundary Test' is NOT a directional bias. If the price is in the 'NO-TRADE CHOP ZONE', expressly forbid entering new positions until boundaries are tested.
    3. TRIGGER-BASED EXECUTION: Provide actionable triggers WITH acceptance criteria (e.g., "Wait for daily close > [High] with volume confirmation"). Do not front-run the breakout.
    4. PRIORITIZE PRICE ACTION: Use Market Structure and Levels as the primary signal.
    5. INTEGRATE SEARCH GROUNDING: You MUST use your Google Search tool to cover global macro conditions, key data releases, and actionable insights for {ticker}. Focus on:
       - Key economic data release breakdowns
       - Cross-asset positioning & flow insights
       - Central bank watch & rate path expectations
       - Recent deep-dive research notes
       Detail exactly how these live catalysts impact the execution plan. Know what's moving markets before you trade. Use live, real-time news from strictly within the last 7 days.
    JD Capital Institutional style. Elaborate deeply on the technical and macro interactions.
    DO NOT use markdown symbols like ** or ##. Use plain text.
    CRITICAL: DO NOT use the '$' symbol for currency amounts (e.g., write 4,668.79 USD or just 4,668.79 instead of $4,668.79) to avoid markdown math rendering issues.
    """

    if use_grounding and HAS_GENAI:
        return get_grounded_response(prompt, api_key, model_name)
    
    # Fallback to new SDK (no grounding) if available
    if HAS_GENAI:
        try:
            client = get_genai_client(api_key)
            m_name = model_name if model_name else config.GEMINI_MODEL_NAME
            response = client.models.generate_content(model=m_name, contents=prompt)
            return response.text
        except: pass

    # Ultimate fallback to old SDK
    if HAS_OLD_GENAI:
        try:
            m_name = model_name if model_name else config.GEMINI_MODEL_NAME
            old_genai.configure(api_key=api_key)
            model = old_genai.GenerativeModel(m_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI Analyst Error: {str(e)}"
    
    return "AI Analyst Error: No compatible SDK found."

def generate_deep_dive_thesis(ticker, price, change, regime, ml_signal, gex_data, cot_data, levels, news_summary, macro_data, api_key, model_name=None, use_grounding=False, ms_trend="N/A"):
    if not api_key: return "Deep Dive unavailable (No Key)."
    if 'gemini_calls' in st.session_state: st.session_state['gemini_calls'] += 1
    
    # Process Regime & Trend
    regime_val = regime['regime'] if regime else 'Unknown'
    
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
    You are a Senior Portfolio Manager at JD Capital writing a comprehensive, multi-page Investment Thesis for {ticker}.
    DATA: Price: {price:,.2f} ({change:.2f}%), Regime: {regime_val}, MarketStructure: {ms_trend},
    ML: {ml_signal}, GEX: {gex_text}, COT: {cot_str}
    MACRO: {macro_str}
    NEWS: {news_summary}

    LENGTH REQUIREMENT: Each section below MUST be at minimum 3 substantial paragraphs (150+ words each).
    Do NOT summarize. Be exhaustive, analytical, and institutional in tone.
    Write as if this is a formal multi-page brief being presented to a fund's investment committee.

    OUTPUT FORMAT:
    Use plain text only. DO NOT use markdown characters like "##", "###", or "**".
    CRITICAL: DO NOT use the '$' symbol for currency (write 4,668.79 USD or just 4,668.79 instead of $4,668.79) to avoid rendering issues.

    1. PRICE ACTION & MARKET STRUCTURE
    Distinguish Trend vs Range in exhaustive detail. If RANGE, define exact boundaries and what each boundary represents structurally. Identify whether current price is in a 'Boundary Test', 'NO-TRADE CHOP ZONE', or trending. Elaborate on the significance of the current price action within the broader market context.

    2. INSTITUTIONAL POSITIONING & FLOW
    Analyze in depth how options gamma exposure (GEX) and the COT report positioning affect potential volatility and directional bias. Explain smart money behavior, what the COT data implies about medium-term conviction, and how gamma dynamics could amplify or dampen any move.

    3. THE MACRO CROSSROADS & RECENT NEWS
    You MUST use your Google Search tool to find live, real-time news specifically about {ticker} from strictly within the last 7 days. You must cover global macro conditions, key data releases, and actionable insights. Focus on:
       - Key economic data release breakdowns
       - Cross-asset positioning & flow insights
       - Central bank watch & rate path expectations
       - Weekly deep-dive research notes
    Detail how each live catalyst (macro events, geopolitical, Fed policy, economic data) directly impacts the execution plan. Explain the interplay between all macro forces. Know what's moving markets before you trade.

    4. CORE THESIS & EXECUTION BIAS
    State the primary thesis in full. Provide explicit Trigger-Based Execution with exact price levels. State the Acceptance Criteria (daily close + volume confirmation). If in the Chop Zone, detail exactly what market behavior is required before initiating a position. Explain the risk-reward dynamic.

    5. KEY LEVELS & INVALIDATION
    Define all precise support, resistance, and pivot levels with context. State exact conditions that would invalidate the long thesis and the short thesis separately. Define what constitutes a false breakout vs a confirmed one.
    """

    if use_grounding and HAS_GENAI:
        return get_grounded_response(prompt, api_key, model_name, max_tokens=16384)

    # Fallback to new SDK (no grounding) if available
    if HAS_GENAI:
        try:
            client = get_genai_client(api_key)
            m_name = model_name if model_name else config.GEMINI_MODEL_NAME
            gen_config = types.GenerateContentConfig(
                max_output_tokens=16384,
                temperature=0.9
            )
            response = client.models.generate_content(model=m_name, contents=prompt, config=gen_config)
            return response.text
        except: pass

    # Ultimate fallback to old SDK
    if HAS_OLD_GENAI:
        try:
            m_name = model_name if model_name else config.GEMINI_MODEL_NAME
            old_genai.configure(api_key=api_key)
            model = old_genai.GenerativeModel(m_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Thesis Generation Failed: {str(e)}"
            
    return "Thesis Generation Failed: No compatible SDK found."

def get_grounded_response(prompt, api_key, model_name=None, max_tokens=16384):
    """Core function to handle Google Search Grounding using the new GenAI SDK."""
    if not HAS_GENAI: return "Search Grounding Error: google-genai SDK not installed."
    
    client = get_genai_client(api_key)
    if not client: return "Search Grounding Error: Client initialization failed."

    m_name = model_name if model_name else config.GEMINI_MODEL_NAME
    
    config_genai = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=1.0,
        max_output_tokens=max_tokens
    )

    try:
        response = client.models.generate_content(
            model=m_name,
            contents=prompt,
            config=config_genai
        )
        
        full_text = response.text
        
        # Add citations if available
        if response.candidates and response.candidates[0].grounding_metadata:
            metadata = response.candidates[0].grounding_metadata
            citations = "\n\n--- GROUNDING SOURCES ---\n"
            
            if hasattr(metadata, 'web_search_queries') and metadata.web_search_queries:
                citations += f"Search Queries: {', '.join(metadata.web_search_queries)}\n"

            if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                for i, chunk in enumerate(metadata.grounding_chunks):
                    if hasattr(chunk, 'web') and chunk.web:
                        citations += f"[{i+1}] {chunk.web.title}: {chunk.web.uri}\n"
                
                full_text += citations
                
        return full_text
    except Exception as e:
        return f"Grounding API Error: {str(e)}"

def calculate_news_sentiment(news_items):
    if not HAS_NLP or not news_items: return pd.DataFrame()
    
    scores = []
    for news in news_items:
        try:
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
