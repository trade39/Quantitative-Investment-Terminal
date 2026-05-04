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
    You are a Senior Risk Manager and Quantitative Strategist. Your primary mission is NOT to explain the past, but to QUANTIFY UNCERTAINTY and define ACTIONABLE TRIGGERS for execution.
    SYSTEM STATE: Adaptive Parameters Active (Dynamic tuning based on uncertainty density).
    DATA: Price: {price:,.2f} ({daily_pct:.2f}%), Regime: {regime_val} (Stability: {macro_data.get('regime_stability', 'N/A')}), MarketStructure: {ms_trend},
    ML: {ml_signal} (Kelly Sizing: {macro_data.get('kelly_size', 'N/A')}), GEX: {gex_text}, COT: {cot_str}, Levels: {lvl_text},
    UNCERTAINTY MAP (Market-Implied): 1-Day: {macro_data.get('expected_move_1d', 'N/A')}, 1-Week: {macro_data.get('expected_move_1w', 'N/A')}
    RISK DENSITY: VaR(95%): {macro_data.get('var_95', 'N/A')}, CVaR(Tail Risk): {macro_data.get('cvar_95', 'N/A')}, MOMENTUM DETERIORATION: {macro_data.get('momentum_status', 'Stable')} (Score: {macro_data.get('momentum_score', 0)})
    MACRO CONTEXT: {macro_str}
    TASK:
    1. UNCERTAINTY QUANTIFICATION: Use the Implied Expected Moves to define the current 'Vol-Space'. Is price compressed or overextended relative to the 1-week uncertainty band?
    2. TAIL RISK BUDGETING: Analyze CVaR to define the absolute 'Panic Floor'. How should this impact position sizing relative to the Kelly Criterion?
    3. REGIME ADHERENCE: If Regime is 'COMPRESSION' or MarketStructure is 'RANGE', define the exact 'No-Trade Zone'.
    4. TRIGGER-BASED EXECUTION: Define the EXACT price/volume conditions required to resolve current uncertainty into a trade. No bias without a trigger.
    5. PRIORITIZE PRICE ACTION: Use Market Structure as the primary ground truth.
    6. SEARCH GROUNDING: Use your Google Search tool to identify real-time catalysts that could shift the uncertainty map for {ticker} in the next 7 days.
    JD Capital Institutional style. Focus on the 'Why now' and the 'How to act'.
    DO NOT use markdown symbols like ** or ##. Use plain text.
    CRITICAL: DO NOT use the '$' symbol for currency amounts.
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
    You are a Senior Risk Manager at JD Capital writing a multi-page Uncertainty Quantification & Execution Brief for {ticker}.
    MISSION: Quantify current market uncertainty and define the actionable path forward.
    SYSTEM ARCHITECTURE: Adaptive Quantitative System (Dynamic parameter tuning active).
    DATA: Price: {price:,.2f} ({change:.2f}%), Regime: {regime_val} (Stability: {macro_data.get('regime_stability', 'N/A')}), MarketStructure: {ms_trend},
    ML: {ml_signal} (Kelly Optimal Size: {macro_data.get('kelly_size', 'N/A')}), GEX: {gex_text}, COT: {cot_str}
    UNCERTAINTY BANDS: 1-Day Move: {macro_data.get('expected_move_1d', 'N/A')}, 1-Week Move: {macro_data.get('expected_move_1w', 'N/A')}
    RISK DENSITY: VaR(95%): {macro_data.get('var_95', 'N/A')}, CVaR(Tail Risk): {macro_data.get('cvar_95', 'N/A')}
    MACRO: {macro_str}
    NEWS: {news_summary}

    OUTPUT FORMAT:
    Use plain text only. DO NOT use markdown characters like "##", "###", or "**".
    CRITICAL: DO NOT use the '$' symbol for currency.

    1. UNCERTAINTY MAPPING & STRUCTURE
    Analyze the price action within the 1-week uncertainty bands. Is the market compressed or expanding? Define the 'Structural Walls' (exact boundaries) and the stability of the current regime.

    2. PROBABILISTIC RISK & TAIL DEFENSE
    Quantify the 'Left Tail' risk using VaR and CVaR. How does the current 'Risk Density' impact position sizing? Explain the optimal Kelly allocation relative to the volatility-adjusted stop levels.

    3. LIQUIDITY & POSITIONING FLOWS
    Analyze GEX and COT positioning. How do these flows shift the uncertainty map? Identify 'Gamma Walls' and 'Positioning Extremes' that could trigger a cascade.

    4. REAL-TIME CATALYSTS (SEARCH GROUNDING)
    Use your Google Search tool to identify catalysts (macro, geopolitical, central bank) from the last 7 days that threaten to resolve or amplify current uncertainty.

    5. EXECUTION TRIGGERS & INVALIDATION
    Define the EXACT triggers for entry. State the 'Acceptance Criteria' (Price + Volume + Close). Define the 'Invalidation Point' where the thesis must be abandoned. No narrative, only execution logic.
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
