import streamlit as st
import requests
import time
from logger import log_api_status
import pandas as pd

# Rate limiting settings for CoinGecko Demo API
# 30 calls per minute = 1 call every 2 seconds
COINGECKO_RATE_LIMIT_DELAY = 2.05
_last_call_time = 0.0

def _wait_for_rate_limit():
    global _last_call_time
    current_time = time.time()
    elapsed = current_time - _last_call_time
    if elapsed < COINGECKO_RATE_LIMIT_DELAY:
        time.sleep(COINGECKO_RATE_LIMIT_DELAY - elapsed)
    _last_call_time = time.time()

BASE_URL = "https://api.coingecko.com/api/v3"

def _make_request(endpoint, params=None, headers=None, api_key=None):
    if headers is None:
        headers = {}
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
        
    _wait_for_rate_limit()
    
    url = f"{BASE_URL}{endpoint}"
    try:
        if 'coingecko_calls' in st.session_state: 
            st.session_state['coingecko_calls'] += 1
            
        response = requests.get(url, params=params, headers=headers)
        
        # Log the status
        log_api_status(endpoint, response.status_code, response.text[:100] if response.status_code != 200 else "")
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        log_api_status(endpoint, 500, str(e))
        return None

# Endpoints mapping to Streamlit cache to prevent redundant hits.
# 10,000 credits/mo is ~333 per day, so caching for 5-10 minutes is crucial.

@st.cache_data(ttl=300)
def get_cg_simple_price(ids, vs_currencies="usd", api_key=None):
    params = {"ids": ids, "vs_currencies": vs_currencies, "include_market_cap": "true", "include_24hr_vol": "true"}
    return _make_request("/simple/price", params=params, api_key=api_key)

@st.cache_data(ttl=600)
def get_cg_coin_data(cg_id, api_key=None):
    params = {
        "localization": "false", "tickers": "false", "market_data": "true",
        "community_data": "true", "developer_data": "true", "sparkline": "false"
    }
    return _make_request(f"/coins/{cg_id}", params=params, api_key=api_key)

@st.cache_data(ttl=600)
def get_cg_markets(vs_currency="usd", api_key=None):
    params = {"vs_currency": vs_currency, "order": "market_cap_desc", "per_page": 100, "page": 1}
    return _make_request("/coins/markets", params=params, api_key=api_key)

@st.cache_data(ttl=3600) # cache for an hour
def get_cg_historical_data(cg_id, days="365", api_key=None):
    # market_chart for daily data
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    return _make_request(f"/coins/{cg_id}/market_chart", params=params, api_key=api_key)

@st.cache_data(ttl=1800)
def get_cg_ohlc(cg_id, days="30", api_key=None):
    params = {"vs_currency": "usd", "days": days}
    return _make_request(f"/coins/{cg_id}/ohlc", params=params, api_key=api_key)

@st.cache_data(ttl=900)
def get_cg_trending(api_key=None):
    return _make_request("/search/trending", api_key=api_key)

@st.cache_data(ttl=3600)
def get_cg_categories(api_key=None):
    return _make_request("/coins/categories", api_key=api_key)

@st.cache_data(ttl=3600)
def get_cg_exchanges(api_key=None):
    return _make_request("/exchanges", api_key=api_key)

@st.cache_data(ttl=900)
def get_cg_global(api_key=None):
    return _make_request("/global", api_key=api_key)
