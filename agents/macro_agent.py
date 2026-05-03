"""
MarketMind AI — Macro Agent (India Edition)
Uses NIFTY 50 (^NSEI) and Bank Nifty (^NSEBANK) from yfinance, 
and provides a generic RBI Repo Rate and CPI estimate if live values are unavailable. 
"""
from __future__ import annotations

from typing import Any
import pandas as pd
import yfinance as yf

from agents import build_signal, get_llm, safe_parse_json
from state import MarketState

def _fetch_index_trend(index_ticker: str) -> dict[str, Any]:
    try:
        df = yf.download(index_ticker, period="30d", interval="1d", progress=False)
        if df.empty or len(df) < 7:
            return {"error": "Insufficient data"}
        
        close = df["Close"].squeeze()
        current = float(close.iloc[-1])
        pct_7d = round(float((current - close.iloc[-7]) / close.iloc[-7] * 100), 2)
        pct_30d = round(float((current - close.iloc[0]) / close.iloc[0] * 100), 2)
        
        return {
            "current_value": round(current, 2),
            "pct_change_7d": pct_7d,
            "pct_change_30d": pct_30d,
            "direction": "up" if pct_7d > 0 else "down"
        }
    except Exception as e:
        return {"error": str(e)}

def macro_agent(state: MarketState) -> dict[str, Any]:
    ticker: str = state["ticker"]

    # India-specific Macro Data
    nifty50 = _fetch_index_trend("^NSEI")
    bank_nifty = _fetch_index_trend("^NSEBANK")
    
    # Mocking live Repo rate / CPI for the Indian market hackathon scope 
    # (Because free real-time RBI API is not reliably available without web scraping)
    rbi_repo_rate = 6.50
    india_cpi = 5.09

    raw_data: dict[str, Any] = {
        "nifty_50": nifty50,
        "bank_nifty": bank_nifty,
        "rbi_repo_rate": rbi_repo_rate,
        "india_cpi": india_cpi
    }

    prompt = f"""You are a top macroeconomic analyst for the Indian financial markets.
Evaluate the following India macro signals and gauge how they will impact {ticker}.

- NIFTY 50 Trend: {nifty50}
- BANK NIFTY Trend: {bank_nifty}
- Current RBI Repo Rate: {rbi_repo_rate}%
- India retail inflation (CPI): ~{india_cpi}%

Consider Indian market conditions, the impact of the RBI policy on liquidity, and emerging market volatility.
Respond with ONLY a JSON object:
{{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "short explanation"}}"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        parsed = safe_parse_json(response.content)

        return {
            "agent_signals": [
                build_signal(
                    agent_name="Macro Agent",
                    signal=parsed.get("signal", "NEUTRAL"),
                    confidence=float(parsed.get("confidence", 0.5)),
                    summary=parsed.get("summary", "India Macro analysis complete."),
                    raw_data=raw_data,
                )
            ]
        }

    except Exception as e:
        return {
            "agent_signals": [
                build_signal(
                    agent_name="Macro Agent",
                    signal="NEUTRAL",
                    confidence=0.0,
                    summary=f"Error during macro analysis: {e}",
                    raw_data={"error": str(e)},
                )
            ]
        }
