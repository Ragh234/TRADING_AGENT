"""
MarketMind AI — Price Agent (India Edition)
Fetches 90-day price data via yfinance (auto-appends .NS for stocks),
computes RSI / MACD / Bollinger Bands, and asks LLM for interpretation.
"""
from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from agents import build_signal, format_india_ticker, get_llm, safe_parse_json
from state import MarketState

def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2) if not rsi.empty else 50.0

def _compute_macd(series: pd.Series) -> dict[str, float]:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": round(float(macd_line.iloc[-1]), 4),
        "signal": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
    }

def _compute_bollinger(series: pd.Series, period: int = 20) -> dict[str, float]:
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    last_price = float(series.iloc[-1])
    return {
        "upper": round(float(upper.iloc[-1]), 4),
        "middle": round(float(sma.iloc[-1]), 4),
        "lower": round(float(lower.iloc[-1]), 4),
        "price": round(last_price, 4),
        "pct_b": round(
            (last_price - float(lower.iloc[-1]))
            / max(float(upper.iloc[-1]) - float(lower.iloc[-1]), 1e-9),
            4,
        ),
    }

def price_agent(state: MarketState) -> dict[str, Any]:
    ticker: str = state["ticker"]
    asset_type: str = state["asset_type"]

    if asset_type == "crypto":
        yf_ticker = f"{ticker}-USD"
    else:
        yf_ticker = format_india_ticker(ticker)

    try:
        df = yf.download(yf_ticker, period="90d", interval="1d", progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {yf_ticker}. Is the symbol correct?")

        close = df["Close"].squeeze()
        rsi = _compute_rsi(close)
        macd = _compute_macd(close)
        bollinger = _compute_bollinger(close)

        if len(close) >= 7:
            pct_7d = round(float((close.iloc[-1] - close.iloc[-7]) / close.iloc[-7] * 100), 2)
        else:
            pct_7d = 0.0

        raw_data = {
            "resolved_ticker": yf_ticker,
            "rsi": rsi,
            "macd": macd,
            "bollinger": bollinger,
            "pct_change_7d": pct_7d,
            "last_price": round(float(close.iloc[-1]), 4),
        }

        prompt = f"""You are a senior technical analyst in the Indian market. Analyze these indicators for {yf_ticker}:

- RSI (14): {rsi}
- MACD: line={macd['macd']}, signal={macd['signal']}, histogram={macd['histogram']}
- Bollinger Bands: upper={bollinger['upper']}, middle={bollinger['middle']}, lower={bollinger['lower']}, %B={bollinger['pct_b']}
- 7-day price change: {pct_7d}%
- Current price: {raw_data['last_price']}

Consider Indian market conditions and emerging market volatility.
Respond with ONLY a JSON object:
{{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "short explanation"}}"""

        llm = get_llm()
        response = llm.invoke(prompt)
        parsed = safe_parse_json(response.content)

        return {
            "agent_signals": [
                build_signal(
                    agent_name="Price Agent",
                    signal=parsed.get("signal", "NEUTRAL"),
                    confidence=float(parsed.get("confidence", 0.5)),
                    summary=parsed.get("summary", "Technical analysis complete."),
                    raw_data=raw_data,
                )
            ]
        }

    except Exception as e:
        return {
            "agent_signals": [
                build_signal(
                    agent_name="Price Agent",
                    signal="NEUTRAL",
                    confidence=0.0,
                    summary=f"Error fetching price data: {e}",
                    raw_data={"error": str(e)},
                )
            ]
        }
