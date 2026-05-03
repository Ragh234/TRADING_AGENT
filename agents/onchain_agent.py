"""
MarketMind AI — On-Chain / Market Microstructure Agent (India Edition)
- Crypto: fetches CoinGecko data (market cap rank, volume, price changes)
- Stocks: uses yfinance (volume ratio, beta, short ratio) formatted for India NSE
"""
from __future__ import annotations

from typing import Any

import yfinance as yf
from pycoingecko import CoinGeckoAPI

from agents import build_signal, format_india_ticker, get_llm, safe_parse_json
from state import MarketState

_CRYPTO_ID_MAP: dict[str, str] = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "DOGE": "dogecoin",
    "ADA": "cardano", "XRP": "ripple", "DOT": "polkadot", "AVAX": "avalanche-2",
    "MATIC": "matic-network", "LINK": "chainlink", "BNB": "binancecoin",
    "LTC": "litecoin", "SHIB": "shiba-inu", "UNI": "uniswap", "ATOM": "cosmos",
}

def _fetch_crypto_data(ticker: str) -> dict[str, Any]:
    cg = CoinGeckoAPI()
    coin_id = _CRYPTO_ID_MAP.get(ticker.upper(), ticker.lower())
    try:
        data = cg.get_coin_by_id(coin_id, localization=False, tickers=False, community_data=False, developer_data=False)
        market = data.get("market_data", {})
        return {
            "source": "coingecko",
            "market_cap_rank": data.get("market_cap_rank"),
            "total_volume_usd": market.get("total_volume", {}).get("usd"),
            "market_cap_usd": market.get("market_cap", {}).get("usd"),
            "price_change_24h_pct": market.get("price_change_percentage_24h"),
            "price_change_7d_pct": market.get("price_change_percentage_7d"),
        }
    except Exception as e:
        return {"source": "coingecko", "error": str(e)}

def _fetch_stock_data(ticker: str) -> dict[str, Any]:
    yf_ticker = format_india_ticker(ticker)
    try:
        info = yf.Ticker(yf_ticker).info
        hist = yf.download(yf_ticker, period="30d", interval="1d", progress=False)
        if hist.empty:
            return {"source": "yfinance", "error": "No data available"}
        
        avg_vol = float(hist["Volume"].mean())
        last_vol = float(hist["Volume"].iloc[-1])
        vol_ratio = round(last_vol / max(avg_vol, 1), 2)

        return {
            "source": "yfinance", "resolved_ticker": yf_ticker,
            "volume_ratio": vol_ratio, "avg_volume_30d": round(avg_vol),
            "last_volume": round(last_vol), "beta": info.get("beta"),
            "market_cap": info.get("marketCap"), "pe_ratio": info.get("trailingPE"),
        }
    except Exception as e:
        return {"source": "yfinance", "error": str(e)}

def onchain_agent(state: MarketState) -> dict[str, Any]:
    ticker: str = state["ticker"]
    asset_type: str = state["asset_type"]

    raw_data = _fetch_crypto_data(ticker) if asset_type == "crypto" else _fetch_stock_data(ticker)
    data_summary = "\n".join(f"  {k}: {v}" for k, v in raw_data.items() if k != "source")
    label = "on-chain / crypto" if asset_type == "crypto" else "Indian market microstructure"

    prompt = f"""You are a {label} analyst. Analyze the following data for {ticker}:

{data_summary}

Respond with ONLY a JSON object:
{{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "short explanation"}}"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        parsed = safe_parse_json(response.content)

        return {
            "agent_signals": [
                build_signal(
                    agent_name="On-Chain Agent",
                    signal=parsed.get("signal", "NEUTRAL"),
                    confidence=float(parsed.get("confidence", 0.5)),
                    summary=parsed.get("summary", "Microstructure analysis complete."),
                    raw_data=raw_data,
                )
            ]
        }

    except Exception as e:
        return {
            "agent_signals": [
                build_signal(
                    agent_name="On-Chain Agent",
                    signal="NEUTRAL",
                    confidence=0.0,
                    summary=f"Error during analysis: {e}",
                    raw_data={"error": str(e)},
                )
            ]
        }
