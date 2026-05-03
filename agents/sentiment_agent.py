"""
MarketMind AI — Sentiment Agent (India Edition)
Fetches recent headlines via GNews API (country=in) for Indian context,
and asks the LLM to classify overall sentiment.
"""
from __future__ import annotations

import os
import urllib.parse
from datetime import datetime, timedelta
from typing import Any

import requests

from agents import build_signal, format_india_ticker, get_llm, safe_parse_json
from state import MarketState

def _fetch_gnews(query: str, days: int = 3, max_articles: int = 15) -> list[str]:
    """Return a list of recent headline strings from GNews API for India."""
    api_key = os.getenv("GNEWS_API_KEY", "")
    if not api_key:
        return ["[GNEWS_API_KEY not set — skipping headlines]"]

    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    encoded_query = urllib.parse.quote(query)
    url = f"https://gnews.io/api/v4/search?q={encoded_query}&country=in&lang=en&max={max_articles}&from={from_date}&sortby=publishedAt&apikey={api_key}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [a.get("title", "") for a in articles if a.get("title")]
    except Exception:
        return ["[Failed to fetch news from GNews]"]

def sentiment_agent(state: MarketState) -> dict[str, Any]:
    ticker: str = state["ticker"]
    asset_type: str = state["asset_type"]

    # Clean the ticker for search (remove .NS for better news search)
    search_term = ticker.replace(".NS", "").replace(".BO", "")
    query = search_term if asset_type == "crypto" else f"{search_term} India stock market"

    headlines = _fetch_gnews(query)
    headlines_text = "\n".join(f"- {h}" for h in headlines[:15])

    raw_data: dict[str, Any] = {
        "headline_count": len(headlines),
        "headlines_sample": headlines[:5],
    }

    prompt = f"""You are a financial sentiment analyst focusing on the Indian market. Analyze the following recent news headlines about {search_term}:

{headlines_text}

Consider Indian market dynamics, emerging market trends, and RBI policies if mentioned.
Based on the overall sentiment of these headlines, respond with ONLY a JSON object:
{{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "short explanation", "positive_count": int, "negative_count": int, "neutral_count": int}}"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        parsed = safe_parse_json(response.content)

        raw_data["sentiment_counts"] = {
            "positive": parsed.get("positive_count", 0),
            "negative": parsed.get("negative_count", 0),
            "neutral": parsed.get("neutral_count", 0),
        }

        return {
            "agent_signals": [
                build_signal(
                    agent_name="Sentiment Agent",
                    signal=parsed.get("signal", "NEUTRAL"),
                    confidence=float(parsed.get("confidence", 0.5)),
                    summary=parsed.get("summary", "Sentiment analysis complete."),
                    raw_data=raw_data,
                )
            ]
        }

    except Exception as e:
        return {
            "agent_signals": [
                build_signal(
                    agent_name="Sentiment Agent",
                    signal="NEUTRAL",
                    confidence=0.0,
                    summary=f"Error during sentiment analysis: {e}",
                    raw_data={"error": str(e)},
                )
            ]
        }
