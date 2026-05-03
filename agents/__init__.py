"""
MarketMind AI — agents package (India Focused)
Provides a shared Groq LLM client and JSON parsing utilities.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Return a single, cached Groq LLM instance."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in your .env file.")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=api_key,
        max_retries=3,
    )

def safe_parse_json(text: str) -> dict[str, Any]:
    """Robustly extract and parse JSON from LLM output."""
    if not text:
        return {}

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}

def build_signal(
    agent_name: str,
    signal: str,
    confidence: float,
    summary: str,
    raw_data: dict | None = None,
) -> dict[str, Any]:
    """Construct a normalised agent signal dict."""
    valid_signals = {"BULLISH", "BEARISH", "NEUTRAL"}
    signal_upper = signal.upper().strip()
    if signal_upper not in valid_signals:
        signal_upper = "NEUTRAL"
    return {
        "agent": agent_name,
        "signal": signal_upper,
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
        "summary": summary,
        "raw_data": raw_data or {},
    }

def format_india_ticker(ticker: str) -> str:
    """Helper to ensure Indian stocks have the .NS extension for Yahoo Finance."""
    t = ticker.upper().strip()
    if not t.endswith(".NS") and not t.endswith(".BO"):
        return f"{t}.NS"
    return t
