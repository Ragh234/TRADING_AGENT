"""
MarketMind AI — Shared State Definition (India Focused)
"""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

class AgentSignal(TypedDict):
    agent: str
    signal: str          # BULLISH | BEARISH | NEUTRAL
    confidence: float    # 0.0 – 1.0
    summary: str
    raw_data: dict

class MarketState(TypedDict):
    ticker: str
    asset_type: str                                        # "stock" or "crypto"
    agent_signals: Annotated[list[AgentSignal], operator.add]
    final_verdict: Optional[str]
    final_confidence: Optional[float]
    final_reasoning: Optional[str]
