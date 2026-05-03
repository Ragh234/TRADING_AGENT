"""
MarketMind AI — Synthesis Agent (India Edition)
Combines all agent signals with weighted reasoning towards Indian macro & sentiment conditions.
"""
from __future__ import annotations

from typing import Any

from agents import get_llm, safe_parse_json
from state import MarketState

def synthesis_agent(state: MarketState) -> dict[str, Any]:
    signals = state.get("agent_signals", [])

    if not signals:
        return {
            "final_verdict": "NEUTRAL",
            "final_confidence": 0.0,
            "final_reasoning": "No agent signals available for synthesis.",
        }

    signal_lines: list[str] = []
    for s in signals:
        signal_lines.append(
            f"- {s['agent']}: {s['signal']} (confidence {s['confidence']}) — {s['summary']}"
        )
    signals_text = "\n".join(signal_lines)

    bull = sum(1 for s in signals if s["signal"] == "BULLISH")
    bear = sum(1 for s in signals if s["signal"] == "BEARISH")
    neutral = sum(1 for s in signals if s["signal"] == "NEUTRAL")

    prompt = f"""You are the chief quantitative strategist for an Indian Institutional Investment fund (DII). 
You have received analysis from {len(signals)} specialized agents:

{signals_text}

Signal distribution: {bull} BULLISH, {bear} BEARISH, {neutral} NEUTRAL

Your task:
1. Weigh each agent's signal by its confidence level. Discard Neutral/0% confidence signals.
2. Consider conflicting signals carefully.
3. Give higher weight to the Macro Agent and Sentiment Agent reflecting Indian economic conditions.
4. Produce a final unified verdict.

Respond with ONLY a JSON object:
{{
  "verdict": "BULLISH" or "BEARISH" or "NEUTRAL",
  "confidence": 0.0 to 1.0,
  "reasoning": "detailed but concise 2-4 sentence explanation of your decision, mentioning key factors impacting the Indian market"
}}"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        parsed = safe_parse_json(response.content)

        verdict = parsed.get("verdict", "NEUTRAL").upper().strip()
        if verdict not in {"BULLISH", "BEARISH", "NEUTRAL"}:
            verdict = "NEUTRAL"

        confidence = float(parsed.get("confidence", 0.5))
        confidence = round(max(0.0, min(1.0, confidence)), 2)

        reasoning = parsed.get(
            "reasoning",
            "Synthesis complete. See individual agent signals for details.",
        )

        return {
            "final_verdict": verdict,
            "final_confidence": confidence,
            "final_reasoning": reasoning,
        }

    except Exception as e:
        if bull > bear:
            fallback_verdict = "BULLISH"
        elif bear > bull:
            fallback_verdict = "BEARISH"
        else:
            fallback_verdict = "NEUTRAL"

        avg_confidence = round(
            sum(s["confidence"] for s in signals) / len(signals), 2
        )

        return {
            "final_verdict": fallback_verdict,
            "final_confidence": avg_confidence,
            "final_reasoning": f"Fallback vote: {bull}B/{bear}Be/{neutral}N. LLM error: {e}",
        }
