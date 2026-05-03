"""
MarketMind AI — Risk Agent (India Edition)
Computes volatility, max drawdown, and beta against NIFTY 50 (^NSEI),
and integrates RBI Repo Rate awareness.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import yfinance as yf

from agents import build_signal, format_india_ticker, get_llm, safe_parse_json
from state import MarketState

def _compute_risk_metrics(ticker: str, asset_type: str) -> dict[str, Any]:
    if asset_type == "crypto":
        yf_ticker = f"{ticker}-USD"
        benchmark = "BTC-USD"
    else:
        yf_ticker = format_india_ticker(ticker)
        benchmark = "^NSEI"  # NIFTY 50 as benchmark

    try:
        df = yf.download(yf_ticker, period="90d", interval="1d", progress=False)
        bench = yf.download(benchmark, period="90d", interval="1d", progress=False)

        if df.empty:
            return {"error": "No price data available"}

        close = df["Close"].squeeze()
        returns = close.pct_change().dropna()

        volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

        beta = None
        if not bench.empty:
            bench_close = bench["Close"].squeeze()
            bench_returns = bench_close.pct_change().dropna()
            min_len = min(len(returns), len(bench_returns))
            if min_len > 5:
                r = returns.iloc[-min_len:].values
                b = bench_returns.iloc[-min_len:].values
                cov = np.cov(r, b)
                if cov[1, 1] != 0:
                    beta = round(float(cov[0, 1] / cov[1, 1]), 2)

        if volatility > 0.6 or max_drawdown < -0.25:
            risk_level = "HIGH"
        elif volatility > 0.3 or max_drawdown < -0.12:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "resolved_ticker": yf_ticker,
            "volatility_annualized": round(volatility, 4),
            "max_drawdown": round(max_drawdown, 4),
            "beta_vs_benchmark": beta,
            "risk_level": risk_level,
            "daily_return_std": round(float(returns.std()), 6) if len(returns) > 1 else None,
            "period_return": round(float((close.iloc[-1] / close.iloc[0] - 1) * 100), 2) if len(close) > 1 else None,
        }

    except Exception as e:
        return {"error": str(e)}

def risk_agent(state: MarketState) -> dict[str, Any]:
    ticker: str = state["ticker"]
    asset_type: str = state["asset_type"]

    raw_data = _compute_risk_metrics(ticker, asset_type)

    if "error" in raw_data:
        return {
            "agent_signals": [
                build_signal(
                    agent_name="Risk Agent",
                    signal="NEUTRAL",
                    confidence=0.0,
                    summary=f"Could not compute risk: {raw_data['error']}",
                    raw_data=raw_data,
                )
            ]
        }

    data_summary = "\n".join(f"  {k}: {v}" for k, v in raw_data.items())

    prompt = f"""You are an Indian financial risk analyst. Analyze the following risk metrics for {ticker} against the NIFTY 50 baseline:

{data_summary}

Consider the Indian macroeconomic environment (RBI Repo rates, FII/DII inflows).
Translate your risk assessment into a trading signal:
- LOW risk + positive returns → BULLISH
- HIGH risk + negative returns → BEARISH
- If high risk but stable returns -> NEUTRAL

Respond with ONLY a JSON object:
{{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "short explanation"}}"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        parsed = safe_parse_json(response.content)

        return {
            "agent_signals": [
                build_signal(
                    agent_name="Risk Agent",
                    signal=parsed.get("signal", "NEUTRAL"),
                    confidence=float(parsed.get("confidence", 0.5)),
                    summary=parsed.get("summary", "Risk analysis complete."),
                    raw_data=raw_data,
                )
            ]
        }

    except Exception as e:
        return {
            "agent_signals": [
                build_signal(
                    agent_name="Risk Agent",
                    signal="NEUTRAL",
                    confidence=0.0,
                    summary=f"Error during India risk analysis: {e}",
                    raw_data={"error": str(e)},
                )
            ]
        }
