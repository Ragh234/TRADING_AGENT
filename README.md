# MarketMind AI 🇮🇳

[![CI](https://github.com/Ragh234/TRADING_AGENT/actions/workflows/ci.yml/badge.svg)](https://github.com/Ragh234/TRADING_AGENT/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A multi-agent market analysis tool for Indian equities (NSE) and crypto,
built on **LangGraph**. Five independent agents analyze a ticker in
parallel — technicals, news sentiment, market microstructure, macro
conditions, and risk — and a sixth agent synthesizes their signals into a
single verdict with a confidence score and plain-language reasoning.

> **This is an educational/portfolio project, not investment advice.**
> Signals are LLM-generated opinions over public market data and should
> not be used to make real trading decisions.

## Why "MarketMind" in a repo called `TRADING_AGENT`

The repo predates the product name — `MarketMind AI` is what the Streamlit
app calls itself. Left the repo name as-is rather than rename it out from
under an existing GitHub history.

## Architecture

`graph.py` builds a LangGraph `StateGraph` that fans out from `__start__`
into five agents running independently, then fans back into one synthesis
node:

```text
                 ┌─────────────┐
        ┌───────▶│ Price Agent │────┐
        │        └─────────────┘    │
        │        ┌─────────────┐    │
        ├───────▶│ Sentiment   │────┤
        │        └─────────────┘    │
__start__──────▶┌─────────────┐    ├───────▶ Synthesis Agent ──▶ END
        │        │ On-Chain    │────┤        (verdict + confidence
        │        └─────────────┘    │         + reasoning)
        │        ┌─────────────┐    │
        ├───────▶│ Macro       │────┤
        │        └─────────────┘    │
        │        ┌─────────────┐    │
        └───────▶│ Risk        │────┘
                 └─────────────┘
```

Each agent returns a partial state update — `{"agent_signals": [...]}` —
and `MarketState.agent_signals` is declared with `Annotated[list, operator.add]`
(see [`state.py`](state.py)), so LangGraph merges the five parallel results
by concatenation instead of one overwriting another.

## What each agent does

| Agent | File | Data source | Signal basis |
|---|---|---|---|
| Price | [`agents/price_agent.py`](agents/price_agent.py) | yfinance, 90-day daily bars | RSI(14), MACD, Bollinger Bands, 7-day % change |
| Sentiment | [`agents/sentiment_agent.py`](agents/sentiment_agent.py) | GNews API, `country=in` | LLM classification of recent headlines |
| On-Chain / Microstructure | [`agents/onchain_agent.py`](agents/onchain_agent.py) | CoinGecko (crypto) or yfinance (stocks) | Crypto: market cap rank, volume, 24h/7d change. Stocks: volume ratio, beta, P/E |
| Macro | [`agents/macro_agent.py`](agents/macro_agent.py) | yfinance (`^NSEI`, `^NSEBANK`) | NIFTY 50 / Bank Nifty 7d & 30d trend, plus a fixed RBI repo rate and CPI figure (see Limitations) |
| Risk | [`agents/risk_agent.py`](agents/risk_agent.py) | yfinance, 90-day daily bars | Annualized volatility, max drawdown, beta vs. NIFTY 50/BTC |
| Synthesis | [`agents/synthesis_agent.py`](agents/synthesis_agent.py) | The other five agents' outputs | LLM-weighted vote, biased toward Macro + Sentiment |

Every agent (including Synthesis) calls the same Groq LLM
(`llama-3.3-70b-versatile`, see [`agents/__init__.py`](agents/__init__.py))
to turn its computed metrics into a `{signal, confidence, summary}` JSON
object. `safe_parse_json()` strips markdown code fences and falls back to
regex extraction if the model doesn't return clean JSON.

**If the LLM call fails, the system still produces an answer.** Each agent
catches its own exceptions and returns a `NEUTRAL, confidence=0.0` signal
with the error in `summary` rather than crashing the graph. If the
*Synthesis* agent's own LLM call fails, it falls back to a plain majority
vote across the other agents' signals (see the `except` branch in
`synthesis_agent.py`).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # then fill in your keys
streamlit run app.py
```

### Environment variables (`.env.example`)

| Variable | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | Yes | Every agent's LLM call goes through Groq. Get one at [console.groq.com/keys](https://console.groq.com/keys). |
| `GNEWS_API_KEY` | No | Powers the Sentiment Agent's headline fetch. If unset, that agent runs with zero headlines and the LLM still returns a signal rather than failing. |

## Demo

_Screenshot goes here — the most representative view is the sidebar with
a ticker analyzed and the verdict card + per-agent signal cards visible
below it (that's the whole product in one screen). A second optional
screenshot: one agent's "Raw Data" expander open, showing the computed
metrics behind its signal. Add images under a `docs/images/` folder and
reference them here, e.g. `![Verdict view](docs/images/verdict-view.png)`._

## Usage

In the sidebar, enter a ticker (`RELIANCE`, `TCS`, `INFY`, `HDFCBANK`, ... or
`BTC` for crypto), pick asset type, and click **Analyze**. Indian stock
tickers are auto-suffixed with `.NS` for Yahoo Finance
(`format_india_ticker()` in `agents/__init__.py`).

You can also run the graph directly, without Streamlit:

```bash
python graph.py
```

## Limitations

- **Macro data is partly hardcoded.** `macro_agent.py` fetches live NIFTY 50
  / Bank Nifty trends from yfinance, but the RBI repo rate (6.50%) and CPI
  (5.09%) are fixed constants — a comment in the source explains this was a
  scope call for a hackathon timeline, since a reliable free real-time RBI
  API wasn't available. Treat those two figures as illustrative, not live.
- Every agent depends on a live third-party API (yfinance, GNews,
  CoinGecko, Groq); rate limits or an unlisted ticker will degrade a given
  agent to its `NEUTRAL / 0.0 confidence` fallback rather than error out.
- No backtesting or historical accuracy tracking — verdicts are not scored
  against what actually happened afterward.

## Tech stack

LangGraph · LangChain (Groq) · Streamlit · yfinance · pycoingecko ·
pandas / numpy

## Author

Raghav Malani
