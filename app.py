"""
MarketMind AI — Streamlit Dashboard (India Edition)
Premium multi-agent financial analysis interface.
"""
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="MarketMind AI 🇮🇳",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.main-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.main-header p {
    color: #8892a4;
    font-size: 1.05rem;
    font-weight: 400;
}

.verdict-card {
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    margin: 1.5rem 0;
    border: 1px solid rgba(255,255,255,0.08);
}
.verdict-bullish {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.08));
    border-color: rgba(16,185,129,0.3);
}
.verdict-bearish {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border-color: rgba(239,68,68,0.3);
}
.verdict-neutral {
    background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(217,119,6,0.08));
    border-color: rgba(245,158,11,0.3);
}

.verdict-label {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: 2px;
}
.verdict-bullish .verdict-label { color: #10b981; }
.verdict-bearish .verdict-label { color: #ef4444; }
.verdict-neutral .verdict-label { color: #f59e0b; }

.confidence-bar {
    height: 8px;
    border-radius: 4px;
    background: rgba(255,255,255,0.1);
    margin: 1rem auto;
    max-width: 400px;
}
.confidence-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 1s ease;
}

.agent-card {
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 0.8rem;
}

.signal-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
}
.signal-BULLISH { background: rgba(16,185,129,0.2); color: #10b981; }
.signal-BEARISH { background: rgba(239,68,68,0.2); color: #ef4444; }
.signal-NEUTRAL { background: rgba(245,158,11,0.2); color: #f59e0b; }

.stButton > button {
    width: 100%;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.75rem 2rem;
    background: linear-gradient(135deg, #FF9933 0%, #138808 100%);
    color: white;
    border: none;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(19, 136, 8, 0.4);
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="main-header">
    <h1>🧠 MarketMind AI 🇮🇳</h1>
    <p>India Market Focused Multi-Agent Financial Analysis</p>
</div>
""", unsafe_allow_html=True)

st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    st.markdown("---")

    ticker = st.text_input(
        "Ticker Symbol",
        value="RELIANCE",
        placeholder="e.g. RELIANCE, TCS, INFY, HDFCBANK, BTC",
        help="Enter a stock ticker (e.g., RELIANCE, TCS) or crypto symbol (BTC)",
    ).upper().strip()

    st.caption("Examples: `RELIANCE`, `TCS`, `INFY`, `HDFCBANK`, `BTC`")

    asset_type = st.selectbox(
        "Asset Type",
        options=["stock", "crypto"],
        index=0,
        help="Select 'crypto' for cryptocurrencies or 'stock' for Indian equities",
    )

    st.markdown("---")
    st.markdown("""
    **📊 Agents Running:**
    - 📈 Price (NFO/NSE Technicals)
    - 📰 Sentiment (India News via GNews)
    - 🔗 Microstructure (NSE Volume/Beta)
    - 🏛️ Macro (NIFTY/RBI repo)
    - ⚠️ Risk (India Volatility)
    """)

    analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)

if analyze_btn:
    if not ticker:
        st.error("Please enter a ticker symbol.")
        st.stop()

    from graph import build_graph

    st.markdown(f"### Analyzing **{ticker}** as **{asset_type}**…")

    progress = st.progress(0, text="Initialising agents…")

    agent_names = [
        "📈 Price Agent",
        "📰 Sentiment Agent",
        "🔗 On-Chain Agent",
        "🏛️ Macro Agent",
        "⚠️ Risk Agent",
    ]

    try:
        progress.progress(10, text="Compiling LangGraph workflow…")
        graph = build_graph()

        progress.progress(20, text="Running parallel analysis agents…")
        result = graph.invoke(
            {
                "ticker": ticker,
                "asset_type": asset_type,
                "agent_signals": [],
            }
        )
        progress.progress(100, text="✅ Analysis complete!")

    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        st.stop()

    verdict = result.get("final_verdict", "NEUTRAL")
    confidence = result.get("final_confidence", 0.0)
    reasoning = result.get("final_reasoning", "")

    verdict_lower = verdict.lower()
    verdict_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(
        verdict_lower, "⚪"
    )

    conf_pct = int(confidence * 100)
    if verdict_lower == "bullish":
        bar_color = "#10b981"
    elif verdict_lower == "bearish":
        bar_color = "#ef4444"
    else:
        bar_color = "#f59e0b"

    st.markdown(f"""
    <div class="verdict-card verdict-{verdict_lower}">
        <div style="font-size:1rem; color:#8892a4; margin-bottom:0.3rem;">FINAL VERDICT</div>
        <div class="verdict-label">{verdict_emoji} {verdict}</div>
        <div style="font-size:1.2rem; margin-top:0.5rem; color:#c0c8d8;">
            Confidence: <strong>{conf_pct}%</strong>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{conf_pct}%; background:{bar_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if reasoning:
        st.markdown("#### 💡 Reasoning")
        st.info(reasoning)

    st.markdown("---")
    st.markdown("### 🤖 Individual Agent Signals")

    signals = result.get("agent_signals", [])
    emoji_map = {
        "Price Agent": "📈",
        "Sentiment Agent": "📰",
        "On-Chain Agent": "🔗",
        "Macro Agent": "🏛️",
        "Risk Agent": "⚠️",
    }

    cols = st.columns(min(len(signals), 3))

    for i, sig in enumerate(signals):
        col = cols[i % len(cols)]
        with col:
            agent_name = sig["agent"]
            emoji = emoji_map.get(agent_name, "🤖")
            sig_color = {
                "BULLISH": "#10b981",
                "BEARISH": "#ef4444",
                "NEUTRAL": "#f59e0b",
            }.get(sig["signal"], "#8892a4")

            st.markdown(f"""
            <div class="agent-card">
                <div style="font-weight:600; font-size:1rem; margin-bottom:0.5rem;">
                    {emoji} {agent_name}
                </div>
                <span class="signal-badge signal-{sig['signal']}">{sig['signal']}</span>
                <span style="color:#8892a4; margin-left:0.5rem;">
                    {int(sig['confidence']*100)}% confidence
                </span>
                <div style="margin-top:0.6rem; font-size:0.9rem; color:#a0aab8;">
                    {sig['summary']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Detailed Agent Data")

    for sig in signals:
        agent_name = sig["agent"]
        emoji = emoji_map.get(agent_name, "🤖")
        with st.expander(f"{emoji} {agent_name} — Raw Data (Beta)"):
            st.json(sig.get("raw_data", {}))

else:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#8892a4;">
        <div style="font-size:4rem; margin-bottom:1rem;">📊</div>
        <h3 style="color:#c0c8d8;">Ready to Analyze the Indian Market</h3>
        <p>Enter an NSE ticker symbol in the sidebar and click <strong>Analyze</strong> to start.</p>
        <p style="font-size:0.85rem; margin-top:1rem;">
            MarketMind 🇮🇳 runs 5 specialized AI agents trained on Indian Market macros in parallel.
        </p>
    </div>
    """, unsafe_allow_html=True)
