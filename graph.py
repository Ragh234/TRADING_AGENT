"""
MarketMind AI — LangGraph Workflow (India Edition)
Fan-out parallel agents → merge → synthesis → END
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.macro_agent import macro_agent
from agents.onchain_agent import onchain_agent
from agents.price_agent import price_agent
from agents.risk_agent import risk_agent
from agents.sentiment_agent import sentiment_agent
from agents.synthesis_agent import synthesis_agent
from state import MarketState

def build_graph() -> StateGraph:
    """Construct and compile the MarketMind analysis graph."""
    graph = StateGraph(MarketState)

    graph.add_node("price_agent", price_agent)
    graph.add_node("sentiment_agent", sentiment_agent)
    graph.add_node("onchain_agent", onchain_agent)
    graph.add_node("macro_agent", macro_agent)
    graph.add_node("risk_agent", risk_agent)
    graph.add_node("synthesis_agent", synthesis_agent)

    graph.add_edge("__start__", "price_agent")
    graph.add_edge("__start__", "sentiment_agent")
    graph.add_edge("__start__", "onchain_agent")
    graph.add_edge("__start__", "macro_agent")
    graph.add_edge("__start__", "risk_agent")

    graph.add_edge("price_agent", "synthesis_agent")
    graph.add_edge("sentiment_agent", "synthesis_agent")
    graph.add_edge("onchain_agent", "synthesis_agent")
    graph.add_edge("macro_agent", "synthesis_agent")
    graph.add_edge("risk_agent", "synthesis_agent")

    graph.add_edge("synthesis_agent", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke(
        {
            "ticker": "RELIANCE",
            "asset_type": "stock",
            "agent_signals": [],
        }
    )
    print("=== Final Verdict ===")
    print(f"Signal  : {result.get('final_verdict')}")
    print(f"Confidence: {result.get('final_confidence')}")
    print(f"Reasoning : {result.get('final_reasoning')}")
