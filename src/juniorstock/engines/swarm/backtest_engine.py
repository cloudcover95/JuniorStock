# path: src/juniorstock/engines/swarm/backtest_engine.py
#!/usr/bin/env python3
"""
Feature: Simple Backtest Engine

Replay historical contexts through the swarm for validation.
"""

from typing import List, Dict, Any

from src.juniorstock.engines.swarm.consensus_graph import JCSwarmOrchestrator


class BacktestEngine:
    def __init__(self):
        self.swarm = JCSwarmOrchestrator(
            risk_profile={"max_allocation_pct": 0.10, "max_drift": 0.15},
            hitl=False
        )

    def run(self, historical_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for ctx in historical_contexts:
            ticker = ctx.get("ticker", "UNKNOWN")
            # Simplified replay
            f = self.swarm.fundamental.analyze(ticker, ctx)
            t = self.swarm.technical.analyze(ticker, ctx)
            s = self.swarm.sentiment.analyze(ticker, ctx)
            consensus = self.swarm.debate_loop.process_consensus([f, t, s])
            results.append({"ticker": ticker, "consensus": consensus})
        return results
