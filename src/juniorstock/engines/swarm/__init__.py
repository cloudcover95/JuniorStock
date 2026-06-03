# path: src/juniorstock/engines/swarm/__init__.py
#!/usr/bin/env python3
"""
JuniorStock Swarm / Multi-Agent Consensus Package (V5.5 / V6.1)

Exposes the deterministic, math-first trading agent graph with MLX support.
"""

from .consensus_graph import JCSwarmOrchestrator
try:
    from .trading_agents import ConsensusOrchestrator as MLXConsensusOrchestrator
except ImportError:
    MLXConsensusOrchestrator = None

__all__ = ["JCSwarmOrchestrator", "MLXConsensusOrchestrator"]
