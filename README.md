# JuniorStock

**Sovereign Edge-Native Multi-Agent Quantitative Trading Orchestrator**

Production-grade package for autonomous trading desks on Apple Silicon and future NVIDIA clusters.

## Current Architecture (V6.1)

- `src/juniorstock/engines/swarm/consensus_graph.py` — Full multi-agent supervisor with MLX vectorized TechnicalAnalyst
- `src/juniorstock/engines/swarm/trading_agents.py` — MLX-vectorized manifold audits + ConsensusOrchestrator
- `src/juniorstock/engine/orchestrator.py` — Core LangGraph + Ray + Placement Groups

## Black Box Strategy

Core orchestration is stable. Proprietary logic (Freedman ladder, physics-based inference, optimized execution) lives in clearly marked extension points.

We move fast on main without fragmenting into version forks.

## Quick Start

```bash
git clone https://github.com/cloudcover95/JuniorStock.git
cd JuniorStock
pip install -e "[cluster]"
python -m src.juniorstock.engines.swarm.consensus_graph
```

Built for power users who want Bloomberg-level intelligence without the cloud lock-in.