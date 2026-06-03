# JuniorStock

**Sovereign Edge-Native Multi-Agent Quantitative Trading Orchestrator (V6.2 Complete)**

## Ecosystem Map (JuniorCloud LLC)

JuniorStock is the central quant trading hub. It integrates with:

- **BitNet-mlx** → Local 1.58-bit inference (used in Cognitive Bridge)
- **crispy-mouse** → Deterministic low-level execution macros
- **stocksnode** → Data lake & telemetry
- **JuniorMemSys-Suite** → Long-term topological memory
- **JuniorQuant** → Low-power manifold math
- **JuniorHome** → Central edge orchestrator (future)
- **web3node** → On-chain signals

All repos under `cloudcover95` / JuniorCloud LLC.

## V6.2 Highlights
- BitNet-MLX Cognitive Bridge (sovereign LLM reasoning + Obsidian logs)
- Production Swarm Daemon (infinite loop)
- MLX-vectorized manifold audits
- Full dependency wiring with graceful fallbacks

## Quick Start

```bash
git clone https://github.com/cloudcover95/JuniorStock.git
cd JuniorStock
pip install -e ".[full]"   # pulls real ecosystem packages when available
python run_swarm_daemon.py
```

We build fast on `main`.