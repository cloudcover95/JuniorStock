# JuniorStock

**Sovereign Edge-Native Multi-Agent Quantitative Trading Orchestrator (V6.3)**

## Blackbox Integration (V1.0)

JuniorStock can now consume the isolated **BitNet-mlx Proprietary Blackbox**:

- Run the blackbox: `python run_offline_blackbox.py` (in BitNet-mlx repo)
- It listens on `http://127.0.0.1:8001`
- Use `BitNetBlackboxClient` for air-gapped inference

This maintains strict separation between high-level orchestration and low-level proprietary math.

All work on `main`.