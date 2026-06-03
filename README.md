# JuniorStock

**Sovereign Edge-Native Multi-Agent Quantitative Trading Orchestrator (V6.4)**

## V6.4: SovereignExecutionBus

New low-latency backend execution port:

- Friction evaluation (tax drag, slippage)
- Unix Domain Socket dispatch to `crispy-mouse`
- Monthly Parquet telemetry ledgers

Fully integrated into the main `SwarmDaemon` loop.

All work consolidated on `main`.