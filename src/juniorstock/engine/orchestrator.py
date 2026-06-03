# path: src/juniorstock/engine/orchestrator.py
#!/usr/bin/env python3
"""
JuniorStock Core Orchestrator

Production-grade LangGraph multi-agent trading desk with Ray + Placement Groups.

Black-box ready for proprietary inference and manifold logic.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    ray = None


class ComputeBackend(str, Enum):
    MLX = "mlx"
    CUDA = "cuda"
    CPU = "cpu"
    AUTO = "auto"


@dataclass
class JuniorStockConfig:
    ticker: str = "EDGE_TDA_NODE"
    backend: ComputeBackend = ComputeBackend.AUTO
    local_only: bool = True
    cluster_enabled: bool = False
    ray_address: Optional[str] = None
    ray_num_gpus: Optional[float] = None
    placement_group_enabled: bool = False
    pg_strategy: str = "PACK"


class AgentState(TypedDict, total=False):
    ticker: str
    tda_manifold: Any
    analyst_reports: Dict[str, str]
    debate_log: List[str]
    final_signal: str
    execution_status: str
    metadata: Dict[str, Any]


def detect_backend(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    if HAS_MLX:
        return "mlx"
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def init_ray_cluster(config: JuniorStockConfig) -> bool:
    if not config.cluster_enabled or not HAS_RAY:
        return False
    try:
        if not ray.is_initialized():
            ray.init(address=config.ray_address, num_gpus=config.ray_num_gpus)
        return True
    except Exception:
        return False


def create_placement_group(config: JuniorStockConfig):
    if not (config.cluster_enabled and config.placement_group_enabled and HAS_RAY):
        return None
    try:
        from ray.util.placement_group import placement_group
        bundles = [{"CPU": 2, "GPU": config.ray_num_gpus or 0.0}]
        pg = placement_group(bundles=bundles, strategy=config.pg_strategy)
        ray.get(pg.ready())
        return pg
    except Exception:
        return None


def technical_analyst_node(state: AgentState, config: JuniorStockConfig) -> AgentState:
    manifold = state.get("tda_manifold")
    if manifold is not None and HAS_MLX and isinstance(manifold, mx.array):
        norm = mx.sqrt(mx.sum(manifold * manifold)).item()
        state["analyst_reports"]["technical"] = f"MLX Manifold norm: {norm:.4f}"
    else:
        state["analyst_reports"]["technical"] = "Manifold received (non-MLX)"
    return state


def debate_node(state: AgentState, config: JuniorStockConfig) -> AgentState:
    tech = state["analyst_reports"].get("technical", "")
    state["debate_log"].append(f"Debate on: {tech[:80]}")
    state["final_signal"] = "EXECUTE_LONG"
    return state


def build_junior_graph(config: Optional[JuniorStockConfig] = None):
    if config is None:
        config = JuniorStockConfig()
    if not HAS_LANGGRAPH:
        raise RuntimeError("langgraph required")

    workflow = StateGraph(AgentState)
    workflow.add_node("technical", lambda s: technical_analyst_node(s, config))
    workflow.add_node("debate", lambda s: debate_node(s, config))
    workflow.set_entry_point("technical")
    workflow.add_edge("technical", "debate")
    workflow.add_edge("debate", END)
    return workflow.compile()


def orchestrate_portfolio_distributed(tickers: List[str], config: Optional[JuniorStockConfig] = None):
    if config is None:
        config = JuniorStockConfig()
    ray_active = init_ray_cluster(config)
    pg = create_placement_group(config)

    results = {}
    if ray_active and HAS_RAY:
        @ray.remote(num_gpus=config.ray_num_gpus or 0.0)
        def _remote_cycle(ticker):
            app = build_junior_graph(config)
            state = {"ticker": ticker, "tda_manifold": None, "analyst_reports": {}, "debate_log": [], "final_signal": "", "execution_status": "", "metadata": {}}
            return app.invoke(state)
        futures = [_remote_cycle.remote(t) for t in tickers]
        for t, fut in zip(tickers, futures):
            results[t] = ray.get(fut)
    else:
        app = build_junior_graph(config)
        for t in tickers:
            state = {"ticker": t, "tda_manifold": None, "analyst_reports": {}, "debate_log": [], "final_signal": "", "execution_status": "", "metadata": {}}
            results[t] = app.invoke(state)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default="EDGE_TDA_NODE")
    parser.add_argument("--cluster", action="store_true")
    args = parser.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",")]
    config = JuniorStockConfig(cluster_enabled=args.cluster)
    results = orchestrate_portfolio_distributed(tickers, config)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
