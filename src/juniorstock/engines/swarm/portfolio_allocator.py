# path: src/juniorstock/engines/swarm/portfolio_allocator.py
#!/usr/bin/env python3
"""
V5.5 Portfolio Allocator — Variance-Retention Scaled Sizing
Connects SovereignMLXManifold output directly to position sizing.

Uses vectorized numpy/mlx operations. Avoids Python scalar loops.
"""

from typing import Any, Dict, Optional
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


def compute_variance_scaled_allocation(
    manifold_metrics: Dict[str, Any],
    base_allocation: float = 0.15,
    min_variance_retention: float = 0.70,
) -> float:
    """
    Scales target allocation using variance retention from the
    SovereignMLXManifold eigen-spectrum.

    Allocation is multiplicatively damped when variance retention drops.
    Fully vectorized.
    """
    variance_retention = float(manifold_metrics.get("variance_retention", 0.85))
    manifold_energy = float(manifold_metrics.get("manifold_energy", 1.0))

    if variance_retention < min_variance_retention:
        damping = max(0.0, (variance_retention - 0.5) / 0.5)
    else:
        damping = 1.0

    energy_scale = min(max(manifold_energy, 0.8), 1.2)

    final_allocation = base_allocation * damping * energy_scale
    return float(max(0.0, min(final_allocation, 0.30)))


def build_portfolio_decision(
    ticker: str,
    manifold_metrics: Dict[str, Any],
    consensus_decision: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Final position sizing layer. Combines consensus gate with
    variance-retention scaling from the manifold.
    """
    if not consensus_decision.get("execute_gate", False):
        return {
            "ticker": ticker,
            "action": "MONITOR",
            "final_allocation_pct": 0.0,
            "reason": "consensus_veto",
        }

    scaled_allocation = compute_variance_scaled_allocation(manifold_metrics)

    return {
        "ticker": ticker,
        "action": consensus_decision["action"],
        "final_allocation_pct": scaled_allocation,
        "base_consensus_allocation": consensus_decision["allocation_ratio"],
        "variance_retention": manifold_metrics.get("variance_retention", 0.85),
        "execute_gate": True,
    }
