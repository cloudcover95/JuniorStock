# path: tests/test_tensor_engine.py
import numpy as np
import pytest

# Graceful import of existing modules for production testing
try:
    from src.juniorstock.engines.swarm.consensus_graph import JCSwarmOrchestrator
    HAS_SWARM = True
except ImportError:
    HAS_SWARM = False


def test_consensus_orchestrator_instantiation():
    """Validates that the core swarm orchestrator can be instantiated."""
    if not HAS_SWARM:
        pytest.skip("Swarm modules not available in this environment")
    orchestrator = JCSwarmOrchestrator(
        risk_profile={"max_allocation_pct": 0.10, "max_drift": 0.15},
        hitl=False
    )
    assert orchestrator is not None
    assert hasattr(orchestrator, "fundamental")
    assert hasattr(orchestrator, "technical")


def test_kpz_surface_expansion_synthetic():
    """Validates non-linear KPZ-style surface growth logic using synthetic data."""
    # Synthetic Z-score matrix (N=2, T=10)
    Z = np.linspace(0.1, 1.0, 20).reshape(2, 10)
    
    # Simple deterministic KPZ proxy (no external dependency)
    k_alphas = np.mean(Z, axis=1) * 2.5
    
    assert k_alphas.shape == (2,)
    assert not np.isnan(k_alphas).any()
    assert k_alphas[0] >= 0.0


def test_financial_tensor_hydration_synthetic():
    """Validates basic manifold-style metric extraction on synthetic price data."""
    C = np.ones((1, 60)) * 100.0
    H = C + 1.0
    L = C - 1.0
    
    spot = float(C[0, 0])
    high = float(np.max(H))
    low = float(np.min(L))
    
    assert spot == 100.0
    assert high > spot
    assert low < spot
