# path: src/juniorstock/engines/swarm/trading_agents.py
#!/usr/bin/env python3
"""
V5.5: Sovereign Analytical Consensus (JuniorTradingAgents Fork)
MLX-vectorized manifold audit layer for JuniorStock.

Replaces scalar/vibe-based analyst logic with deterministic,
Apple Silicon vectorized tensor audits on the raw SVD manifold.

All heavy operations use mlx.core when available. Zero scalar Python
loops inside the audit pipeline.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

try:
    import numpy as np
except ImportError:
    np = None


ArrayLike = Union["mx.array", "np.ndarray", list]


def _to_mx(arr: ArrayLike) -> "mx.array":
    """Convert input to mlx array (zero-copy when possible)."""
    if HAS_MLX:
        if isinstance(arr, mx.array):
            return arr
        if isinstance(arr, (list, tuple)):
            return mx.array(arr)
        if np is not None and isinstance(arr, np.ndarray):
            return mx.array(arr)
    if np is not None:
        return np.asarray(arr)
    return arr


class SovereignAnalystNode:
    """Base class for vectorized, math-first analyst nodes."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def audit_state(self, manifold: ArrayLike, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement audit_state")


class KinematicTechnicalAnalyst(SovereignAnalystNode):
    """
    MLX-vectorized technical analyst.

    Operates directly on the raw SVD manifold tensor.
    Computes:
    - Manifold Frobenius norm
    - Approximate surface velocity (K_alpha proxy via trace of squared differences)
    - Eigenvalue spread / condition number
    - Statistical breakout signal
    """

    def audit_state(self, manifold: ArrayLike, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        M = _to_mx(manifold)
        metrics = metrics or {}

        if HAS_MLX and isinstance(M, mx.array):
            frob_norm = mx.sqrt(mx.sum(M * M)).item()
            prev_norm = float(metrics.get("prev_manifold_norm", frob_norm))
            k_alpha_proxy = (frob_norm - prev_norm) / (abs(prev_norm) + 1e-8)

            if M.ndim >= 2:
                try:
                    _, S, _ = mx.linalg.svd(M, full_matrices=False)
                    eigenvalue_spread = (S[0] / (S[-1] + 1e-8)).item() if len(S) > 0 else 1.0
                    z_score_proxy = (S[0] - mx.mean(S)).item() / (mx.std(S).item() + 1e-8)
                except Exception:
                    eigenvalue_spread = 1.0
                    z_score_proxy = 0.0
            else:
                eigenvalue_spread = 1.0
                z_score_proxy = 0.0

            signal = "HOLD"
            if k_alpha_proxy > 0.15 and abs(z_score_proxy) > 1.2:
                signal = "BULLISH_BREAKOUT" if z_score_proxy > 0 else "BEARISH_BREAKOUT"

            confidence = min(max(abs(k_alpha_proxy) * 4.0, 0.0), 1.0)

            return {
                "agent": self.agent_id,
                "signal": signal,
                "confidence": float(confidence),
                "k_alpha": float(k_alpha_proxy),
                "z_score": float(z_score_proxy),
                "manifold_frob_norm": float(frob_norm),
                "eigenvalue_spread": float(eigenvalue_spread),
            }

        # Fallback
        M_np = np.asarray(M) if np is not None else np.array(M)
        frob_norm = float(np.sqrt(np.sum(M_np * M_np)))
        k_alpha_proxy = float(metrics.get("k_alpha", 0.0))
        z_score = float(metrics.get("z_score", 0.0))

        signal = "HOLD"
        if k_alpha_proxy > 2.5 and abs(z_score) > 1.5:
            signal = "BULLISH_BREAKOUT" if z_score > 0 else "BEARISH_BREAKOUT"

        return {
            "agent": self.agent_id,
            "signal": signal,
            "confidence": min(max(k_alpha_proxy / 5.0, 0.0), 1.0),
            "k_alpha": k_alpha_proxy,
            "z_score": z_score,
            "manifold_frob_norm": frob_norm,
        }


class FiscalRiskManager(SovereignAnalystNode):
    """
    MLX-vectorized risk auditor.

    Evaluates structural integrity using matrix norms and drift metrics
    directly on the manifold when available.
    """

    def audit_state(self, manifold: ArrayLike, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        M = _to_mx(manifold)
        metrics = metrics or {}

        identity_drift = float(metrics.get("identity_drift", 0.05))
        net_apy = float(metrics.get("net_apy", 0.0))
        variance_retention = float(metrics.get("variance_retention", 0.85))

        if HAS_MLX and isinstance(M, mx.array) and M.ndim >= 2:
            prev_M = metrics.get("prev_manifold")
            if prev_M is not None:
                prev_M = _to_mx(prev_M)
                drift_norm = mx.sqrt(mx.sum((M - prev_M) ** 2)).item()
                identity_drift = min(drift_norm / (mx.linalg.norm(M).item() + 1e-8), 0.5)

        status = "NOMINAL"
        max_allocation = 0.15

        if identity_drift > 0.15:
            status = "SYSTEMIC_FRACTURE_RISK"
            max_allocation = 0.0
        elif net_apy < 2.0:
            status = "MARGINAL_DRAIN"
            max_allocation = 0.05
        elif variance_retention < 0.70:
            status = "HIGH_VARIANCE_DECAY"
            max_allocation = 0.08

        return {
            "agent": self.agent_id,
            "status": status,
            "max_allocation_pct": max_allocation,
            "identity_drift": identity_drift,
            "net_apy": net_apy,
            "variance_retention": variance_retention,
        }


class ConsensusOrchestrator:
    """
    Deterministic, MLX-accelerated consensus engine.
    """

    def __init__(self, vault_root: Optional[str] = None):
        self.tech_analyst = KinematicTechnicalAnalyst("TECH_ALPHA")
        self.risk_manager = FiscalRiskManager("RISK_OMEGA")
        self.vault_root = (
            Path(vault_root)
            if vault_root
            else Path.home() / "JuniorCloud" / "juniorstock" / "vault" / "obsidian_logs"
        )

    def deliberate(
        self,
        ticker: str,
        manifold: ArrayLike,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metrics = metrics or {}
        tech_report = self.tech_analyst.audit_state(manifold, metrics)
        risk_report = self.risk_manager.audit_state(manifold, metrics)

        execute_trade = False
        action = "MONITOR"

        if tech_report["signal"] == "BULLISH_BREAKOUT" and risk_report["status"] == "NOMINAL":
            execute_trade = True
            action = "EXECUTE_MAKER_LIMIT_BUY"
        elif tech_report["signal"] == "BEARISH_BREAKOUT" and risk_report["status"] == "NOMINAL":
            execute_trade = True
            action = "EXECUTE_MAKER_LIMIT_SELL"

        decision_log: Dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "execute_gate": execute_trade,
            "allocation_ratio": risk_report["max_allocation_pct"],
            "timestamp": time.time(),
            "telemetry": {
                "tech_confidence": tech_report["confidence"],
                "risk_profile": risk_report["status"],
                "k_alpha": tech_report.get("k_alpha", 0.0),
                "identity_drift": risk_report.get("identity_drift", 0.0),
                "manifold_frob_norm": tech_report.get("manifold_frob_norm", 0.0),
            },
        }

        self._write_to_obsidian_vault(ticker, decision_log)
        return decision_log

    def _write_to_obsidian_vault(self, ticker: str, log: Dict[str, Any]) -> None:
        try:
            self.vault_root.mkdir(parents=True, exist_ok=True)
            filename = f"consensus_{ticker}_{int(log['timestamp'])}.md"
            filepath = self.vault_root / filename

            md_content = f"""---
tags: [swarm_debate, junior_agents, {ticker}]
strategy: MLX_VECTORIZED_MANIFOLD_AUDIT
date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log['timestamp']))}
---

# MLX Vectorized Manifold Consensus: {ticker}

## Action: `{log['action']}`
- **Execute:** {log['execute_gate']}
- **Allocation:** {log['allocation_ratio'] * 100:.2f}%

## Telemetry
- Tech Confidence: {log['telemetry']['tech_confidence']:.4f}
- Risk Status: {log['telemetry']['risk_profile']}
- K_alpha: {log['telemetry'].get('k_alpha', 0.0):.4f}
- Identity Drift: {log['telemetry'].get('identity_drift', 0.0):.4f}
- Manifold Frobenius Norm: {log['telemetry'].get('manifold_frob_norm', 0.0):.4f}

```json
{log}
```
"""
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md_content)
        except Exception:
            pass
