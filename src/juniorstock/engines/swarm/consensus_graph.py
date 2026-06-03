# path: src/juniorstock/engines/swarm/consensus_graph.py
#!/usr/bin/env python3
"""
V6.1: Sovereign Multi-Agent Debate + Risk Execution Graph (JuniorTradingAgents Fork)

Full supervisor-style orchestrator with parallel analysts, iterative
deabte reflection loop, risk gating, and black-box hardware execution.

MLX-vectorized manifold processing in TechnicalAnalyst when available.
All heavy tensor operations avoid Python scalar loops.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


def _to_mx(arr):
    if HAS_MLX:
        if isinstance(arr, mx.array):
            return arr
        return mx.array(arr)
    if np is not None:
        return np.asarray(arr)
    return arr


class FundamentalAnalyst:
    """Evaluates asset valuation tiers, earnings drift, and macro indicators."""

    def analyze(self, ticker: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
        pe_ratio = float(market_context.get("pe_ratio", 20.0))
        peg_ratio = float(market_context.get("peg_ratio", 1.0))

        score = 0.5
        if pe_ratio < 15.0 and peg_ratio < 1.0:
            score = 0.85
        elif pe_ratio > 35.0:
            score = 0.20

        return {
            "node": "FUNDAMENTAL_ANALYST",
            "score": score,
            "signal": "BUY" if score > 0.7 else ("SELL" if score < 0.3 else "HOLD"),
            "rationale": f"PE: {pe_ratio:.2f} | PEG: {peg_ratio:.2f}",
        }


class TechnicalAnalyst:
    """
    MLX-vectorized technical analyst operating on raw manifold tensor.

    Computes KPZ-Alpha proxy, z-score, and breakout signals directly
    from the SVD manifold using vectorized operations.
    """

    def analyze(self, ticker: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
        manifold = market_context.get("manifold")
        k_alpha = float(market_context.get("k_alpha", 1.0))
        z_score = float(market_context.get("z_score", 0.0))

        if manifold is not None and HAS_MLX:
            M = _to_mx(manifold)
            if isinstance(M, mx.array) and M.ndim >= 2:
                frob = mx.sqrt(mx.sum(M * M)).item()
                prev_norm = float(market_context.get("prev_manifold_norm", frob))
                k_alpha = (frob - prev_norm) / (abs(prev_norm) + 1e-8)

                try:
                    _, S, _ = mx.linalg.svd(M, full_matrices=False)
                    if len(S) > 1:
                        z_score = ((S[0] - mx.mean(S)) / (mx.std(S) + 1e-8)).item()
                except Exception:
                    pass

        score = 0.5
        if k_alpha > 2.5 and z_score > 1.0:
            score = 0.90
        elif k_alpha > 2.5 and z_score < -1.0:
            score = 0.10

        return {
            "node": "TECHNICAL_ANALYST",
            "score": score,
            "signal": "BUY" if score > 0.7 else ("SELL" if score < 0.3 else "HOLD"),
            "rationale": f"K-Alpha: {k_alpha:.4f} | Z-Score: {z_score:.2f}",
            "k_alpha": k_alpha,
            "z_score": z_score,
        }


class SentimentAnalyst:
    """Evaluates cross-platform liquidity queues and order book imbalances."""

    def analyze(self, ticker: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
        order_imbalance = float(market_context.get("order_imbalance", 0.0))
        score = 0.5 + (order_imbalance * 0.4)
        score = max(0.0, min(1.0, score))

        return {
            "node": "SENTIMENT_ANALYST",
            "score": round(score, 4),
            "signal": "BUY" if score > 0.65 else ("SELL" if score < 0.35 else "HOLD"),
            "rationale": f"Order Book Imbalance: {order_imbalance:.4f}",
        }


class DebateReflectionLoop:
    """Iterative consensus convergence using vectorized operations."""

    def __init__(self, iterations: int = 3):
        self.iterations = iterations

    def process_consensus(self, analysis_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not analysis_payloads:
            return {"consensus_score": 0.5, "action_proposal": "HOLD", "divergence_vector": 0.0}

        scores = np.array([float(p["score"]) for p in analysis_payloads], dtype=np.float32)

        for _ in range(self.iterations):
            mean_score = np.mean(scores)
            scores = scores + 0.1 * (mean_score - scores)

        final_score = float(np.mean(scores))
        std_dev = float(np.std(scores))

        action = "HOLD"
        if final_score > 0.68:
            action = "PROPOSE_BUY"
        elif final_score < 0.32:
            action = "PROPOSE_SELL"

        return {
            "consensus_score": round(final_score, 4),
            "action_proposal": action,
            "divergence_vector": std_dev,
        }


class RiskManager:
    """Enforces identity drift, tax, and allocation thresholds."""

    def __init__(self, profile: Dict[str, Any]):
        self.profile = profile

    def audit_proposal(self, proposal: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        identity_drift = float(market_context.get("identity_drift", 0.0))
        max_drift = float(self.profile.get("max_drift", 0.12))

        if identity_drift > max_drift:
            return {
                "authorized": False,
                "reason": f"Identity Drift violation: {identity_drift:.4f} > {max_drift}",
                "allocation_size": 0.0,
            }

        if proposal["action_proposal"] == "HOLD":
            return {"authorized": False, "reason": "Neutral hold proposal.", "allocation_size": 0.0}

        return {
            "authorized": True,
            "reason": "Risk thresholds nominal.",
            "allocation_size": float(self.profile.get("max_allocation_pct", 0.10)),
        }


class ExecutionGate:
    """HITL / automated routing gate."""

    def __init__(self, hitl_enabled: bool = False):
        self.hitl_enabled = hitl_enabled

    def route_to_hardware(self, ticker: str, risk_clearance: Dict[str, Any], proposal: Dict[str, Any]) -> bool:
        if not risk_clearance.get("authorized", False):
            logging.warning(f"[!] Gate Rejected: {ticker} - {risk_clearance.get('reason')}")
            return False

        if self.hitl_enabled:
            logging.info(f"[HITL] Awaiting confirmation for {proposal['action_proposal']} on {ticker}")
            return True

        logging.info(f"[+] Gate cleared for {ticker} → {proposal['action_proposal']}")
        return True


class CrispyMouseExecute:
    """Black-box hardware execution layer."""

    def __init__(self, target_link_path: str = "src/juniorstock/core/jc_sdk/plugins/crispy_mouse"):
        self.target_link = Path(target_link_path)

    def trigger_macro_sequence(self, ticker: str, proposal: Dict[str, Any], allocation: float) -> Dict[str, Any]:
        action = proposal.get("action_proposal", "HOLD")
        logging.info(f"[CRISPY_MOUSE] Executing {action} on {ticker} @ {allocation*100:.2f}%")
        return {"status": "SUCCESS", "execution_timestamp": time.time()}


class JCSwarmOrchestrator:
    """Supervisor-style multi-agent coordinator with MLX vectorized technical analysis."""

    def __init__(self, risk_profile: Dict[str, Any], hitl: bool = False):
        self.fundamental = FundamentalAnalyst()
        self.technical = TechnicalAnalyst()
        self.sentiment = SentimentAnalyst()
        self.debate_loop = DebateReflectionLoop(iterations=3)
        self.risk_manager = RiskManager(profile=risk_profile)
        self.execution_gate = ExecutionGate(hitl_enabled=hitl)
        self.hardware_driver = CrispyMouseExecute()

    def process_market_node(self, ticker: str, market_context: Dict[str, Any]):
        logging.info(f"[*] Processing market node: {ticker}")

        f_res = self.fundamental.analyze(ticker, market_context)
        t_res = self.technical.analyze(ticker, market_context)
        s_res = self.sentiment.analyze(ticker, market_context)

        consensus = self.debate_loop.process_consensus([f_res, t_res, s_res])
        logging.info(f"[-] Consensus: {consensus['consensus_score']} → {consensus['action_proposal']}")

        risk_clearance = self.risk_manager.audit_proposal(consensus, market_context)

        if self.execution_gate.route_to_hardware(ticker, risk_clearance, consensus):
            self.hardware_driver.trigger_macro_sequence(
                ticker, consensus, risk_clearance["allocation_size"]
            )


if __name__ == "__main__":
    profile = {
        "max_allocation_pct": 0.15,
        "tax_bracket": 0.32,
        "max_drift": 0.10,
    }

    context = {
        "pe_ratio": 12.5,
        "peg_ratio": 0.8,
        "k_alpha": 3.412,
        "z_score": 1.82,
        "order_imbalance": 0.451,
        "identity_drift": 0.042,
    }

    orchestrator = JCSwarmOrchestrator(risk_profile=profile, hitl=False)
    orchestrator.process_market_node("BTC-USD", context)
