# juniorstock/formatters/topo_report.py
import mlx.core as mx
import json
from typing import List

class TopologicalFormatter:
    """
    Translates raw Betti manifolds and ATML matrices into parsed telemetry.
    Optimized for the iPad WS API and downstream logging.
    """
    @staticmethod
    def format_betti_signature(manifold: mx.array) -> List[int]:
        """
        Extracts Betti persistence from the MLX tensor and formats to standard list.
        """
        # Compress tensor shape to 1D binary signature
        signature = mx.where(manifold > mx.mean(manifold), 1, 0).flatten()
        return signature.tolist()

    @staticmethod
    def generate_audit_report(system_voltage: float, feature_score: float, active_bots: int) -> str:
        """
        Outputs strict JSON telemetry for the Audit UI.
        """
        report = {
            "node_status": "ONLINE",
            "voltage_v": round(system_voltage, 2),
            "feature_disagreement_score": round(feature_score, 4),
            "active_agents": active_bots
        }
        return json.dumps(report)
