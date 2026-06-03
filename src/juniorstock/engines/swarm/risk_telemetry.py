# path: src/juniorstock/engines/swarm/risk_telemetry.py
#!/usr/bin/env python3
"""
Feature: Risk Telemetry Exporter

Simple JSON + Prometheus-style metrics export for monitoring.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any


class RiskTelemetryExporter:
    def __init__(self, output_dir: str = "data/telemetry"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, ticker: str, risk_data: Dict[str, Any]):
        ts = int(time.time())
        filepath = self.output_dir / f"risk_{ticker}_{ts}.json"
        with open(filepath, "w") as f:
            json.dump({"timestamp": ts, "ticker": ticker, **risk_data}, f, indent=2)

    def to_prometheus(self, risk_data: Dict[str, Any]) -> str:
        lines = []
        for k, v in risk_data.items():
            if isinstance(v, (int, float)):
                lines.append(f"junior_risk_{k} {v}")
        return "\n".join(lines)
