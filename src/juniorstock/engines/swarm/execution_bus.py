# path: src/juniorstock/engines/swarm/execution_bus.py
import json
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class SovereignExecutionBus:
    """
    V6.4: Low-Latency Sovereign Execution Bus.

    - Evaluates execution friction (tax drag, slippage)
    - Dispatches hardware macros via Unix Domain Socket (crispy-mouse)
    - Writes high-density monthly Parquet telemetry ledgers
    """

    def __init__(
        self,
        workspace_root: str | None = None,
        socket_path: str = "/tmp/crispy_mouse_gateway.sock",
        stcg_drag: float = 0.32,
        max_slippage_tolerance: float = 0.005,
    ):
        if workspace_root:
            self.root = Path(workspace_root)
        else:
            self.root = Path.home() / "JuniorCloud" / "juniorstock"

        self.vault_dir = self.root / "vault" / "global_telemetry"
        self.vault_dir.mkdir(parents=True, exist_ok=True)

        self.socket_path = socket_path
        self.stcg_drag = stcg_drag
        self.max_slippage_tolerance = max_slippage_tolerance

    def process_execution_payload(
        self, ticker: str, consensus_log: Dict[str, Any], risk_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        action = consensus_log.get("action_proposal", "HOLD")
        execute_gate = consensus_log.get("execute_gate", False)

        if action == "HOLD" or not execute_gate:
            return {"status": "SKIPPED", "reason": "Execution gate locked or neutral consensus."}

        start_time = time.time()
        raw_allocation = float(consensus_log.get("allocation_ratio", 0.10))
        net_allocation = raw_allocation * (1.0 - self.stcg_drag)

        execution_metrics: Dict[str, Any] = {
            "timestamp": time.time(),
            "ticker": ticker,
            "action": action,
            "consensus_score": float(consensus_log.get("consensus_score", 0.5)),
            "net_allocation_pct": float(net_allocation),
            "latency_ms": 0.0,
        }

        # Dispatch to hardware via Unix Domain Socket
        ipc_status = self._dispatch_hardware_frame(ticker, action, net_allocation)
        execution_metrics["hardware_ipc_status"] = ipc_status

        # Write high-density telemetry
        execution_metrics["latency_ms"] = float((time.time() - start_time) * 1000)
        self._append_to_parquet_ledger(execution_metrics)

        return execution_metrics

    def _dispatch_hardware_frame(self, ticker: str, action: str, allocation: float) -> str:
        if not os.path.exists(self.socket_path):
            logging.warning("crispy-mouse socket not found. Falling back.")
            return "SOCKET_OFFLINE_FALLBACK"

        payload = json.dumps({"ticker": ticker, "action": action, "weight": allocation})
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(self.socket_path)
                client.sendall(payload.encode("utf-8"))
            return "SOCKET_DISPATCH_SUCCESS"
        except Exception as e:
            logging.error(f"IPC error: {e}")
            return "SOCKET_FAILED"

    def _append_to_parquet_ledger(self, metrics: Dict[str, Any]) -> None:
        month_str = time.strftime("%Y_%m")
        file_path = self.vault_dir / f"execution_history_{month_str}.parquet"

        # Create single-row table
        table = pa.Table.from_pydict({k: [v] for k, v in metrics.items()})

        try:
            if not file_path.exists():
                pq.write_table(table, file_path, compression="ZSTD")
            else:
                # Efficient append using ParquetWriter
                with pq.ParquetWriter(file_path, table.schema, compression="ZSTD") as writer:
                    writer.write_table(table)
        except Exception as e:
            logging.error(f"Parquet write error: {e}")
