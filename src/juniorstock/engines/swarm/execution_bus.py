# path: src/juniorstock/engines/swarm/execution_bus.py
import json
import logging
import os
import socket
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class SovereignExecutionBus:
    """
    V6.4: Low-Latency Sovereign Execution Bus with batched writes.

    Features:
    - In-memory buffer to reduce SSD writes
    - Configurable batch size and flush interval
    - Designed for long-running edge deployments
    """

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        socket_path: str = "/tmp/crispy_mouse_gateway.sock",
        stcg_drag: float = 0.32,
        batch_size: int = 50,
        flush_interval_seconds: float = 30.0,
    ):
        if workspace_root:
            self.root = Path(workspace_root)
        else:
            self.root = Path.home() / "JuniorCloud" / "juniorstock"

        self.vault_dir = self.root / "vault" / "global_telemetry"
        self.vault_dir.mkdir(parents=True, exist_ok=True)

        self.socket_path = socket_path
        self.stcg_drag = stcg_drag

        # Batching configuration
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds

        self._buffer: deque = deque()
        self._last_flush_time = time.time()

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

        metrics: Dict[str, Any] = {
            "timestamp": time.time(),
            "ticker": ticker,
            "action": action,
            "consensus_score": float(consensus_log.get("consensus_score", 0.5)),
            "net_allocation_pct": float(net_allocation),
        }

        ipc_status = self._dispatch_hardware_frame(ticker, action, net_allocation)
        metrics["hardware_ipc_status"] = ipc_status
        metrics["latency_ms"] = float((time.time() - start_time) * 1000)

        # Add to buffer instead of immediate write
        self._buffer.append(metrics)

        # Check if we should flush
        self._maybe_flush()

        return metrics

    def _maybe_flush(self):
        now = time.time()
        time_since_flush = now - self._last_flush_time

        if len(self._buffer) >= self.batch_size or time_since_flush >= self.flush_interval_seconds:
            self._flush_buffer()

    def _flush_buffer(self):
        if not self._buffer:
            return

        records = list(self._buffer)
        self._buffer.clear()
        self._last_flush_time = time.time()

        if not records:
            return

        month_str = time.strftime("%Y_%m")
        file_path = self.vault_dir / f"execution_history_{month_str}.parquet"

        table = pa.Table.from_pydict({k: [r[k] for r in records] for k in records[0].keys()})

        try:
            if not file_path.exists():
                pq.write_table(table, file_path, compression="ZSTD")
            else:
                with pq.ParquetWriter(file_path, table.schema, compression="ZSTD") as writer:
                    writer.write_table(table)
            logging.info(f"Flushed {len(records)} execution records to Parquet")
        except Exception as e:
            logging.error(f"Parquet flush error: {e}")
            # Re-queue failed records (simple strategy)
            for r in records:
                self._buffer.appendleft(r)

    def flush(self):
        """Force flush remaining buffer (call on shutdown)."""
        self._flush_buffer()

    def _dispatch_hardware_frame(self, ticker: str, action: str, allocation: float) -> str:
        if not os.path.exists(self.socket_path):
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
