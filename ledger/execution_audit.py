# juniorstock/ledger/execution_audit.py
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import time
import hashlib
from typing import Dict, Any
from juniorstock.security.path_isolation import EnvironmentIsolationGate

class ExecutionLedger:
    """
    High-density cryptographic ledger for all SDK actions.
    Persists deterministic execution states to .parquet, bypassing standard DB bloat.
    """
    def __init__(self, ledger_path: str = "juniorstock/ledger/audit_trail.parquet"):
        EnvironmentIsolationGate.verify_io_path(ledger_path)
        self.ledger_path = ledger_path
        self._buffer = []
        self.buffer_limit = 500

    def _generate_state_hash(self, payload: Dict[str, Any]) -> str:
        """
        Calculates SHA-256 hash of the execution payload for immutable auditing.
        """
        serialized = str(payload).encode('utf-8')
        return hashlib.sha256(serialized).hexdigest()

    def record_action(self, action_type: str, target: str, payload: Dict[str, Any], status: str):
        """
        Appends execution call to memory buffer.
        """
        record = {
            "timestamp": time.time_ns(),
            "action_type": action_type,
            "target_node": target,
            "status": status,
            "state_hash": self._generate_state_hash(payload)
        }
        self._buffer.append(record)
        
        if len(self._buffer) >= self.buffer_limit:
            self.flush_ledger()

    def flush_ledger(self):
        """
        Zero-bloat memory flush to physical storage via PyArrow.
        """
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer)
        table = pa.Table.from_pandas(df)
        
        try:
            # Append to existing Parquet dataset
            pq.write_to_dataset(table, root_path=self.ledger_path)
            self._buffer.clear()
        except Exception as e:
            print(f"[LEDGER FAULT] Parquet serialization failed. Memory leak averted. ERR: {e}")
            self._buffer.clear() # Force clear to prevent RAM exhaustion
