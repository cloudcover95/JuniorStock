# juniorstock/pipelines/secure_mesh_harvester.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import io
import time
from juniorstock.crypto.state_encryption import SovereignStateCipher
from juniorstock.pipelines.market_mesh_harvester import TSDataHarvester

class SecureTSDataHarvester(TSDataHarvester):
    """
    Overrides the base Harvester to enforce AES-GCM encryption prior to disk flush.
    """
    def __init__(self, buffer_size: int = 10000, storage_path: str = "juniorstock/pipelines/mesh_archive.enc"):
        super().__init__(buffer_size, storage_path)
        self.cipher = SovereignStateCipher()

    def flush_to_parquet(self):
        """
        Intercepts the pyarrow table, serializes to memory, encrypts, and writes binary.
        """
        if not self._cache:
            return

        df = pd.DataFrame(self._cache)
        table = pa.Table.from_pandas(df)
        
        try:
            # Serialize Parquet to in-memory buffer
            sink = pa.BufferOutputStream()
            pq.write_table(table, sink)
            serialized_buffer = sink.getvalue().to_pybytes()
            
            # Apply AES-GCM Transformation
            encrypted_payload = self.cipher.encrypt_manifold_buffer(serialized_buffer)
            
            # Write physical file (Append mode for binary blob logs)
            with open(self.storage_path, "ab") as f:
                f.write(encrypted_payload)
                
            self._cache.clear()
        except Exception as e:
            print(f"[SECURE HARVESTER ERR] Cryptographic flush failed. Dimension/Memory fault: {e}")
