# juniorstock/pipelines/market_mesh_harvester.py
import mlx.core as mx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
from typing import List
from juniorstock.nodes.asset_node import UniversalAssetNode
from juniorstock.math.omni_math import SovereignOmniMath

class TSDataHarvester:
    """
    High-density TS data harvester.
    Ingests continuous streams, applies Topological SVD, and flushes to .parquet.
    Strictly isolated from 01_Legal and 02_Assets.
    """
    def __init__(self, buffer_size: int = 10000, storage_path: str = "juniorstock/pipelines/mesh_archive.parquet"):
        self.buffer_size = buffer_size
        self.storage_path = storage_path
        self.math_kernel = SovereignOmniMath()
        self._cache = []

    def ingest_node_stream(self, node: UniversalAssetNode):
        """
        Extracts normalized MLX tensor from node, compresses via Bit Drift SVD.
        """
        raw_tensor = node.extract_tensor()
        if raw_tensor.size == 0:
            return

        compressed_manifold = self.math_kernel.bit_drift_svd(raw_tensor)
        
        # Convert MLX array to numpy for Parquet serialization
        np_manifold = mx.array(compressed_manifold).tolist()
        
        self._cache.append({
            "timestamp": time.time_ns(),
            "asset_id": node.asset_id,
            "betti_manifold": np_manifold
        })

        if len(self._cache) >= self.buffer_size:
            self.flush_to_parquet()

    def flush_to_parquet(self):
        """
        Zero-bloat memory flush to physical storage.
        """
        df = pd.DataFrame(self._cache)
        table = pa.Table.from_pandas(df)
        
        try:
            pq.write_to_dataset(table, root_path=self.storage_path)
            self._cache.clear()
        except Exception as e:
            # Immediate root node identification
            print(f"[HARVESTER ERR] Parquet I/O flush failed. Dimension/Memory fault: {e}")
