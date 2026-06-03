# path: src/juniorstock/engines/swarm/manifold_persistence.py
#!/usr/bin/env python3
"""
Feature: Manifold Persistence Layer

Saves and loads MLX/numpy manifolds to Parquet with versioning.
"""

import time
from pathlib import Path
from typing import Any, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


class ManifoldPersistence:
    def __init__(self, base_path: str = "data/manifolds"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, ticker: str, manifold: Any, metadata: dict = None) -> Path:
        if not HAS_PARQUET:
            raise RuntimeError("pyarrow required for persistence")

        ts = int(time.time())
        filepath = self.base_path / f"{ticker}_{ts}.parquet"

        if HAS_MLX and isinstance(manifold, mx.array):
            arr = manifold.numpy() if hasattr(manifold, "numpy") else np.array(manifold)
        else:
            arr = np.asarray(manifold)

        table = pa.Table.from_arrays([pa.array(arr.flatten())], names=["manifold"])
        pq.write_table(table, filepath)
        return filepath

    def load_latest(self, ticker: str):
        files = sorted(self.base_path.glob(f"{ticker}_*.parquet"))
        if not files:
            return None
        table = pq.read_table(files[-1])
        return table.to_pandas().values
