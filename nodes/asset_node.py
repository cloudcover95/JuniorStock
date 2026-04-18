# juniorstock/nodes/asset_node.py
import mlx.core as mx
from typing import Dict, Any

class UniversalAssetNode:
    """
    Abstracts asset origin. Normalizes ERC-20, AMMs, and Centralized LOBs (FIX/REST)
    into a uniform tensor format for the Omni Math Kernel.
    """
    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        self.stream_buffer = []

    def ingest_tick(self, price: float, volume: float, velocity: float, cross_asset_correlation: float):
        """
        Constructs the 4D input vector for the asset.
        """
        vector = [price, volume, velocity, cross_asset_correlation]
        self.stream_buffer.append(vector)

    def extract_tensor(self) -> mx.array:
        """
        Returns the batch normalized tensor array for MLX ingestion.
        """
        raw_array = mx.array(self.stream_buffer)
        self.stream_buffer = [] # clear buffer post-extraction
        return raw_array
