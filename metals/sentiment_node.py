# juniorstock/metals/sentiment_node.py
import mlx.core as mx
from typing import List
from juniorstock.nodes.asset_node import UniversalAssetNode

class MetalsSentimentNode(UniversalAssetNode):
    """
    Specialized UniversalAssetNode for XAU/XAG order flow.
    Ingests macro-economic sentiment factors alongside standard limit order depth.
    """
    def __init__(self, asset_id: str = "XAU_USD"):
        super().__init__(asset_id)
        self.sentiment_buffer = []

    def ingest_metals_tick(self, price: float, volume: float, dxy_correlation: float, sentiment_index: float):
        """
        Constructs the 4D input vector specifically tuned for precious metals.
        Incorporates Dollar Index (DXY) inverse correlation and NLP sentiment indices.
        """
        vector = [price, volume, dxy_correlation, sentiment_index]
        self.stream_buffer.append(vector)
        
    def extract_tensor(self) -> mx.array:
        """
        Extracts the buffered data as an MLX array for SVD ingestion.
        """
        raw_array = mx.array(self.stream_buffer)
        self.stream_buffer = [] 
        return raw_array
