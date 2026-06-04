# path: JuniorStock/core/embeddings/liquidity_signatures.py
#!/usr/bin/env python3
"""
JuniorCloud LLC - Sovereign Quant Architecture
Module: Liquidity Signature TDA Embedding
Hardware Target: Apple Silicon (M-Series / Metal)
Dependencies: mlx (Strict vectorization, zero scalar loops)

Integrates with the broader ecosystem:
- Uses same ternary projection philosophy as ManifoldFoldingQuantizer
- Can be called from TriStateExecutionEngine as a domain-specific black box
- Results can be stored in SecondBrainPipeline with TDA metadata
- Supports HighLevelOrchestrator task routing (e.g. "extract liquidity manifold")
"""

import mlx.core as mx
import mlx.nn as nn


class TernaryLiquidityManifold(nn.Module):
    def __init__(self, lookback_window: int = 20, embed_dim: int = 256):
        super().__init__()
        self.lookback = lookback_window
        self.embed_dim = embed_dim
        
        # Dense layer for manifold projection before ternary quantization
        self.projection = nn.Linear(3, embed_dim, bias=False)

    def _ternary_quantize(self, x: mx.array) -> mx.array:
        gamma = mx.mean(mx.abs(x), axis=-1, keepdims=True)
        gamma = mx.where(gamma == 0, mx.array(1e-8), gamma)
        quantized = mx.round(x / gamma)
        return mx.maximum(mx.minimum(quantized, mx.array(1.0)), mx.array(-1.0))

    def compute_turtle_soup_nodes(self, high: mx.array, low: mx.array, close: mx.array) -> mx.array:
        seq_len = close.shape[1]
        
        rolling_high = mx.pad(high, ((0, 0), (self.lookback, 0)))[:, :-self.lookback]
        rolling_low = mx.pad(low, ((0, 0), (self.lookback, 0)))[:, :-self.lookback]

        buyside_sweep = mx.where(high > rolling_high, mx.array(1.0), mx.array(0.0))
        sellside_sweep = mx.where(low < rolling_low, mx.array(-1.0), mx.array(0.0))
        
        midpoint = (rolling_high + rolling_low) / 2.0
        range_expansion = mx.where(close > midpoint, mx.array(1.0), mx.array(-1.0))

        return mx.stack([buyside_sweep, sellside_sweep, range_expansion], axis=-1)

    def __call__(self, high: mx.array, low: mx.array, close: mx.array) -> mx.array:
        kinematic_vectors = self.compute_turtle_soup_nodes(high, low, close)
        projected_manifold = self.projection(kinematic_vectors)
        return self._ternary_quantize(projected_manifold)


def test_deploy_manifold():
    mx.random.seed(42)
    batch_size = 4
    seq_len = 100
    
    high = mx.random.uniform(100, 105, (batch_size, seq_len))
    low = mx.random.uniform(95, 100, (batch_size, seq_len))
    close = mx.random.uniform(96, 104, (batch_size, seq_len))

    manifold = TernaryLiquidityManifold(lookback_window=15, embed_dim=128)
    
    print("[SYSTEM] Compiling Ternary TDA Liquidity Manifold...")
    embedded_signatures = manifold(high, low, close)
    
    print(f"[SUCCESS] Liquidity Manifold Computed. Shape: {embedded_signatures.shape}")
    print(f"[INTEGRITY] Ternary constraint: Max={mx.max(embedded_signatures).item()}, Min={mx.min(embedded_signatures).item()}")


if __name__ == "__main__":
    test_deploy_manifold()
