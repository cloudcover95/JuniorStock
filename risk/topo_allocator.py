# juniorstock/risk/topo_allocator.py
import mlx.core as mx

class TopologicalRiskAllocator:
    """
    Capital allocation matrix. Evaluates multi-asset feature disagreement scores
    and outputs b1.58 ternary capital routing signals (-1: Short, 0: Hold, 1: Long).
    Strictly avoids scalar loops for optimal M4/M1 SoC throughput.
    """
    def __init__(self, risk_threshold: float = 0.75):
        self.risk_threshold = risk_threshold

    def calculate_capital_weights(self, signal_tensor: mx.array) -> mx.array:
        """
        Takes an N-dimensional signal tensor from the ATML forward pass.
        Quantizes risk weights to b1.58 bounds.
        """
        if signal_tensor.size == 0:
            return mx.array([])

        # Absolute signal magnitude
        magnitude = mx.abs(signal_tensor)
        
        # Ternary Quantization Gate (W = sgn(S - tau))
        # 1.0 for capital injection, -1.0 for liquidity extraction, 0.0 for static state
        weights = mx.where(
            signal_tensor > self.risk_threshold,
            1.0,
            mx.where(signal_tensor < -self.risk_threshold, -1.0, 0.0)
        )
        
        # Verify weight distribution prevents hyper-leverage
        exposure_norm = mx.sum(mx.abs(weights))
        if exposure_norm > 10.0:  # Hard-coded max concurrent node execution limit
            print(f"[RISK FAULT] Exposure manifold exceeds limits (Norm: {exposure_norm.item()}). Damping required.")
            weights = mx.zeros_like(weights)

        return weights
