# juniorstock/quant/bitnet/quantize.py
import mlx.core as mx

class BitNetQuantizer:
    """
    Implements the BitNet 1.58b weight and activation quantization logic.
    Replaces traditional floating-point network state with ternary {-1, 0, 1} bounds.
    """
    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def quantize_weights(self, weights: mx.array) -> mx.array:
        """
        AbsMean weight quantization.
        Scales weights by the mean of their absolute values, rounds to nearest integer,
        and clamps strictly to [-1, 1].
        """
        # Calculate absolute mean gamma
        gamma = mx.mean(mx.abs(weights))
        
        # Scale and round
        scaled_weights = weights / (gamma + self.eps)
        ternary_weights = mx.round(scaled_weights)
        
        # Clamp to {-1, 0, 1} to prevent precision overflow
        return mx.clip(ternary_weights, -1.0, 1.0)

    def quantize_activations(self, activations: mx.array, bits: int = 8) -> tuple:
        """
        Quantizes input manifolds/activations to int8 bounds.
        Returns the quantized tensor and the de-quantization scale factor.
        """
        q_max = (2 ** (bits - 1)) - 1
        
        # Absmax scaling for activations
        beta = mx.max(mx.abs(activations))
        scale = beta / q_max
        
        scaled_acts = mx.round(activations / (scale + self.eps))
        q_acts = mx.clip(scaled_acts, -q_max, q_max)
        
        return q_acts, scale
