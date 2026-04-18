# juniorstock/quant/calibration/dynamic_gamma.py
import mlx.core as mx

class DynamicGammaCalibrator:
    """
    Maintains continuous BitNet b1.58 quantization bounds without full fp16 recalculation.
    Utilizes an EMA to smoothly shift the ternary threshold as market volatility changes.
    """
    def __init__(self, decay_factor: float = 0.99, initial_gamma: float = 1.0):
        self.alpha = mx.array([1.0 - decay_factor])
        self.one_minus_alpha = mx.array([decay_factor])
        self.current_gamma = mx.array([initial_gamma])

    def update_gamma(self, current_activation: mx.array) -> mx.array:
        """
        Updates the quantization threshold locally based on the L1 norm of incoming tensors.
        Bypasses standard float scaling, directly utilizing Metal SIMD addition.
        """
        if current_activation.size == 0:
            return self.current_gamma

        # Calculate current tick absolute mean
        tick_gamma = mx.mean(mx.abs(current_activation))
        
        # EMA update
        self.current_gamma = (self.alpha * tick_gamma) + (self.one_minus_alpha * self.current_gamma)
        
        return self.current_gamma

    def recalibrate_tensor(self, tensor: mx.array) -> mx.array:
        """
        Applies the dynamically adjusted gamma to immediately force 
        activations back into the strict {-1, 0, 1} bounds.
        """
        scaled = tensor / (self.current_gamma + 1e-9)
        return mx.clip(mx.round(scaled), -1.0, 1.0)
