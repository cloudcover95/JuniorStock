# juniorstock/blackbox/strategy_container.py
import mlx.core as mx
from juniorstock.quant.atml_core import AdaptiveTensorModulationLoop
from juniorstock.math.omni_math import SovereignOmniMath

class BlackBoxStrategyContainer:
    """
    Encapsulates alpha-generation logic to prevent runtime state leakage.
    Fuses MLX operations into a single compiled graph, pushing evaluation 
    directly to the Metal backend without Python interpreter introspection.
    """
    def __init__(self, svd_components: int = 5, threshold: float = 0.65):
        self.math_kernel = SovereignOmniMath(k_components=svd_components)
        self.atml = AdaptiveTensorModulationLoop(threshold=threshold)
        
        # Compile the forward pass to prevent intermediate tensor exposure
        self._fused_alpha_generation = mx.compile(self._raw_alpha_generation)

    def _raw_alpha_generation(self, raw_tensor: mx.array, dummy_weights: mx.array) -> mx.array:
        """
        Internal, uncompiled logic chain. Must remain private.
        """
        # Compress and project
        manifold = self.math_kernel.bit_drift_svd(raw_tensor)
        # B1.58 Ternary Optimization
        signal = self.atml.forward_pass(manifold, dummy_weights)
        return signal

    def execute_blind_pass(self, tick_tensor: mx.array) -> mx.array:
        """
        Public interface for the Event Router.
        Executes the compiled Metal graph. No intermediate states are returned.
        """
        if tick_tensor.size == 0:
            return mx.array([0.0])
            
        # Deterministic weight alignment for the compiled pass
        weights_shape = (tick_tensor.shape[0], self.math_kernel.k)
        internal_weights = mx.ones(weights_shape)
        
        # Execute fused ops
        alpha_signal = self._fused_alpha_generation(tick_tensor, internal_weights)
        return mx.mean(mx.abs(alpha_signal))
