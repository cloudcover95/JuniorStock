# juniorstock/nodes/ingestion/l2_compressor.py
import mlx.core as mx
from juniorstock.quant.calibration.dynamic_gamma import DynamicGammaCalibrator

class L2DeltaCompressor:
    """
    Starlink bandwidth optimizer. Ingests dense orderbook streams and isolates
    only the tensors that trigger a ternary state change.
    Drops duplicate/flat orderbook states at the hardware ingestion layer.
    """
    def __init__(self, tensor_dim: int = 10):
        self.calibrator = DynamicGammaCalibrator(decay_factor=0.95)
        self.previous_ternary_state = mx.zeros((1, tensor_dim), dtype=mx.int8)

    def process_tick(self, raw_l2_tensor: mx.array) -> tuple:
        """
        Returns a boolean flag indicating a required manifold update, 
        plus the compressed b1.58 tensor.
        """
        if raw_l2_tensor.size == 0:
            return False, mx.array([])

        # 1. Update dynamic quantizer threshold based on new L2 depth
        self.calibrator.update_gamma(raw_l2_tensor)
        
        # 2. Extract instantaneous ternary state
        current_ternary = mx.astype(self.calibrator.recalibrate_tensor(raw_l2_tensor), mx.int8)
        
        # 3. Fast bitwise XOR to detect delta
        # Since state is -1, 0, 1 mapped in int8, unequal states yield non-zero diff
        delta_matrix = current_ternary != self.previous_ternary_state
        
        if mx.any(delta_matrix):
            # State mutation detected. Pass to OmniMath kernel.
            self.previous_ternary_state = current_ternary
            return True, current_ternary
            
        # Topologically flat tick. Drop to conserve thermal mass and memory bandwidth.
        return False, mx.array([])
