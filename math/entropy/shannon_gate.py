# juniorstock/math/entropy/shannon_gate.py
import mlx.core as mx

class TernaryEntropyGate:
    """
    Calculates the Shannon entropy of a b1.58 compressed manifold.
    Acts as a hardware-level execution block. If the incoming data stream 
    lacks sufficient topological chaos (entropy), the execution graph halts,
    dropping SoC power draw to idle baselines.
    """
    def __init__(self, entropy_threshold: float = 0.85):
        self.tau_entropy = entropy_threshold
        # Pre-allocate tiny epsilon to prevent log2(0) NaN faults natively on Metal
        self.eps = mx.array([1e-9])

    def evaluate_manifold_chaos(self, ternary_tensor: mx.array) -> bool:
        """
        Computes H(M). Returns True if entropy exceeds the execution threshold.
        """
        if ternary_tensor.size == 0:
            return False

        # Flatten manifold for global state distribution analysis
        flat_tensor = mx.flatten(ternary_tensor)
        total_elements = flat_tensor.size

        # Count occurrences of {-1, 0, 1}
        # Vectorized via MLX equality broadcasting
        p_neg = mx.sum(flat_tensor == -1) / total_elements
        p_zero = mx.sum(flat_tensor == 0) / total_elements
        p_pos = mx.sum(flat_tensor == 1) / total_elements

        # Stack probabilities
        probs = mx.stack([p_neg, p_zero, p_pos])
        
        # Filter zero probabilities to avoid log(0)
        safe_probs = mx.where(probs > 0, probs, self.eps)

        # Shannon Entropy calculation natively on Metal SIMD
        entropy = -mx.sum(safe_probs * mx.log2(safe_probs))

        # Root node check
        if entropy > self.tau_entropy:
            return True
            
        return False
