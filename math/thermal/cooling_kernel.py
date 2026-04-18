# juniorstock/math/thermal/cooling_kernel.py
import mlx.core as mx

class ManifoldCoolingKernel:
    """
    Dissipates accumulated topological entropy.
    If the Shannon Entropy Gate (Phase 22) detects stagnation (market stall),
    the cooling kernel resets the manifold to a neutral b1.58 ground state
    to prevent predictive feedback loops and conserve SoC wattage.
    """
    def __init__(self, cooling_factor: float = 0.01):
        self.cooling_factor = cooling_factor

    def apply_cooling(self, manifold: mx.array) -> mx.array:
        """
        Stochastically cools the manifold towards the zero-state.
        """
        noise = mx.random.uniform(-1, 1, manifold.shape)
        # Probabilistic flip towards zero
        cooled_manifold = mx.where(mx.abs(noise) < self.cooling_factor, mx.array(0, dtype=mx.int8), manifold)
        return cooled_manifold
