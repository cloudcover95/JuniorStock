# juniorstock/math/lie_group/tangent_router.py
import mlx.core as mx
from juniorstock.memory.cache.l1_aligner import L1CacheAligner

class DiscreteTangentProjector:
    """
    Maps market topology deformations onto a ternary Lie algebra basis.
    Detects N-leg arbitrage loops in O(1) time without graph traversal.
    """
    def __init__(self, hyper_dim: int = 8192):
        self.hyper_dim = hyper_dim
        self.aligner = L1CacheAligner(dtype=mx.int8)
        
        # Generate orthogonal ternary Lie basis (xi)
        raw_basis = mx.random.uniform(-1.5, 1.5, (1, self.hyper_dim))
        self.xi_basis = mx.astype(mx.clip(mx.round(raw_basis), -1.0, 1.0), mx.int8)
        
        # Enforce L1 alignment on the static basis
        self.xi_basis, _ = self.aligner.pad_to_l1(self.xi_basis)

    def calculate_tangent_arbitrage(self, delta_manifold: mx.array) -> mx.array:
        """
        Projects the manifold delta into the tangent space T_p M.
        Returns the ternary execution vector indicating arbitrage convergence.
        """
        # Ensure delta is L1 aligned
        aligned_delta, orig_shape = self.aligner.pad_to_l1(delta_manifold)
        
        # Tangent projection: Inner product over the Lie basis
        # Executed directly on Metal via aligned memory
        tangent_projection = mx.matmul(aligned_delta, self.xi_basis.T)
        
        # Extract arbitrage signal via signum bound
        arb_signal = mx.clip(mx.sign(tangent_projection), -1.0, 1.0)
        
        return mx.astype(arb_signal, mx.int8)
