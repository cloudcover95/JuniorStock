# juniorstock/math/fsd/topological_pruner.py
import mlx.core as mx

class BettiPruningGate:
    """
    Dynamically prunes inactive asset streams from the market manifold.
    Reduces the N-dimensional space before it hits the SVD Bit Drift kernel.
    """
    def __init__(self, variance_threshold: float = 1e-4):
        self.tau_p = variance_threshold

    def prune_dead_nodes(self, market_manifold: mx.array) -> tuple:
        """
        Calculates variance per dimension.
        Returns the pruned tensor and the surviving index mask for downstream routing.
        """
        if market_manifold.size == 0:
            return market_manifold, mx.array([])

        # Calculate variance along the time/batch axis (axis 0)
        mean = mx.mean(market_manifold, axis=0)
        variance = mx.mean(mx.square(market_manifold - mean), axis=0)
        
        # Generate boolean mask for active dimensions
        active_mask = variance >= self.tau_p
        
        # Extract surviving dimensions
        # mx.compress equivalent logic for Metal backend optimization
        indices = mx.arange(market_manifold.shape[1])[active_mask]
        
        if indices.size == 0:
            print("[PRUNING GATE] Total manifold collapse. Zero variance detected.")
            return mx.zeros((market_manifold.shape[0], 1)), indices
            
        pruned_manifold = market_manifold[:, indices]
        
        return pruned_manifold, indices
