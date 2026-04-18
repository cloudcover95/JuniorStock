# juniorstock/quant/hdc/hyper_binder.py
import mlx.core as mx

class HyperDimensionalBinder:
    """
    Maps localized low-dimensional market manifolds into 8192-d ternary hypervectors.
    Enables O(1) cross-asset correlation without O(N^2) attention matrices.
    """
    def __init__(self, target_dimensions: int = 8192):
        self.d = target_dimensions
        # Orthogonal base vectors initialized for standard asset classes
        self.base_vectors = {}

    def _generate_orthogonal_base(self, asset_id: str) -> mx.array:
        """
        Generates a static, pseudo-orthogonal hypervector for a given asset.
        """
        if asset_id not in self.base_vectors:
            # Generate random ternary vector. In high-D space, random vectors are nearly orthogonal.
            raw = mx.random.uniform(-1.5, 1.5, (1, self.d))
            self.base_vectors[asset_id] = mx.clip(mx.round(raw), -1.0, 1.0)
        return self.base_vectors[asset_id]

    def project_and_bind(self, asset_id: str, scalar_velocity: float) -> mx.array:
        """
        Binds a dynamic scalar value (e.g., predicted velocity) to the static asset hypervector.
        Binding in b1.58 space is achieved via scalar expansion and element-wise multiplication.
        """
        base_hv = self._generate_orthogonal_base(asset_id)
        
        # Quantize scalar to ternary bounds
        q_vel = 1.0 if scalar_velocity > 0.5 else (-1.0 if scalar_velocity < -0.5 else 0.0)
        
        return base_hv * q_vel

    def bundle_hypervectors(self, hv_list: list) -> mx.array:
        """
        Bundles multiple bounded hypervectors into a single systemic state vector.
        Bundling is element-wise addition followed by signum projection back to {-1, 0, 1}.
        """
        if not hv_list:
            return mx.zeros((1, self.d))

        # Stack and sum
        stacked = mx.concatenate(hv_list, axis=0)
        superposition = mx.sum(stacked, axis=0, keepdims=True)
        
        # Project back to ternary b1.58 space
        return mx.where(superposition > 0, 1.0, mx.where(superposition < 0, -1.0, 0.0))
