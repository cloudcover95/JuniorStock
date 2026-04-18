# juniorstock/pipelines/historical_mesh_replay.py
import mlx.core as mx
import pyarrow.parquet as pq
from juniorstock.quant.atml_core import AdaptiveTensorModulationLoop

class MeshSimulationEngine:
    """
    Offline historical replay via MLX.
    Ingests Parquet compressed manifolds, bypassing heavy I/O for zero-bloat backtesting.
    """
    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        self.atml = AdaptiveTensorModulationLoop(threshold=0.65)
        self.dataset = pq.ParquetDataset(self.parquet_path)

    def execute_replay(self, batch_size: int = 1000):
        """
        Vectorized ingestion of TS data arrays for backtesting b1.58 loops.
        Strictly avoids scalar iterations.
        """
        table = self.dataset.read()
        
        # Extract Betti manifolds directly to numpy, then cast to MLX array
        manifolds_np = table.column('betti_manifold').to_numpy()
        
        # Process in parallel tensor batches
        for i in range(0, len(manifolds_np), batch_size):
            batch = manifolds_np[i:i+batch_size]
            batch_tensor = mx.array(batch.tolist())
            
            # Run dummy predictive weights for the simulation pass
            prediction_weights = mx.random.normal(batch_tensor.shape)
            
            # hardware-accelerated b1.58 loop testing
            optimized_signal = self.atml.forward_pass(batch_tensor, prediction_weights)
            
            # Root node failure check on divergence
            feature_disagreement_score = mx.mean(mx.abs(optimized_signal))
            if mx.isnan(feature_disagreement_score):
                raise ValueError(f"Tensor divergence detected at batch index {i}. Verify SVD dimensions.")
