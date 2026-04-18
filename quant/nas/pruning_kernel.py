# juniorstock/quant/nas/pruning_kernel.py
import mlx.core as mx

class TernaryNAS:
    """
    Self-optimizing Spiking Automata.
    Prunes synapses with low 'Topological Vigor'—the ratio of activation 
    contribution to trade execution probability.
    """
    def __init__(self, spike_history_depth: int = 1000):
        self.history_depth = spike_history_depth
        self.spike_accumulator = None

    def evaluate_and_prune(self, weight_matrix: mx.array, spike_train: mx.array, threshold: float = 0.05):
        """
        Identifies 'Dead Manifold' synapses.
        Weights contributing to less than 5% of execution triggers are zeroed out (ternary 0).
        """
        # Calculate synapse utilization frequency
        # Vectorized L1 contribution
        contribution = mx.abs(weight_matrix * spike_train)
        
        # Calculate mean contribution over the history depth
        avg_vigor = mx.mean(contribution, axis=0)
        
        # Generate pruning mask
        pruning_mask = avg_vigor > threshold
        
        # Apply mask: Force weights to zero if vigor is below threshold
        optimized_weights = mx.where(pruning_mask, weight_matrix, mx.array(0, dtype=mx.int8))
        
        pruned_count = mx.sum(weight_matrix != optimized_weights).item()
        if pruned_count > 0:
            print(f"[T-NAS] Pruned {pruned_count} dead synapses. Architecture stabilized.")
            
        return optimized_weights
