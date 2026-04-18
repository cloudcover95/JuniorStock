# juniorstock/quant/learning/forward_ternary.py
import mlx.core as mx

class TernaryForwardAdapter:
    """
    Executes continuous edge adaptation without backpropagation.
    Utilizes random ternary perturbations to estimate directional gradients,
    maintaining strict b1.58 quantization and bounding thermal draw.
    """
    def __init__(self, learning_rate: float = 1.0):
        # In ternary space, LR is effectively a logical gate (1.0 = full flip allowed)
        self.eta = learning_rate

    def generate_ternary_perturbation(self, shape: tuple) -> mx.array:
        """
        Generates a sparse ternary perturbation matrix V ~ {-1, 0, 1}.
        """
        raw_noise = mx.random.uniform(-1.5, 1.5, shape)
        return mx.clip(mx.round(raw_noise), -1.0, 1.0)

    def compute_ternary_update(self, current_weights: mx.array, base_loss: float, perturbed_loss: float, perturbation: mx.array) -> mx.array:
        """
        Applies the directional derivative update strictly within ternary bounds.
        """
        delta_loss = perturbed_loss - base_loss
        
        # Determine the update direction. If loss decreased (delta < 0), move towards perturbation.
        # If loss increased (delta > 0), move away from perturbation.
        loss_sign = mx.where(delta_loss > 0, 1.0, mx.where(delta_loss < 0, -1.0, 0.0))
        
        # Calculate update matrix
        update = -(self.eta * loss_sign * perturbation)
        
        # Apply update and re-clip to {-1, 0, 1}
        new_weights = mx.clip(current_weights + update, -1.0, 1.0)
        
        return mx.astype(new_weights, mx.int8)

    def calculate_manifold_loss(self, predicted_state: mx.array, actual_state: mx.array) -> float:
        """
        Calculates the Topological Feature Disagreement score (L1 Norm equivalent).
        """
        return mx.mean(mx.abs(predicted_state - actual_state)).item()
